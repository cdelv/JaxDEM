# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Pure-Python wrapper to train JaxDEM RL environments with PufferLib (4.0+).

Bridges :class:`~jaxdem.rl.env_wrappers.PufferEnvWrapper` (JIT-compiled JAX
environments) directly to PufferLib 4.x's PyTorch models
(``pufferlib.models.Policy``, ``DefaultEncoder``, ``DefaultDecoder``,
``MinGRU``/``MLP``) and optimizer (``pufferlib.muon.Muon`` or Adam).

``pufferlib`` is an optional dependency required only when using this module or
:mod:`jaxdem.rl.trainers.protein_sweep`.
"""

from __future__ import annotations

import datetime
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm.auto import trange  # type: ignore[import-untyped]

from ..action_spaces import ActionSpace
from ..env_wrappers.puffer import PufferEnvWrapper
from ..environments import Environment

_PUFFERLIB_INSTALL_HINT = (
    "PufferLib (>=4.0) is an optional dependency required only by "
    "`jaxdem.rl.trainers.puffer_trainer` and `jaxdem.rl.trainers.protein_sweep`. "
    "Install it with `pip install git+https://github.com/PufferAI/PufferLib.git@4.0` "
    "(or `pip install 'pufferlib>=4.0'`)."
)


class _JaxDemPufferVecEnv:
    """Vectorized environment adapter exposing :class:`PufferEnvWrapper` to PufferLib."""

    def __init__(
        self,
        wrapper: PufferEnvWrapper,
        *,
        clip_actions: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.wrapper = wrapper
        self.num_agents = wrapper.total_agents
        self.act_low = float(clip_range[0]) if clip_actions else -1e6
        self.act_high = float(clip_range[1]) if clip_actions else 1e6

        self._obs_np = np.zeros((wrapper.total_agents, wrapper.obs_size), dtype=np.float32)
        self._rew_np = np.zeros((wrapper.total_agents,), dtype=np.float32)
        self._term_np = np.zeros((wrapper.total_agents,), dtype=np.float32)
        self._trunc_np = np.zeros((wrapper.total_agents,), dtype=np.float32)
        self.env_step_s = 0.0

    def reset_step_timer(self) -> None:
        self.env_step_s = 0.0

    def reset(self) -> None:
        obs, rew, term = self.wrapper.reset()
        self._obs_np = np.array(jax.device_get(obs), dtype=np.float32, copy=True)
        self._rew_np = np.array(jax.device_get(rew), dtype=np.float32, copy=True)
        self._term_np = np.array(jax.device_get(term), dtype=np.float32, copy=True)
        self._trunc_np.fill(0.0)

    def recv(
        self,
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        return self._obs_np, self._rew_np, self._term_np, self._trunc_np

    def send(self, actions: np.ndarray[Any, Any]) -> None:
        t0 = time.perf_counter()
        act_dtype = jnp.int32 if self.wrapper.discrete else jnp.float32
        act_jax = jnp.asarray(actions, dtype=act_dtype)
        obs, rew, term, trunc = self.wrapper.step_split(act_jax)
        self._obs_np = np.array(jax.device_get(obs), dtype=np.float32, copy=True)
        self._rew_np = np.array(jax.device_get(rew), dtype=np.float32, copy=True)
        self._term_np = np.array(jax.device_get(term), dtype=np.float32, copy=True)
        self._trunc_np = np.array(jax.device_get(trunc), dtype=np.float32, copy=True)
        self.env_step_s += time.perf_counter() - t0

    def close(self) -> None:
        return None


def _sample_logits(logits: Any, action: Any | None = None) -> tuple[Any, Any, Any]:
    """Sample actions and compute log-prob + entropy for PufferLib 4.x policies."""
    from torch.distributions import Categorical, Normal  # type: ignore[import-not-found]

    if isinstance(logits, Normal):
        dist: Any = logits
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, logprob, entropy
    dist = Categorical(logits=logits)
    if action is None:
        action = dist.sample()
    return action, dist.log_prob(action.long()), dist.entropy()


def _compute_gae_vtrace_torch(
    values: Any,
    rewards: Any,
    terminals: Any,
    ratio: Any,
    gamma: float,
    gae_lambda: float,
    rho_clip: float,
    c_clip: float,
) -> Any:
    """Pure-PyTorch V-trace / GAE advantage matching PufferLib's ``compute_puff_advantage``."""
    import torch  # type: ignore[import-not-found]

    n_agents, horizon = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(n_agents, device=values.device, dtype=values.dtype)
    for t in range(horizon - 2, -1, -1):
        not_done = 1.0 - terminals[:, t + 1]
        rho_t = torch.clamp(ratio[:, t], max=rho_clip)
        c_t = torch.clamp(ratio[:, t], max=c_clip)
        delta = rho_t * (
            rewards[:, t + 1] + gamma * values[:, t + 1] * not_done - values[:, t]
        )
        last_gae = delta + gamma * gae_lambda * c_t * not_done * last_gae
        advantages[:, t] = last_gae
    return advantages


class _PufferPPOBackend:
    """Pure-Python PPO/Muon loop driving PufferLib 4.x ``Policy`` & ``Muon``."""

    def __init__(
        self,
        config: dict[str, Any],
        vecenv: _JaxDemPufferVecEnv,
        policy: Any,
    ) -> None:
        import torch  # type: ignore[import-not-found]

        self.config = config
        self.vecenv = vecenv
        self.policy = policy
        self.device = torch.device(config["device"])
        self.horizon = int(config["bptt_horizon"])
        self.total_agents = vecenv.num_agents
        self.batch_size = self.total_agents * self.horizon
        self.minibatch_segments = max(1, int(config["minibatch_size"]) // self.horizon)
        self.total_epochs = max(1, int(config["total_timesteps"]) // self.batch_size)
        self.discrete = vecenv.wrapper.discrete
        self.epoch = 0
        self.losses: dict[str, float] = {}

        if config["optimizer"] == "muon":
            from pufferlib.muon import Muon  # type: ignore[import-not-found]

            self.optimizer: Any = Muon(
                self.policy.parameters(),
                lr=float(config["learning_rate"]),
                momentum=float(config["adam_beta1"]),
                eps=float(config["adam_eps"]),
            )
        elif config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=float(config["learning_rate"]),
                betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
                eps=float(config["adam_eps"]),
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']!r}")

        act_shape = () if self.discrete else (vecenv.wrapper.action_size,)
        act_dtype = torch.int64 if self.discrete else torch.float32
        self.observations = torch.zeros(
            self.horizon,
            self.total_agents,
            vecenv.wrapper.obs_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            self.horizon,
            self.total_agents,
            *act_shape,
            dtype=act_dtype,
            device=self.device,
        )
        self.values = torch.zeros(self.horizon, self.total_agents, device=self.device)
        self.logprobs = torch.zeros(self.horizon, self.total_agents, device=self.device)
        self.rewards = torch.zeros(self.horizon, self.total_agents, device=self.device)
        self.terminals = torch.zeros(self.horizon, self.total_agents, device=self.device)
        self.ratio = torch.ones(self.total_agents, self.horizon, device=self.device)

        self.vecenv.reset()
        self.state = (
            self.policy.initial_state(self.total_agents, device=self.device)
            if hasattr(self.policy, "initial_state")
            else ()
        )

    def evaluate(self) -> None:
        import torch  # type: ignore[import-not-found]

        if self.state:
            self.state = tuple(torch.zeros_like(s) for s in self.state)

        for t in range(self.horizon):
            obs_np, rew_np, term_np, _trunc_np = self.vecenv.recv()
            o_dev = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            r_dev = torch.as_tensor(rew_np, dtype=torch.float32, device=self.device)
            d_dev = torch.as_tensor(term_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits, value, self.state = self.policy.forward_eval(o_dev, self.state)
                action, logprob, _ = _sample_logits(logits)

            self.observations[t] = o_dev
            self.actions[t] = action
            self.logprobs[t] = logprob
            self.rewards[t] = r_dev
            self.terminals[t] = d_dev
            self.values[t] = value.flatten()

            act_np = action.detach().cpu().numpy()
            if not self.discrete:
                act_np = np.clip(act_np, self.vecenv.act_low, self.vecenv.act_high)
            self.vecenv.send(act_np)

    def train(self) -> None:
        import torch  # type: ignore[import-not-found]

        cfg = self.config
        if cfg["anneal_lr"] and self.epoch > 0:
            lr_ratio = min(1.0, self.epoch / self.total_epochs)
            lr_min = float(cfg["learning_rate"]) * float(cfg["min_lr_ratio"])
            lr = lr_min + 0.5 * (float(cfg["learning_rate"]) - lr_min) * (
                1.0 + math.cos(math.pi * lr_ratio)
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        obs = self.observations.transpose(0, 1).contiguous()
        act = self.actions.transpose(0, 1).contiguous()
        val = self.values.T.contiguous()
        lp = self.logprobs.T.contiguous()
        rew = self.rewards.T.contiguous().clamp(-1.0, 1.0)
        ter = self.terminals.T.contiguous()

        self.ratio.fill_(1.0)
        b0 = float(cfg["prio_beta0"])
        a = float(cfg["prio_alpha"])
        anneal_beta = b0 + (1.0 - b0) * a * self.epoch / self.total_epochs
        clip_coef = float(cfg["clip_coef"])
        vf_clip = float(cfg["vf_clip_coef"])
        num_mb = max(
            1,
            int(
                round(
                    float(cfg["replay_ratio"])
                    * self.batch_size
                    / int(cfg["minibatch_size"])
                )
            ),
        )

        acc: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "old_approx_kl": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "importance": 0.0,
        }
        advantages = torch.zeros_like(val)

        for _ in range(num_mb):
            advantages = _compute_gae_vtrace_torch(
                val,
                rew,
                ter,
                self.ratio,
                float(cfg["gamma"]),
                float(cfg["gae_lambda"]),
                float(cfg["vtrace_rho_clip"]),
                float(cfg["vtrace_c_clip"]),
            )
            adv_mag = advantages.abs().sum(dim=1)
            prio_w = torch.nan_to_num(adv_mag.pow(a), nan=0.0, posinf=0.0, neginf=0.0)
            prio_probs = (prio_w + 1e-6) / (prio_w.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments, replacement=True)
            mb_prio = (self.total_agents * prio_probs[idx, None]).pow(-anneal_beta)

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_adv = advantages[idx]

            logits, newvalue = self.policy(mb_obs)
            flat_actions = (
                mb_actions.reshape(-1)
                if self.discrete
                else mb_actions.reshape(-1, mb_actions.shape[-1])
            )
            _, newlogprob, entropy = _sample_logits(logits, action=flat_actions)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)

            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            adv_norm = mb_prio * (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = torch.max(
                -adv_norm * ratio,
                -adv_norm * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss = 0.5 * torch.max(
                (newvalue - mb_returns) ** 2, (v_clipped - mb_returns) ** 2
            ).mean()

            ent_loss = entropy.mean()
            loss = (
                pg_loss
                + float(cfg["vf_coef"]) * v_loss
                - float(cfg["ent_coef"]) * ent_loss
            )
            val[idx] = newvalue.detach().float()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), float(cfg["max_grad_norm"])
            )
            self.optimizer.step()

            acc["policy_loss"] += float(pg_loss.item()) / num_mb
            acc["value_loss"] += float(v_loss.item()) / num_mb
            acc["entropy"] += float(ent_loss.item()) / num_mb
            acc["old_approx_kl"] += float(old_kl.item()) / num_mb
            acc["approx_kl"] += float(approx_kl.item()) / num_mb
            acc["clipfrac"] += float(clipfrac.item()) / num_mb
            acc["importance"] += float(ratio.mean().item()) / num_mb

        y_pred = val.flatten()
        y_true = advantages.flatten() + y_pred
        var_y = y_true.var()
        acc["explained_variance"] = (
            float("nan")
            if float(var_y.item()) == 0.0
            else float((1.0 - (y_true - y_pred).var() / var_y).item())
        )
        self.losses = acc
        self.epoch += 1

    def close(self) -> None:
        self.vecenv.close()


class PufferTrainer:
    """Train a JaxDEM :class:`Environment` using PufferLib 4.x models & PPO/Muon."""

    def __init__(
        self,
        *,
        env_wrapper: PufferEnvWrapper,
        vecenv: _JaxDemPufferVecEnv,
        trainer: _PufferPPOBackend,
        num_steps_epoch: int,
        num_epochs: int,
        vf_coef: float,
        ent_coef: float,
    ) -> None:
        self.env_wrapper = env_wrapper
        self._vecenv = vecenv
        self._trainer: _PufferPPOBackend | None = trainer
        self.num_steps_epoch = num_steps_epoch
        self.num_epochs = num_epochs
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self.epochs_completed = 0

    @classmethod
    def create(
        cls,
        env: Environment,
        *,
        num_envs: int = 64,
        num_steps_epoch: int = 64,
        skip_frames: int = 0,
        num_epochs: int = 100,
        minibatch_size: int | None = None,
        num_minibatches: int = 4,
        hidden_size: int = 128,
        num_layers: int = 1,
        learning_rate: float = 0.015,
        anneal_lr: bool = True,
        min_lr_ratio: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 2.0,
        vf_clip_coef: float = 0.2,
        ent_coef: float = 0.001,
        max_grad_norm: float = 1.5,
        momentum: float = 0.95,
        replay_ratio: float = 1.0,
        optimizer: str = "muon",
        use_rnn: bool | None = None,
        reset_every_horizon: bool = False,
        policy: Any | None = None,
        device: str | None = None,
        action_space: ActionSpace | None = None,
        clip_actions: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        discrete: bool = False,
        reward_scale: float = 1.0,
        seed: int = 1,
        key: ArrayLike | None = None,
        extra_overrides: Mapping[str, Any] | None = None,
    ) -> PufferTrainer:
        """Create a :class:`PufferTrainer` backed by PufferLib 4.x."""
        if num_steps_epoch <= 0:
            raise ValueError(f"num_steps_epoch must be >= 1, got {num_steps_epoch}")
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        wrapper = PufferEnvWrapper(
            env,
            num_envs=num_envs,
            skip_frames=skip_frames,
            action_space=action_space,
            clip_actions=clip_actions,
            clip_range=clip_range,
            discrete=discrete,
            reward_scale=reward_scale,
            seed=seed,
            key=key,
        )
        total_agents = wrapper.total_agents
        batch_size = total_agents * num_steps_epoch

        if minibatch_size is None:
            if batch_size % num_minibatches != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"num_minibatches ({num_minibatches})"
                )
            minibatch_size = batch_size // num_minibatches
        if minibatch_size <= 0 or minibatch_size > batch_size:
            raise ValueError(
                f"minibatch_size ({minibatch_size}) must lie in [1, {batch_size}]"
            )
        if minibatch_size % num_steps_epoch != 0:
            raise ValueError(
                f"minibatch_size ({minibatch_size}) must be divisible by "
                f"num_steps_epoch ({num_steps_epoch})"
            )

        try:
            import pufferlib.models as puf_models  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_PUFFERLIB_INSTALL_HINT) from exc

        if not (hasattr(puf_models, "Policy") and hasattr(puf_models, "DefaultEncoder")):
            raise ImportError(_PUFFERLIB_INSTALL_HINT)

        resolved_device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        vecenv = _JaxDemPufferVecEnv(
            wrapper, clip_actions=clip_actions, clip_range=clip_range
        )

        if policy is None:
            resolved_use_rnn = (
                (not reset_every_horizon) if use_rnn is None else bool(use_rnn)
            )
            encoder = puf_models.DefaultEncoder(
                wrapper.obs_size, hidden_size=int(hidden_size)
            )
            nvec = (
                (wrapper.action_size,)
                if wrapper.discrete
                else (1,) * wrapper.action_size
            )
            decoder = puf_models.DefaultDecoder(nvec, hidden_size=int(hidden_size))
            net_cls = puf_models.MinGRU if resolved_use_rnn else puf_models.MLP
            network = net_cls(
                hidden_size=int(hidden_size), num_layers=max(1, int(num_layers))
            )
            resolved_policy: Any = puf_models.Policy(encoder, decoder, network).to(
                resolved_device
            )
        else:
            resolved_policy = (
                policy.to(resolved_device) if hasattr(policy, "to") else policy
            )

        puf_cfg: dict[str, Any] = {
            "seed": int(seed),
            "device": resolved_device,
            "optimizer": optimizer,
            "total_timesteps": int(batch_size * int(num_epochs)),
            "learning_rate": float(learning_rate),
            "anneal_lr": bool(anneal_lr),
            "min_lr_ratio": float(min_lr_ratio),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "replay_ratio": float(replay_ratio),
            "clip_coef": float(clip_coef),
            "vf_coef": float(vf_coef),
            "vf_clip_coef": float(vf_clip_coef),
            "max_grad_norm": float(max_grad_norm),
            "ent_coef": float(ent_coef),
            "adam_beta1": float(momentum),
            "adam_beta2": 0.999,
            "adam_eps": 1e-12,
            "batch_size": int(batch_size),
            "minibatch_size": int(minibatch_size),
            "bptt_horizon": int(num_steps_epoch),
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
        }
        if extra_overrides:
            puf_cfg.update(extra_overrides)

        backend = _PufferPPOBackend(puf_cfg, vecenv, resolved_policy)
        return cls(
            env_wrapper=wrapper,
            vecenv=vecenv,
            trainer=backend,
            num_steps_epoch=int(num_steps_epoch),
            num_epochs=int(num_epochs),
            vf_coef=float(vf_coef),
            ent_coef=float(ent_coef),
        )

    @property
    def policy(self) -> Any:
        """Underlying PyTorch policy module."""
        assert self._trainer is not None, "PufferTrainer has been closed."
        return self._trainer.policy

    @property
    def num_parameters(self) -> int:
        """Number of trainable policy parameters."""
        assert self._trainer is not None, "PufferTrainer has been closed."
        return int(sum(p.numel() for p in self._trainer.policy.parameters()))

    def epoch(self) -> dict[str, float]:
        """Run one full rollout (``num_steps_epoch``) and PufferLib PPO/Muon update."""
        assert self._trainer is not None, "PufferTrainer has been closed."
        self._vecenv.reset_step_timer()
        t0 = time.perf_counter()
        self._trainer.evaluate()
        t1 = time.perf_counter()
        self._trainer.train()
        t2 = time.perf_counter()
        self.epochs_completed += 1

        env_s = self._vecenv.env_step_s
        model_s = max((t1 - t0) - env_s, 0.0)
        train_s = t2 - t1

        metrics: dict[str, float] = {
            f"loss/{k}": float(v)
            for k, v in self._trainer.losses.items()
            if isinstance(v, (int, float))
        }
        if (
            "loss/total" not in metrics
            and "loss/policy_loss" in metrics
            and "loss/value_loss" in metrics
        ):
            ent = metrics.get("loss/entropy", 0.0)
            metrics["loss/total"] = (
                metrics["loss/policy_loss"]
                + self._vf_coef * metrics["loss/value_loss"]
                - self._ent_coef * ent
            )

        metrics["perf/model_ms"] = model_s * 1e3
        metrics["perf/env_ms"] = env_s * 1e3
        metrics["perf/train_ms"] = train_s * 1e3
        metrics["perf/epoch_ms"] = (t2 - t0) * 1e3
        return metrics

    def train(
        self,
        *,
        num_epochs: int | None = None,
        verbose: bool = True,
        log: bool = False,
        directory: Path | str = "runs",
        report_every: int = 10,
        on_report: Callable[[int, Mapping[str, float]], None] | None = None,
        should_stop: Callable[[int, Mapping[str, float]], bool] | None = None,
    ) -> dict[str, float]:
        """Run the full training loop and return final summary metrics."""
        n_epochs = self.num_epochs if num_epochs is None else int(num_epochs)
        if n_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {n_epochs}")

        decisions_per_epoch = self.env_wrapper.total_agents * self.num_steps_epoch
        phys_steps_per_epoch = decisions_per_epoch * (1 + self.env_wrapper.skip_frames)

        writer: Any = None
        progress: Any = None
        last_metrics: dict[str, float] = {}
        try:
            if log:
                log_dir = (
                    Path(directory)
                    / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(log_dir, max_queue=1024, flush_secs=10)

            # Warmup epoch (compiles the JAX environment step).
            first = self.epoch()
            env_m0 = self.env_wrapper.consume_metrics()
            first.update({f"env/{k}": v for k, v in env_m0.items()})
            last_metrics = first

            if n_epochs == 1:
                if writer is not None:
                    for name, val in first.items():
                        if val == val:
                            writer.add_scalar(name, val, global_step=0)
                    writer.flush()
                if on_report is not None:
                    on_report(0, dict(first))
                return last_metrics

            t_start = time.perf_counter()
            model_ms_acc = 0.0
            env_ms_acc = 0.0
            train_ms_acc = 0.0

            it: Any = trange(1, n_epochs) if verbose else range(1, n_epochs)
            if verbose:
                progress = it

            for ep in it:
                m = self.epoch()
                model_ms_acc += m["perf/model_ms"]
                env_ms_acc += m["perf/env_ms"]
                train_ms_acc += m["perf/train_ms"]

                is_final = ep == n_epochs - 1
                if ep % report_every == 0 or is_final:
                    elapsed = max(time.perf_counter() - t_start, 1e-9)
                    env_m = self.env_wrapper.consume_metrics()
                    m.update({f"env/{k}": v for k, v in env_m.items()})
                    m["perf/elapsed_s"] = elapsed
                    m["perf/decision_sps"] = ep * decisions_per_epoch / elapsed
                    m["perf/phys_sps"] = ep * phys_steps_per_epoch / elapsed
                    m["perf/avg_epoch_ms"] = elapsed * 1e3 / ep
                    m["perf/avg_model_ms"] = model_ms_acc / ep
                    m["perf/avg_env_ms"] = env_ms_acc / ep
                    m["perf/avg_train_ms"] = train_ms_acc / ep
                    last_metrics = m

                    if progress is not None:
                        progress.set_postfix(
                            {
                                "phys/s": f"{m['perf/phys_sps']:.2e}",
                                "rew": f"{m['env/step_reward']:.4f}",
                                "ret": f"{m['env/active_return']:.2f}",
                            }
                        )
                    if writer is not None:
                        for name, val in m.items():
                            if val == val:
                                writer.add_scalar(name, val, global_step=ep)
                        if is_final:
                            writer.flush()
                    if on_report is not None:
                        on_report(ep, dict(m))
                    if should_stop is not None and not is_final and should_stop(ep, m):
                        break

            if self.env_wrapper.check_overflow():
                print("Warning: overflow detected in collider")
            return last_metrics
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                if progress is not None:
                    progress.close()

    def save_weights(self, path: Path | str) -> None:
        """Save the PyTorch policy ``state_dict`` to *path*."""
        assert self._trainer is not None, "PufferTrainer has been closed."
        import torch  # type: ignore[import-not-found]

        torch.save(self._trainer.policy.state_dict(), str(path))

    def close(self) -> None:
        """Close the underlying trainer and environment adapter."""
        if self._trainer is not None:
            self._trainer.close()
            self._trainer = None

    def __enter__(self) -> PufferTrainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PufferTrainer"]
