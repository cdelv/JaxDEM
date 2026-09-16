"""Opt-in chunked float32 parity with actual PufferLib CUDA kernels.

Run with PUFFERLIB_CUDA_PARITY=1 JAX_PLATFORMS=cuda and an accessible NVIDIA
GPU/toolkit. Intentional differences are aligned explicitly, not hidden by loose
assertions: rewards already clipped, T-1 transitions versus Puffer's final
bootstrap row, no truncations/padding, fixed detached targets for loss gradients.
"""
import os

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.models import Model
from jaxdem.rl.trainers import Trainer, TrajectoryData
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from tests.puffer_cuda_reference import build_reference

pytestmark = pytest.mark.skipif(os.environ.get('PUFFERLIB_CUDA_PARITY') != '1',
    reason='explicit GPU parity run required')


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return build_reference(tmp_path_factory.mktemp('puffer_cuda'))


@pytest.mark.parametrize('vtrace', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_advantages_and_returns_match_native_cuda(native, vtrace, terminal):
    rng = np.random.default_rng(71)
    S,T = 3,8
    values = rng.normal(size=(S,T)).astype('float32')
    rewards = np.clip(rng.normal(size=(S,T)),-1,1).astype('float32')
    dones = np.zeros((S,T),dtype='float32')
    if terminal:
        dones[0,3]=dones[1,-1]=dones[2,1]=1
    ratios = rng.uniform(.2,1.8,(S,T)).astype('float32')
    data = np.concatenate([x.ravel() for x in (values,rewards,dones,ratios)])
    output = np.empty(2*S*T,dtype='float32')
    assert native.advantage(data,output,S,T,vtrace,.99,.95,1.,1.) == 0
    expected_a,expected_r = output.reshape(2,S,T)
    # Puffer's row t+1 reward/done belongs to action t. Its final row is
    # only a bootstrap; JaxDEM receives the matching T-1 actual transitions.
    returns,advantage = Trainer.compute_advantages(
        value=jnp.array(values[:,:-1].T),reward=jnp.array(rewards[:,1:].T),
        ratio=jnp.array(ratios[:,:-1].T if vtrace else np.ones((T-1,S),dtype=np.float32)),
        done=jnp.array(dones[:,1:].T,dtype=bool),
        last_value=jnp.array(values[:,-1]),
        advantage_gamma=jnp.float32(.99),advantage_lambda=jnp.float32(.95),
        advantage_rho_clip=jnp.float32(1),advantage_c_clip=jnp.float32(1))
    np.testing.assert_allclose(advantage,expected_a[:,:-1].T,rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(returns,expected_r[:,:-1].T,rtol=3e-6,atol=3e-6)
    np.testing.assert_array_equal(expected_a[:,-1],0.)
    print(f'advantages vtrace={vtrace} terminal={terminal}: max_abs={np.max(np.abs(np.asarray(advantage)-expected_a[:,:-1].T)):.3g}')


class FixturePolicy(Model):
    def __init__(self, prediction, continuous):
        self.prediction=nnx.Param(jnp.array(prediction))
        self.logstd=nnx.Param(jnp.array([-.3],dtype=jnp.float32))
        self.continuous=continuous
    def __call__(self, obs, sequence=False, **kwargs):
        p=self.prediction[...]
        dist=(distrax.MultivariateNormalDiag(p[...,:-1],jnp.exp(self.logstd[...]))
              if self.continuous else distrax.Categorical(logits=p[...,:-1]))
        return dist,p[...,-1:]


@pytest.mark.parametrize('continuous', [False,True])
def test_policy_value_entropy_and_gradients_match_native_cuda(native, continuous):
    rng=np.random.default_rng(93)
    T=8
    A=1 if continuous else 3
    prediction=rng.normal(size=(T,1,A+1)).astype('float32')
    model=FixturePolicy(prediction,continuous)
    dist,_=model(None)
    action=(rng.normal(size=(T,1,1)).astype('float32') if continuous
            else rng.integers(0,3,size=(T,1),dtype='int32'))
    # Both advantage signs, clipped/unclipped ratios, clipped/unclipped values.
    ratio=np.array([.5,.95,1.5,.7,1.6,1.05,.6,1.4],dtype='float32').reshape(T,1)
    old_lp=np.asarray(dist.log_prob(action))-np.log(ratio)
    advantage=np.array([1.,-2.,3.,-1.,-2.,1.,2.,-3.],dtype='float32').reshape(T,1)
    values=prediction[...,-1]
    rewards=values+advantage
    old_values=values+np.array([0.,.5,-.5,0.,.5,0.,-.5,0.],dtype='float32').reshape(T,1)
    done=jnp.ones((T,1),dtype=bool)
    td=TrajectoryData(obs=jnp.zeros((T,1,1)),action=jnp.array(action),
        value=jnp.array(old_values),log_prob=jnp.array(old_lp),ratio=jnp.ones((T,1)),
        reward=jnp.array(rewards),done=done,terminated=done,truncated=~done,
        agent_mask=done,bootstrap_value=jnp.zeros((T,1)))
    (loss,aux),grads=nnx.value_and_grad(PPOTrainer.loss_fn,has_aux=True)(model,td,
        ppo_clip_eps=jnp.float32(.2),ppo_value_coeff=jnp.float32(2.),ppo_entropy_coeff=jnp.float32(.03),
        advantage_gamma=jnp.float32(.99),advantage_lambda=jnp.float32(.95),
        advantage_rho_clip=jnp.float32(1),advantage_c_clip=jnp.float32(1),
        last_value=jnp.zeros(1),vtrace=jnp.array(False))
    data=np.concatenate([x.ravel() for x in (prediction,action.astype('float32'),old_lp,
        advantage,old_values,rewards,np.array([-.3],dtype='float32'))]).astype('float32')
    output=np.empty(8+2*T*A+2*T,dtype='float32')
    assert native.loss(data,output,T,continuous,.2,2.,.03)==0
    np.testing.assert_allclose([aux['actor_loss'],aux['value_loss'],aux['entropy'],loss],output[:4],rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(aux['ratio'].ravel(),output[-T:],rtol=3e-6,atol=3e-6)
    grad_policy=output[8:8+T*A].reshape(T,1,A)
    grad_std=output[8+T*A:8+2*T*A].reshape(T,A)
    grad_value=output[8+2*T*A:8+2*T*A+T].reshape(T,1,1)
    expected_grad=np.concatenate((grad_policy,grad_value),axis=-1)
    np.testing.assert_allclose(grads.prediction[...],expected_grad,rtol=4e-5,atol=4e-6)
    if continuous:
        np.testing.assert_allclose(grads.logstd[...],grad_std.sum(axis=0),rtol=4e-5,atol=4e-6)
    print(f'loss continuous={continuous}: max_gradient_abs={np.max(np.abs(np.asarray(grads.prediction[...])-expected_grad)):.3g}')
