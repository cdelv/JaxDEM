import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import optax
import jaxdem as jd

def make_optimizer(rate):
    def optimizer():
        return optax.sgd(rate)
    return optimizer

s = jd.State.create(pos=jnp.zeros((1, 2)))
a = jd.System.create(state=s, minimizer=make_optimizer(.1))
b = jd.System.create(state=s, minimizer=make_optimizer(.2))

@jax.jit
def optimizer_update(system):
    params = {'x': jnp.array(1.)}
    grads = {'x': jnp.array(1.)}
    updates, _ = system.minimizer.update(grads, system.minimizer.init(params), params)
    return updates['x']

print('different optimizers compare equal:', a.minimizer == b.minimizer)
print('rate .1 update:', optimizer_update(a))
print('rate .2 update:', optimizer_update(b), 'expected -0.2')
