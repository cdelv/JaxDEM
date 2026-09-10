import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import jaxdem as jd

def torque_fn(pos,state,system): return jnp.zeros_like(pos),jnp.ones_like(state.torque)
s=jd.State.create(pos=jnp.array([[0.,0.]]),vel=jnp.array([[1.,0.]]),inertia=jnp.ones((1,1)),torque=jnp.ones((1,1)))
y=jd.System.create(state=s,dt=.1,collider_type='',linear_integrator_type='verlet_rescaling',linear_integrator_kw={'temperature':1.,'can_rotate':True},force_manager_kw={'force_functions':(torque_fn,)})
s,y=jd.System.step(s,y)
print('temperature after full step',jd.utils.compute_temperature(s,can_rotate=True,subtract_drift=False),'target',1.,flush=True)
