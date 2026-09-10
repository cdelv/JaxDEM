import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import jaxdem as jd
s=jd.State.create(pos=jnp.array([[4.,4.],[5.,4.]]),rad=jnp.ones(2))
for collider in ('naive','celllist','multicelllist'):
 y=jd.System.create(state=s,collider_type=collider)
 t,y,n,pe=jd.System.minimize(s,y,max_steps=1)
 print(collider,'steps',n,'PE',pe,'force',t.force,flush=True)
