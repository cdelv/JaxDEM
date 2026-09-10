import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataclasses as dc
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd

def case(name, fn):
    try: print(name, fn(), flush=True)
    except Exception as e: print(name, type(e).__name__, str(e)[:700], flush=True)

def batch(b):
    s=jd.State.create(pos=jnp.array([[0.,0.],[1.5,0.]]))
    y=jd.System.create(state=s)
    sb=jd.State.stack([s]*b); yb=jd.System.stack([y]*b)
    out,yout=jd.System.step(sb,yb)
    return out.shape
case('batch1', lambda: batch(1))
case('batch2', lambda: batch(2))

def le_clump():
    s=jd.State.create(pos=jnp.array([[5.,9.9],[5.,9.9]]),pos_p=jnp.array([[0.,-.2],[0.,.2]]),clump_id=jnp.array([0,0]))
    y=jd.System.create(state=s,domain_type='leesedwards',domain_kw={'box_size':jnp.array([10.,10.]),'gamma':.5})
    t,_=y.domain.shift(s,y)
    return np.asarray(t.pos_c).tolist()
case('LE clump',le_clump)

def le_pairs():
    s=jd.State.create(pos=jnp.array([[1.,.1],[6.,9.9]]),rad=jnp.array([.2,.2]))
    out={}
    for c in ['naive','celllist','multicelllist','neighborlist']:
        kw={'cutoff':.4} if c=='neighborlist' else {}
        y=jd.System.create(state=s,collider_type=c,collider_kw=kw,domain_type='leesedwards',domain_kw={'box_size':jnp.array([10.,10.]),'gamma':.5})
        t,y=y.collider.compute_force(s,y)
        out[c]=np.asarray(t.force).tolist()
    return out
case('LE colliders',le_pairs)

def lj_pairs():
    s=jd.State.create(pos=jnp.array([[.9,5.],[4.1,5.]]),rad=jnp.ones(2))
    mat=jd.MaterialTable.from_materials([jd.Material.create('lj',density=1.,epsilon=1.)])
    out={}
    for c in ['naive','celllist','multicelllist']:
        y=jd.System.create(state=s,collider_type=c,force_model_type='lennardjones',mat_table=mat,domain_type='periodic',domain_kw={'box_size':jnp.array([20.,20.])})
        t,_=y.collider.compute_force(s,y); out[c]=np.asarray(t.force).tolist()
    return out
case('LJ colliders',lj_pairs)

def langevin():
    s=jd.State.create(pos=jnp.array([[0.,0.],[0.,0.]]),pos_p=jnp.array([[-.2,0.],[.2,0.]]),clump_id=jnp.array([0,0]))
    y=jd.System.create(state=s,linear_integrator_type='langevin',linear_integrator_kw={'gamma':1.,'k_B':1.,'temperature':1.},rotation_integrator_type=None)
    t,_=jd.System.step(s,y)
    return {'COM':np.asarray(t.pos_c).tolist(),'vel':np.asarray(t.vel).tolist()}
case('Langevin clump',langevin)

def startup():
    s=jd.State.create(pos=jnp.array([[0.,0.]]),rad=jnp.ones(1))
    y=jd.System.create(state=s,dt=.1,force_manager_kw={'gravity':jnp.array([0.,-10.])})
    t,_=jd.System.step(s,y)
    return {'pos':np.asarray(t.pos).tolist(),'vel':np.asarray(t.vel).tolist(),'expected':[[0.,-.05],[0.,-1.]]}
case('Verlet startup',startup)
case('zero-radius neighbor',lambda:jd.System.create(state=jd.State.create(pos=jnp.array([[0.,0.],[1.,0.]]),rad=jnp.zeros(2)),collider_type='neighborlist',collider_kw={'cutoff':2.,'max_neighbors':2}))

def bench():
    from benchmarks.base import create_deformable_state,create_mixed_state
    s=create_deformable_state(N=12); t=create_mixed_state(N=12)
    return {'bonds':np.asarray(s.bond_id).tolist(),'mixed_N':t.N}
case('benchmarks',bench)
