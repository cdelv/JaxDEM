import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import dataclasses as dc
import jaxdem as jd

def case(name,fn):
 try: print(name,fn(),flush=True)
 except Exception as e: print(name,type(e).__name__,str(e)[:600],flush=True)

def gradstep():
 s=jd.State.create(pos=jnp.array([[0.,0.],[3.,0.]]))
 y=jd.System.create(state=s)
 def f(pos):
  t,_=jd.System.step(dc.replace(s,pos_c=pos),y,n=2)
  return t.pos_c.sum()
 return jax.grad(f)(s.pos_c)
case('step reverse mode',gradstep)

def hist_invalidation():
 s=jd.State.create(pos=jnp.array([[1.,.1],[6.,9.9]]),rad=jnp.full(2,.2))
 y=jd.System.create(state=s,collider_type='neighborlist',collider_kw={'cutoff':.4,'secondary_collider_type':'naive'},domain_type='leesedwards',domain_kw={'box_size':jnp.array([10.,10.]),'gamma':0.})
 s,y=y.collider.compute_force(s,y)
 y=dc.replace(y,domain=dc.replace(y.domain,gamma=jnp.array(.5)))
 stale,y2=y.collider.compute_force(s,y)
 naive=dc.replace(y,collider=jd.Collider.create('naive'))
 ref,_=naive.collider.compute_force(s,naive)
 return {'cached':np.asarray(stale.force).tolist(),'reference':np.asarray(ref.force).tolist(),'builds':int(y2.collider.n_build_times)}
case('shear neighbor invalidation',hist_invalidation)

def cache():
 s=jd.State.create(pos=jnp.zeros((1,2)),pos_p=jnp.array([[1.,0.]]))
 s.q.w=jnp.zeros_like(s.q.w); s.q.xyz=jnp.array([[0.,0.,1.]])
 return {'pos':np.asarray(s.pos).tolist(),'expected':np.asarray(s.q.rotate(s.q,s.pos_p)).tolist()}
case('nested quaternion cache',cache)

def hessian_range():
 s=jd.State.create(pos=jnp.array([[0.,0.],[3.2,0.]]),rad=jnp.ones(2))
 y=jd.System.create(state=s,force_model_type='lennardjones',mat_table=jd.MaterialTable.from_materials([jd.Material.create('lj',density=1.,epsilon=1.)]))
 _,_,h=jd.utils.non_bonded_hessian(s,y)
 _,_,hr=jd.utils.non_bonded_hessian(s,y,cutoff=5.)
 return {'default_norm':float(jnp.linalg.norm(h)),'full_norm':float(jnp.linalg.norm(hr))}
case('hessian cutoff',hessian_range)

def clump_inertia():
 s=jd.State.add_clump(jd.State.create(dim=2),pos=jnp.array([[0.,0.],[3.,0.]]),rad=jnp.ones(2))
 return {'mass':np.asarray(s.mass).tolist(),'inertia':np.asarray(s.inertia).tolist(),'expected_inertia_unit_body_mass':2.75}
case('default clump inertia',clump_inertia)

def spring_clump_friction():
 s=jd.State.create(pos=jnp.array([[0.,0.],[0.,0.],[1.4,0.]]),pos_p=jnp.array([[0.,-.2],[0.,.2],[0.,0.]]),clump_id=jnp.array([0,0,1]),ang_vel=jnp.array([[1.],[1.],[0.]]))
 mt=jd.MaterialTable.from_materials([jd.Material.create('elasticfrict',density=1.,young=1.,poisson=.3,mu=1.,e=.5)])
 y=jd.System.create(state=s,force_model_type='cundallstrack',mat_table=mt)
 f1,t1=y.force_model.force(1,2,s.pos,s,y)
 # Equivalent independent spheres with member center velocity v_COM + omega x offset.
 s2=dc.replace(s,vel=s.vel.at[1].set(jnp.array([-.2,0.])),pos_c=s.pos,pos_p=jnp.zeros_like(s.pos_p),clump_id=jnp.arange(3))
 f2,t2=y.force_model.force(1,2,s2.pos,s2,y)
 return {'clump_force':np.asarray(f1).tolist(),'correct_member_velocity_force':np.asarray(f2).tolist()}
case('Cundall clump contact velocity',spring_clump_friction)
