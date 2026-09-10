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
    except Exception as e: print(name, type(e).__name__, str(e)[:500], flush=True)

@jd.ForceModel.register('audit_history')
@jax.tree_util.register_dataclass
@dc.dataclass
class Hist(jd.ForceModel):
    @property
    def requires_history(self): return True
    def init_history(self,shape): return jnp.zeros(shape)
    @staticmethod
    def force(i,j,pos,state,system):
        return jnp.zeros(jnp.shape(j)+(state.dim,)),jnp.zeros(jnp.shape(j)+(1,))
    @staticmethod
    def force_and_history(i,j,pos,state,system,h):
        f,t=Hist.force(i,j,pos,state,system); return f,t,h+1
    @staticmethod
    def energy(i,j,pos,state,system): return jnp.zeros(jnp.shape(j))

def hist_combo(cls):
    return cls.init_history((2,2))
case('combiner history',lambda:hist_combo(jd.LawCombiner(laws=(Hist(),))))
case('router history',lambda:hist_combo(jd.ForceRouter(table=((Hist(),),))))

def history_energy():
    s=jd.State.create(pos=jnp.array([[0.,0.],[.5,0.],[5.,0.]]),rad=jnp.full(3,.2))
    y=jd.System.create(state=s,force_model=Hist(),collider_type='neighborlist',collider_kw={'cutoff':1.,'max_neighbors':2,'secondary_collider_type':'naive'})
    s,y=y.collider.compute_force(s,y)
    old=(np.asarray(y.collider.neighbor_list).tolist(),np.asarray(y.collider.history).tolist())
    s=dc.replace(s,pos_c=jnp.array([[0.,0.],[5.,0.],[.5,0.]]))
    from jaxdem.colliders.neighbor_list import _check_and_rebuild
    correct_history = _check_and_rebuild(s,y,y.collider)[4]
    _,ye,_=y.collider.compute_potential_energy(s,y)
    _,yf=y.collider.compute_force(s,y)
    return {'old':old,'energy_nl':np.asarray(ye.collider.neighbor_list).tolist(),'energy_history':np.asarray(ye.collider.history).tolist(),'force_history':np.asarray(yf.collider.history).tolist(),'expected_history_after_rebuild':np.asarray(correct_history).tolist()}
case('history energy rebuild',history_energy)

def refresh_hist():
    s=jd.State.create(pos=jnp.array([[0.,0.],[.5,0.]]))
    y=jd.System.create(state=s,force_model=Hist(),collider_type='neighborlist',collider_kw={'cutoff':2.,'max_neighbors':2,'secondary_collider_type':'naive'})
    s,y=jd.System.step(s,y)
    y=dc.replace(y,collider=jd.colliders.refresh_collider(s,y.collider))
    return jd.System.step(s,y)[1].collider.history
case('refresh history step',refresh_hist)

def callbacks():
    import tempfile
    from pathlib import Path
    from jaxdem.system import _save_state_system
    # importable existing function with observable effect
    from jaxdem.domains.periodic import PeriodicDomain
    s=jd.State.create(pos=jnp.array([[0.,0.]]))
    y=jd.System.create(state=s,user_post_step_actions=PeriodicDomain.shift,domain_type='leesedwards',domain_kw={'gamma':.2,'alpha':1,'beta':0,'box_size':jnp.ones(2)*10})
    with tempfile.TemporaryDirectory() as d:
        with jd.CheckpointWriter(directory=Path(d)) as w: w.save(s,y)
        with jd.CheckpointLoader(directory=Path(d)) as r: st,yr=r.load()
    return {'post_preserved':yr.user_post_step_actions is PeriodicDomain.shift,'post_default':yr.user_post_step_actions is _save_state_system,'axes':(yr.domain.alpha,yr.domain.beta),'axis_vectors':(np.asarray(yr.domain.alpha_axis).tolist(),np.asarray(yr.domain.beta_axis).tolist())}
case('checkpoint callback and axes',callbacks)

def connected():
    s=jd.State.add_facet(jd.State.create(dim=2),jnp.array([[0.,0.],[1.,0.]]),rigid=False)
    old=np.asarray(s.facet_vertices).tolist()
    t=jd.State.add_connected_facet(s,[1,jnp.array([2.,1.])],rigid=False)
    return {'before':old,'after':np.asarray(t.facet_vertices).tolist(),'facet_ids':np.asarray(t.facet_id).tolist()}
case('connected facet',connected)

def volume():
    s=jd.State.add_clump(jd.State.create(dim=2),pos=jnp.array([[0.,0.],[3.,0.]]),rad=jnp.ones(2))
    return {'stored':np.asarray(s.volume).tolist(),'computed':float(jd.utils.compute_particle_volume(s)),'expected':float(2*jnp.pi)}
case('clump volume',volume)
case('zero-capacity overflow',lambda:jd.Collider.create('naive').create_cross_neighbor_list(jnp.zeros((1,2)),jnp.zeros((1,2)),jd.System.create((1,2)),1.,0))

def same_as():
    from jaxdem.rl.action_spaces.box_space import BoxSpace
    return BoxSpace(-1.,1.).same_as(BoxSpace(-20.,20.))
case('Box same_as different params',same_as)
