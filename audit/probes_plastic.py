import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import jaxdem as jd
s=jd.State.create(pos=jnp.array([[0.,0.],[2.,0.]]),fixed=jnp.ones(2,dtype=bool))
m=jd.BondedForceModel.create('PlasticDeformableParticleModel',edges=jnp.array([[0,1]]),initial_edge_lengths=jnp.array([1.]),el=1.,tau_s=1.)
y=jd.System.create(state=s,dt=.1,bonded_force_model=m,collider_type='')
print('explicit update',m.update_reference_state(s.pos,s,y).initial_edge_lengths,flush=True)
for i in range(3):
 s,y=jd.System.step(s,y)
 print('step',i+1,'reference',y.bonded_force_model.initial_edge_lengths,'force',s.force,flush=True)

# Follow-up: use the same container builder and System attachment as the
# deformable construction examples, with a small deterministic polygon.
from jaxdem.utils.particle_creation import create_dp_container

vertices = jnp.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
for plasticity in ('edge', 'perimeter', 'bending'):
    state = jd.State.create(
        pos=vertices,
        bond_id=jnp.broadcast_to(jnp.arange(4), (4, 4)),
        fixed=jnp.ones(4, dtype=bool),
    )
    model = create_dp_container(
        state, el=1., eb=1., plasticity_type=plasticity, tau_s=1.,
    )
    state.pos_c = state.pos_c.at[2].set(jnp.array([2., 1.5]))
    system = jd.System.create(
        state=state, dt=.1, bonded_force_model=model,
        collider_type='', rotation_integrator_type='',
    )
    field = 'initial_bendings' if plasticity == 'bending' else 'initial_edge_lengths'
    reference = getattr(model, field)
    expected = getattr(model.update_reference_state(state.pos, state, system), field)
    print('example builder', plasticity, 'direct reference change',
          float(jnp.max(jnp.abs(expected - reference))), flush=True)
    for step in range(3):
        state, system = jd.System.step(state, system)
        actual = getattr(system.bonded_force_model, field)
        print('  step', step + 1, 'persisted reference change',
              float(jnp.max(jnp.abs(actual - reference))), flush=True)
