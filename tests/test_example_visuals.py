# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Documentation scenes preserve real VTK frame identity and physical fields.

These tests need the ordinary VTK Python package, without ParaView or rendering.
The complete two-disk snapshots are tiny data fixtures; no simulation is run.
"""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np
import pytest

vtk = pytest.importorskip("vtk")
from vtk.util.numpy_support import vtk_to_numpy

import jaxdem as jd
from tools.example_scene_exports import export_example


@pytest.fixture
def tiny_snapshot():
    state = jd.State.create(
        pos=jnp.array([[0.2, 0.3], [0.8, 0.7]]),
        rad=jnp.array([0.15, 0.25]),
        vel=jnp.array([[0.1, 0.2], [-0.15, 0.3]]),
        force=jnp.array([[1.0, -0.5], [-0.2, -2.0]]),
    )
    system = jd.System.create(
        state=state,
        dt=0.05,
        collider_type="naive",
        rotation_integrator_type=None,
        domain_type="free",
    )
    return state, system


@pytest.fixture
def recorded_history(tiny_snapshot):
    state, system = tiny_snapshot
    states, systems = [], []
    for index, (time, step) in enumerate(zip((0.125, 0.375, 0.875), (3, 8, 18))):
        states.append(
            replace(
                state,
                pos_c=state.pos_c + jnp.array([0.04 * index, -0.02 * index]),
                rad=state.rad * (1.0 + 0.1 * index),
                vel=state.vel + index,
                force=state.force * (index + 1),
            )
        )
        systems.append(
            replace(system, time=jnp.asarray(time), step_count=jnp.asarray(step))
        )
    return states, systems


def particle_collections(manifest):
    return [
        Path(item["path"]) for item in manifest["series"] if item["kind"] == "particles"
    ]


def read_collection(path):
    frames = []
    for entry in ET.parse(path).findall("./Collection/DataSet"):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path.parent / entry.attrib["file"]))
        reader.Update()
        assert reader.GetErrorCode() == 0
        frames.append((float(entry.attrib["timestep"]), reader.GetOutput()))
    return frames


def assert_physical_fields_match(poly, state):
    assert poly.GetNumberOfPoints() == state.N
    np.testing.assert_allclose(
        vtk_to_numpy(poly.GetPoints().GetData()),
        np.pad(np.asarray(state.pos), ((0, 0), (0, 1))),
    )
    for field in ("rad", "vel", "force"):
        expected = np.asarray(getattr(state, field))
        if expected.ndim == 2:
            expected = np.pad(expected, ((0, 0), (0, 1)))
        np.testing.assert_allclose(
            vtk_to_numpy(poly.GetPointData().GetArray(field)), expected
        )


@pytest.mark.parametrize("stacked", [False, True])
def test_saved_history_keeps_native_times_radii_and_each_frames_fields(
    tmp_path, recorded_history, stacked
):
    states, systems = recorded_history
    namespace = {
        "scene_states": jd.State.stack(states) if stacked else states,
        "scene_systems": jd.System.stack(systems) if stacked else systems,
    }
    manifest = export_example(
        "recorded_history", namespace, tmp_path / "work", tmp_path / "media"
    )
    assert manifest is not None and manifest["animate"]
    collections = particle_collections(manifest)
    assert len(collections) == 1
    frames = read_collection(collections[0])
    assert len(frames) == len(states)
    for (time, poly), state, system in zip(frames, states, systems, strict=True):
        assert time == float(system.time)
        assert_physical_fields_match(poly, state)
    assert manifest["export_metadata"]["groups"][0]["source_step_counts"] == [3, 8, 18]
    particle_metrics = next(
        record
        for record in manifest["export_metadata"]["geometry"]
        if record["kind"] == "particles"
    )
    for record, state in zip(particle_metrics["frames"], states, strict=True):
        summary = record["physical_summary"]
        assert summary["radius_min"] == float(state.rad.min())
        assert summary["radius_max"] == float(state.rad.max())
        np.testing.assert_allclose(
            summary["speed_max"], np.linalg.norm(state.vel, axis=-1).max()
        )


def test_independent_batch_snapshots_do_not_become_a_time_history(
    tmp_path, recorded_history
):
    states, systems = recorded_history
    selected = (0, 2)
    namespace = {
        "state": jd.State.stack([states[i] for i in selected]),
        "system": jd.System.stack([systems[i] for i in selected]),
    }
    manifest = export_example(
        "independent_batch", namespace, tmp_path / "work", tmp_path / "media"
    )
    assert manifest is not None and not manifest["animate"]
    collections = particle_collections(manifest)
    assert len(collections) == len(selected)
    for collection, index in zip(collections, selected, strict=True):
        frames = read_collection(collection)
        assert len(frames) == 1
        time, poly = frames[0]
        assert time == float(systems[index].time)
        assert_physical_fields_match(poly, states[index])
    assert all(
        "offset" in item for item in manifest["series"] if item["kind"] == "particles"
    )


def test_reexport_uses_current_fields_and_removes_obsolete_frames(
    tmp_path, recorded_history
):
    states, systems = recorded_history
    work, media = tmp_path / "work", tmp_path / "media"
    first = export_example(
        "current_run", {"scene_states": states, "scene_systems": systems}, work, media
    )
    assert len(read_collection(particle_collections(first)[0])) == 3
    changed = replace(states[-1], pos_c=states[-1].pos_c + 0.025, vel=-states[-1].vel)
    current = export_example(
        "current_run", {"state": changed, "system": systems[-1]}, work, media
    )
    assert not current["animate"]
    frames = read_collection(particle_collections(current)[0])
    assert len(frames) == 1
    assert frames[0][0] == float(systems[-1].time)
    assert_physical_fields_match(frames[0][1], changed)
    assert len(list(work.rglob("*.vtp"))) == 1


def test_explicit_capture_keeps_initial_frames_omitted_by_existing_writer(
    tmp_path, recorded_history
):
    states, systems = recorded_history
    targets = [np.asarray([[0.3 + i, 0.4 - i]]) for i in range(len(states))]
    with jd.VTKWriter(
        directory=tmp_path / "example_output", writers=["spheres"], max_workers=1
    ) as writer:
        writer.save(states[-1], systems[-1])
        writer.block_until_ready()
        manifest = export_example(
            "intro_to_rl",
            {
                "writer": writer,
                "scene_states": states,
                "scene_systems": systems,
                "scene_targets": targets,
            },
            tmp_path / "work",
            tmp_path / "media",
        )
    frames = read_collection(particle_collections(manifest)[0])
    assert len(frames) == len(states)
    for (time, poly), state, system in zip(frames, states, systems, strict=True):
        assert time == float(system.time)
        assert_physical_fields_match(poly, state)
    goal_series = next(
        item
        for item in manifest["series"]
        if item.get("label", "").endswith("target marker")
    )
    goal_frames = read_collection(Path(goal_series["path"]))
    for (time, poly), target, system in zip(goal_frames, targets, systems, strict=True):
        assert time == float(system.time)
        points = vtk_to_numpy(poly.GetPoints().GetData())
        np.testing.assert_allclose(points[:, :2].mean(axis=0), target[0])


@pytest.mark.parametrize("stacked", [False, True])
def test_history_with_mismatched_system_snapshots_is_rejected(
    tmp_path, recorded_history, stacked
):
    states, systems = recorded_history
    namespace = {
        "scene_states": jd.State.stack(states) if stacked else states,
        "scene_systems": jd.System.stack(systems[:2]) if stacked else systems[:2],
    }
    with pytest.raises(ValueError, match="matching|equally sized"):
        export_example(
            "mismatched_history", namespace, tmp_path / "work", tmp_path / "media"
        )


@pytest.mark.parametrize("times", [(0.125, 0.125, 0.875), (0.375, 0.125, 0.875)])
def test_nonincreasing_history_is_rejected(tmp_path, recorded_history, times):
    states, systems = recorded_history
    namespace = {
        "scene_states": states,
        "scene_systems": [
            replace(system, time=jnp.asarray(time))
            for system, time in zip(systems, times, strict=True)
        ],
    }
    with pytest.raises(ValueError, match="times must increase"):
        export_example(
            "invalid_history", namespace, tmp_path / "work", tmp_path / "media"
        )


@pytest.mark.parametrize("field", ["pos_c", "rad", "vel", "force"])
def test_nonfinite_physical_vtk_output_is_rejected(tmp_path, tiny_snapshot, field):
    state, system = tiny_snapshot
    # Bypass construction validation to model a numerical failure in a live
    # simulation. The exporter must inspect the actual written VTK payload.
    values = getattr(state, field).at[0].set(jnp.nan)
    invalid_state = replace(state, **{field: values})
    with pytest.raises(ValueError, match="Nonfinite physical VTK"):
        export_example(
            "nonfinite_snapshot",
            {"state": invalid_state, "system": system},
            tmp_path / "work",
            tmp_path / "media",
        )


@pytest.mark.parametrize("radius", [0.0, -0.1])
def test_nonpositive_physical_sphere_radius_is_rejected(
    tmp_path, tiny_snapshot, radius
):
    state, system = tiny_snapshot
    invalid_state = replace(state, rad=state.rad.at[0].set(radius))
    with pytest.raises(ValueError, match="Invalid physical VTK radii"):
        export_example(
            "invalid_radius",
            {"state": invalid_state, "system": system},
            tmp_path / "work",
            tmp_path / "media",
        )


@pytest.fixture
def visual_build_contract(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tools import example_visuals

    # Isolate the build-wide provenance globals and avoid reading or modifying
    # repository sources. No test may export a scene or launch a renderer.
    monkeypatch.setattr(example_visuals, "_build_library_hash", None)
    monkeypatch.setattr(example_visuals, "_build_example_hashes", {})
    monkeypatch.setattr(example_visuals, "_library_hash", lambda root: "library-before")
    monkeypatch.setattr(
        example_visuals.shutil, "which", lambda executable: sys.executable
    )
    export = Mock(side_effect=AssertionError("Unexpected scene export"))
    render = Mock(side_effect=AssertionError("Unexpected renderer execution"))
    monkeypatch.setattr(example_visuals, "export_example", export)
    monkeypatch.setattr(example_visuals.subprocess, "run", render)
    examples = tmp_path / "examples"
    examples.mkdir()
    source = examples / "sphere_construction.py"
    source.write_text("value = 1\n")
    output = tmp_path / "docs" / "_static" / "visuals"
    return SimpleNamespace(
        module=example_visuals,
        examples=examples,
        source=source,
        output=output,
        export=export,
        render=render,
    )


def test_visual_build_removes_all_stale_generated_media(visual_build_contract):
    build = visual_build_contract
    stale = build.output / "previous_example" / "scene.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"obsolete image")
    static_asset = build.output.parent / "logo.svg"
    static_asset.write_text("retained static asset")

    build.module.start_visual_build(build.examples, build.output)

    assert not build.output.exists()
    assert static_asset.read_text() == "retained static asset"
    build.export.assert_not_called()
    build.render.assert_not_called()


@pytest.mark.parametrize(
    "failure", [{"execute_script": False}, {"formatted_exception": "AssertionError"}]
)
def test_failed_example_does_not_export_or_render(visual_build_contract, failure):
    from types import SimpleNamespace

    build = visual_build_contract
    build.module.start_visual_build(build.examples, build.output)
    block = SimpleNamespace(lineno=1, content=build.source.read_text())
    variables = {
        "src_file": str(build.source),
        "example_globals": {},
        "execute_script": True,
        **failure,
    }

    assert (
        build.module.paraview_scraper(
            block, variables, {"src_dir": str(build.output.parent.parent)}
        )
        == ""
    )
    assert not build.output.exists()
    build.export.assert_not_called()
    build.render.assert_not_called()


@pytest.mark.parametrize("changed", ["example", "library"])
def test_sources_changed_after_build_start_are_rejected_before_rendering(
    visual_build_contract, monkeypatch, changed
):
    from types import SimpleNamespace

    build = visual_build_contract
    build.module.start_visual_build(build.examples, build.output)
    if changed == "example":
        build.source.write_text("value = 2\n")
    else:
        monkeypatch.setattr(build.module, "_library_hash", lambda root: "library-after")
    block = SimpleNamespace(lineno=1, content=build.source.read_text())
    variables = {
        "src_file": str(build.source),
        "example_globals": {},
        "execute_script": True,
    }

    with pytest.raises(RuntimeError, match="changed during the documentation build"):
        build.module.paraview_scraper(
            block, variables, {"src_dir": str(build.output.parent.parent)}
        )
    assert not build.output.exists()
    build.export.assert_not_called()
    build.render.assert_not_called()
