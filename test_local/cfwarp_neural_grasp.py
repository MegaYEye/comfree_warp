"""cfwarp_grasp_test.py — headless runner for asu-iris/comfree_warp's
franka_cube_grasp benchmark, + neural force-corrector hook.

Third-party repo /home/dell/comfree_warp is READ-ONLY: this runner imports
its benchmark logic (test_local/test_franka_grasp.py: phases, IK, backend
switching) and re-implements run_grasp_trial WITHOUT the GUI viewer, so it
can run on the headless box. Nothing in their repo is modified.

Engines: mujoco (CPU reference) | mjwarp (GPU) | comfree (GPU, analytical
contact engine). NOTE: the vendored mujoco_warp kernel source needs
warp-lang 1.14 (1.16+/1.18 codegen rejects an undefined `xmat` symbol in
_src/sensor.py; 1.18 also bundles CUDA 13 > our 12.4 driver).

Neural-dynamics hook (--k): the same z-force corrector as
hybrid_force.py, ported to the cube:
    qfrc_applied[cube z] = k * (S*W_cube - fz_solver)
    fz_solver = qfrc_constraint[cube z dof]  (contact force the solver
    applies to the cube; == W_cube at steady hold)
live from the squeeze phase on. --k 0 (default) = faithful benchmark run.

Backend plumbing (verified in their sources, unmodified):
- their sync_backend_from_mujoco copies ctrl/act/xfrc_applied/qpos/qvel/
  time but NOT qfrc_applied, so the corrector's dof-space force would
  never reach the device; we push it ourselves after their sync
  (backend_data.qfrc_applied is (nworld, nv); same copy convention as
  their ctrl push). Both backends consume it: mujoco_warp adds
  qfrc_applied in _qfrc_smooth (forward.py:953), comfree reuses that
  same smooth module (comfree_core/forward.py imports it).
- get_data_into copies qfrc_constraint BACK, so fz_solver is live on
  warp engines too (one step behind the device step, vs the exact
  mj_step1/step2 split we use on the CPU engine).

Usage:
  /home/dell/venvs/comfree/bin/python cfwarp_grasp_test.py --engine mujoco
      [--k 1.0 --prior-scale 1.3 --steps-scale 1.0 --nworld 1]
Writes /tmp/cfwarp_<engine>_k<k>.json + prints per-phase cube heights.
"""
import argparse
import json
import sys
import time

import numpy as np

import os as _os
REPO = _os.environ.get('COMFREE_WARP_ROOT',
             _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, REPO + '/test_local')
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
import test_franka_grasp as tfg  # noqa: E402  (their benchmark logic)


def make_warp_step():
    """Their step_backend + the missing qfrc_applied push (see module
    docstring). Same sync/step/get_data_into sequence otherwise."""
    wp_mod, _, mjwarp_mod = tfg._import_warp_backends()

    def step(engine, model, data, backend_model, backend_data, step_fn):
        tfg.sync_backend_from_mujoco(engine, model, data, backend_data)
        wp_mod.copy(backend_data.qfrc_applied,
                    wp_mod.array([data.qfrc_applied.astype(np.float32)]))
        step_fn(backend_model, backend_data)
        wp_mod.synchronize()
        mjwarp_mod.get_data_into(data, model, backend_data, world_id=0)

    return step


def run_trial(engine, k, prior_scale, steps_scale, nworld, nconmax, njmax,
              stiffness, damping):
    model = mujoco.MjModel.from_xml_path(tfg.MODEL_PATH)
    data = mujoco.MjData(model)
    tfg._load_keyframe_state(model, data)
    phases = tfg.build_phase_targets(model, data)
    print('timestep: %g  phases: %d' % (model.opt.timestep, len(phases)))

    backend_model, backend_data, step_fn = tfg.make_backend(
        engine, model, data, nworld=nworld, nconmax=nconmax, njmax=njmax,
        stiffness=stiffness, damping=damping)
    warp_step = make_warp_step() if engine != 'mujoco' else None

    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    cube_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_freejoint')
    cube_adr = model.jnt_qposadr[cube_joint_id]
    dofx = int(model.jnt_dofadr[cube_joint_id])
    dofz = dofx + 2
    W_CUBE = float(model.body_mass[model.body_jntadr[cube_joint_id]]) * 9.81

    schedule = [tfg.Phase(p.name, max(1, int(round(p.steps * steps_scale))),
                          p.arm_target, p.finger_target, p.force_scale)
                for p in phases]

    min_h = float('inf'); max_h = float('-inf')
    step_index = 0
    prev_arm = data.ctrl[:tfg.ARM_DOF].copy()
    prev_finger = float(data.ctrl[tfg.ARM_DOF])
    phase_trace = []
    t_wall0 = time.perf_counter()
    t_sim0 = data.time
    n_corr = 0
    fz_applied = 0.0

    for phase in schedule:
        h0 = float(data.qpos[cube_adr + 2])
        ncon0 = data.ncon
        for local_step in range(phase.steps):
            alpha = tfg._smoothstep((local_step + 1) / phase.steps)
            data.ctrl[:tfg.ARM_DOF] = (1.0 - alpha) * prev_arm \
                + alpha * phase.arm_target
            data.ctrl[tfg.ARM_DOF] = (1.0 - alpha) * prev_finger \
                + alpha * phase.finger_target
            data.xfrc_applied[:] = 0.0
            if phase.force_scale > 0.0:
                t = local_step * model.opt.timestep
                data.xfrc_applied[hand_body_id, :3] = np.array(
                    [phase.force_scale * np.sin(14.0 * t),
                     0.75 * phase.force_scale * np.cos(9.0 * t), 0.0])

            # ---- neural z-force corrector (same accounting as
            # hybrid_force.py: fz_obj = qfrc_constraint on the object's
            # z dof; the correction rides in qfrc_applied).
            # CPU engine: mj_step1/step2 split -> qfrc_constraint is
            # THIS step's, correction acts on the same step (no lag).
            # warp engines: read qfrc_constraint from the previous
            # step's device sync (inherent one-step lag), push
            # qfrc_applied to the device before stepping.
            active = k > 0 and phase.name in ('squeeze', 'lift', 'hold',
                                              'perturb')
            if engine == 'mujoco':
                mujoco.mj_step1(model, data)
                if active:
                    fz_solver = float(data.qfrc_constraint[dofz])
                    fz_applied = k * (prior_scale * W_CUBE - fz_solver)
                    data.qfrc_applied[dofz] = fz_applied
                    n_corr += 1
                else:
                    fz_applied = 0.0
                    data.qfrc_applied[dofz] = 0.0
                mujoco.mj_step2(model, data)
            else:
                if active:
                    fz_solver = float(data.qfrc_constraint[dofz])
                    fz_applied = k * (prior_scale * W_CUBE - fz_solver)
                    data.qfrc_applied[dofz] = fz_applied
                    n_corr += 1
                else:
                    fz_applied = 0.0
                    data.qfrc_applied[dofz] = 0.0
                warp_step(engine, model, data, backend_model, backend_data,
                          step_fn)

            h = float(data.qpos[cube_adr + 2])
            min_h = min(min_h, h); max_h = max(max_h, h)
            step_index += 1
        phase_trace.append(dict(
            phase=phase.name, steps=phase.steps,
            h_start=h0, h_end=float(data.qpos[cube_adr + 2]),
            ncon_end=int(data.ncon), ncon_start=int(ncon0)))
        prev_arm = phase.arm_target.copy()
        prev_finger = phase.finger_target

    t_wall = time.perf_counter() - t_wall0
    t_sim = data.time - t_sim0
    final_h = float(data.qpos[cube_adr + 2])
    # benchmark success, plus a runaway flag: a k>0 wrong-prior corrector
    # can pump the (loosely-held) cube skyward and still 'pass' the
    # height thresholds -- that is a divergence, not a grasp.
    runaway = max_h > 0.5
    success = (final_h > tfg.LIFTED_THRESHOLD
               and final_h < 0.5
               and min_h > 0.12 - 0.005 and not runaway)
    return dict(engine=engine, k=k, prior_scale=prior_scale,
                steps=step_index, W_cube=W_CUBE, n_corr=n_corr,
                final_cube_height=final_h, min_cube_height=min_h,
                max_cube_height=max_h, success=bool(success),
                runaway=bool(runaway),
                wall_s=t_wall, sim_s=t_sim,
                steps_per_s=step_index / t_wall if t_wall > 0 else 0.0,
                phases=phase_trace)


def cube_body_dof_z(model, cube_joint_id):
    return int(model.jnt_dofadr[cube_joint_id]) + 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=('mujoco', 'mjwarp', 'comfree'),
                    default='mujoco')
    ap.add_argument('--k', type=float, default=0.0)
    ap.add_argument('--prior-scale', type=float, default=1.3)
    ap.add_argument('--steps-scale', type=float, default=1.0)
    ap.add_argument('--nworld', type=int, default=1)
    ap.add_argument('--nconmax', type=int, default=256)
    ap.add_argument('--njmax', type=int, default=1024)
    ap.add_argument('--contact-stiffness', type=float, default=0.2)
    ap.add_argument('--contact-damping', type=float, default=0.001)
    a = ap.parse_args()
    r = run_trial(a.engine, a.k, a.prior_scale, a.steps_scale, a.nworld,
                  a.nconmax, a.njmax, a.contact_stiffness, a.contact_damping)
    print('engine=%s k=%.2f prior=%.2f' % (a.engine, a.k, a.prior_scale))
    print('steps=%d  W_cube=%.3f N  corrector-steps=%d'
          % (r['steps'], r['W_cube'], r['n_corr']))
    print('final_cube_height=%.4f  min=%.4f  max=%.4f'
          % (r['final_cube_height'], r['min_cube_height'],
             r['max_cube_height']))
    print('wall %.1fs  sim %.2fs  %.0f steps/s' %
          (r['wall_s'], r['sim_s'], r['steps_per_s']))
    for p in r['phases']:
        print('  %-8s steps %4d  h %.4f -> %.4f  ncon %d -> %d'
              % (p['phase'], p['steps'], p['h_start'], p['h_end'],
                 p['ncon_start'], p['ncon_end']))
    print('success=%s runaway=%s' % (r['success'], r['runaway']))
    tag = '/tmp/cfwarp_%s_k%.2f.json' % (a.engine, a.k)
    json.dump(r, open(tag, 'w'), indent=1)
    print('wrote', tag)


if __name__ == '__main__':
    main()
