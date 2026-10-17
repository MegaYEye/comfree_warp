"""cfwarp_allegro_test.py — second test case from asu-iris/comfree_warp
for the neural z-force corrector: the Allegro right hand + cube scene
(benchmark/test_data/allegro/env_allegro_cube.xml, the scene their
test_throuput_hand.py drives through a GUI viewer; here headless).

Third-party repo is READ-ONLY: this runner imports nothing from it but
loads its scene file. Lives in bigrasp_dp; run with cwd=/home/dell/comfree_warp.

Scene facts (probed): 16 hand joints (ff/mf/rf/th j0-j3), position
actuators kp=0.1 (soft servo), cube 'obj' 7 cm box, mass 0.343 kg
(W = 3.365 N), freejoint 'cube_table', keyframe key0 = closed grasp
(cube center z 0.053 m). Palm welded to world — nothing follows the
cube if the corrector pushes it out of the grip.

Corrector (same accounting as cfwarp_grasp_test.py / hybrid_force.py):
    qfrc_applied[cube z] = k * (S*W_cube - qfrc_constraint[cube z])
live after a 0.5 s settle; k = authority, S = prior scale. The keyframe
starts already grasping, so there is no squeeze/lift — the matrix here
isolates the STATIC-HOLD failure mode (k=1, S>1: constant surplus push
against a hand that cannot move; k=1, S=1: no-op at steady state since
there is no acceleration phase).

Usage:
  /home/dell/venvs/comfree/bin/python cfwarp_allegro_test.py [--k 1.0 --prior-scale 1.3]
Writes /tmp/cf allegro_<k>_<S>.json + prints per-trial stats.
"""
import argparse
import json
import time

import mujoco
import numpy as np

import os as _os
MODEL = _os.environ.get('COMFREE_WARP_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))) + '/benchmark/test_data/allegro/env_allegro_cube.xml'


def hand_cube_geom_sets(m):
    obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'obj')
    obj_body = m.geom_bodyid[obj_geom]
    hand_geoms = set(int(g) for g in range(m.ngeom)
                     if m.geom_bodyid[g] != 0 and m.geom_bodyid[g] != obj_body)
    return {int(obj_geom)}, hand_geoms, int(obj_body)


def pen_stats(m, d, obj_geom, hand_geoms):
    n = 0
    pm = 0.0
    px = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        i1, i2 = con.geom1 in obj_geom, con.geom2 in obj_geom
        if not (i1 or i2):
            continue
        other = con.geom2 if i1 else con.geom1
        if other not in hand_geoms:
            continue
        pen = max(0.0, -con.dist)
        n += 1
        pm += pen
        px = max(px, pen)
    return n, pm / max(1, n) * 1e3, px * 1e3   # ncon, mean pen mm, max pen mm


def run_trial(k, prior_scale, steps=3000, settle=250):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    key = m.key(0)
    d.qpos[:] = key.qpos
    d.qvel[:] = key.qvel
    d.ctrl[:] = key.ctrl
    mujoco.mj_forward(m, d)

    cube_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'cube_table')
    qadr = int(m.jnt_qposadr[cube_j])
    dofz = int(m.jnt_dofadr[cube_j]) + 2
    W = float(m.body_mass[m.body_jntadr[cube_j]]) * 9.81
    obj_geom, hand_geoms, _ = hand_cube_geom_sets(m)

    z0 = float(d.qpos[qadr + 2])
    rec = []
    n_corr = 0
    t0 = time.perf_counter()
    for step in range(steps):
        active = k > 0 and step >= settle
        mujoco.mj_step1(m, d)
        if active:
            fz_solver = float(d.qfrc_constraint[dofz])
            fz_applied = k * (prior_scale * W - fz_solver)
            d.qfrc_applied[dofz] = fz_applied
            n_corr += 1
        else:
            fz_applied = 0.0
            d.qfrc_applied[dofz] = 0.0
        mujoco.mj_step2(m, d)
        if step % 25 == 0:
            ncon_hc, pen_mean, pen_max = pen_stats(m, d, obj_geom, hand_geoms)
            rec.append((float(d.qpos[qadr + 2]), fz_applied +
                        float(d.qfrc_constraint[dofz]), pen_max, ncon_hc))
    wall = time.perf_counter() - t0

    arr = np.array(rec)
    z = arr[:, 0]
    tail = arr[arr[:, 0].size // 2:]          # second half ~ steady state
    fz_tail = tail[:, 1]
    drift = float((z[-1] - z0) * 1e3)
    max_h = float(z.max())
    min_h = float(z.min())
    escaped = bool(max_h > 0.3)               # starts at 0.053; 0.3 = out of hand
    clean = (not escaped) and abs(drift) < 5.0 and arr[:, 2].max() < 2.0
    return dict(k=k, prior_scale=prior_scale, W_cube=W, steps=steps,
                n_corr=n_corr, z0=z0, final_z=float(z[-1]), drift_mm=drift,
                max_z=max_h, min_z=min_h, escaped=escaped,
                pen_max_mm=float(arr[:, 2].max()),
                pen_mean_mm=float(arr[:, 2].mean()),
                ncon_hand_cube_mean=float(arr[:, 3].mean()),
                fz_tail_mean=float(fz_tail.mean()),
                fz_tail_cv=float(fz_tail.std() / (abs(fz_tail.mean()) + 1e-6)),
                wall_s=wall, steps_per_s=steps / wall if wall > 0 else 0.0,
                clean=bool(clean), trace=rec[::4])


def _backend(engine):
    """put_model/put_data/step/get_data_into for the two GPU engines,
    mirroring test_throuput_hand.py's usage (its model/put/capture calls)."""
    import warp as wp
    from comfree_warp import mujoco_warp as mjwarp
    if engine == 'mjwarp':
        return (wp, mjwarp.put_model, mjwarp.put_data, mjwarp.step,
                mjwarp.get_data_into)
    import comfree_warp as cfwarp
    return (wp,
            lambda m: cfwarp.put_model(m, comfree_stiffness=0.1,
                                       comfree_damping=0.001),
            cfwarp.put_data, cfwarp.step, mjwarp.get_data_into)


def run_trial_warp(engine, k, prior_scale, steps=3000, settle=250,
                   nworld=1, nconmax=64, njmax=1000):
    """Same trial on a warp backend: per-step host sync, qfrc_applied
    pushed to the device before every step (their sync does NOT copy
    qfrc_applied — see cfwarp_grasp_test.py's module docstring)."""
    import mujoco as mj
    wp, put_model, put_data, step_fn, get_into = _backend(engine)
    wp.set_device('cuda:0')
    mjm = mj.MjSpec.from_file(MODEL).compile()
    mjm.opt.ccd_iterations = 50
    mjd = mj.MjData(mjm)
    key = mjm.key(0)
    mjd.qpos[:] = key.qpos
    mjd.qvel[:] = key.qvel
    mjd.ctrl[:] = key.ctrl
    mj.mj_forward(mjm, mjd)
    cube_j = mj.mj_name2id(mjm, mj.mjtObj.mjOBJ_JOINT, 'cube_table')
    qadr = int(mjm.jnt_qposadr[cube_j])
    dofz = int(mjm.jnt_dofadr[cube_j]) + 2
    W = float(mjm.body_mass[mjm.body_jntadr[cube_j]]) * 9.81
    m = put_model(mjm)
    d = put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)).astype(np.float32)))
    wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)).astype(np.float32)))
    ctrl = np.tile(mjd.ctrl, (nworld, 1)).astype(np.float32)
    wp.copy(d.ctrl, wp.array(ctrl))
    obj_geom, hand_geoms, _ = hand_cube_geom_sets(mjm)

    z0 = float(mjd.qpos[qadr + 2])
    rec = []
    n_corr = 0
    t0 = time.perf_counter()
    for step in range(steps):
        get_into(mjd, mjm, d, world_id=0)          # device -> host
        active = k > 0 and step >= settle
        if active:
            fz_solver = float(mjd.qfrc_constraint[dofz])
            fz_applied = k * (prior_scale * W - fz_solver)
            mjd.qfrc_applied[dofz] = fz_applied
            n_corr += 1
        else:
            fz_applied = 0.0
            mjd.qfrc_applied[dofz] = 0.0
        wp.copy(d.qfrc_applied,
                wp.array(mjd.qfrc_applied[None].astype(np.float32)))
        step_fn(m, d)
        wp.synchronize()
        if step % 25 == 0:
            ncon_hc, pen_mean, pen_max = pen_stats(mjm, mjd, obj_geom,
                                                   hand_geoms)
            rec.append((float(mjd.qpos[qadr + 2]),
                        fz_applied + float(mjd.qfrc_constraint[dofz]),
                        pen_max, ncon_hc))
    wall = time.perf_counter() - t0

    arr = np.array(rec)
    z = arr[:, 0]
    tail = arr[arr[:, 0].size // 2:]
    fz_tail = tail[:, 1]
    drift = float((z[-1] - z0) * 1e3)
    escaped = bool(float(z.max()) > 0.3)
    return dict(k=k, prior_scale=prior_scale, engine=engine, W_cube=W,
                steps=steps, n_corr=n_corr, z0=z0, final_z=float(z[-1]),
                drift_mm=drift, max_z=float(z.max()), min_z=float(z.min()),
                escaped=escaped, pen_max_mm=float(arr[:, 2].max()),
                ncon_hand_cube_mean=float(arr[:, 3].mean()),
                fz_tail_mean=float(fz_tail.mean()),
                wall_s=wall, steps_per_s=steps / wall if wall > 0 else 0.0,
                trace=rec[::4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=float, default=None,
                    help='single k; omit to run the full matrix')
    ap.add_argument('--prior-scale', type=float, default=1.3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--engine', choices=('cpu', 'mjwarp', 'comfree'),
                    default='cpu')
    a = ap.parse_args()

    pairs = [(a.k, a.prior_scale)] if a.k is not None else \
            [(0.0, 0.0), (0.25, 1.3), (0.5, 1.3), (1.0, 1.0), (1.0, 1.3)]
    results = []
    for k, s in pairs:
        if a.engine == 'cpu':
            r = run_trial(k, s, steps=a.steps)
        else:
            r = run_trial_warp(a.engine, k, s, steps=a.steps)
        results.append(r)
        print('[%s] k=%.2f S=%.2f  W=%.3f N  drift %+.1f mm  z %.4f->%.4f '
              '(max %.4f)  pen max %.2f mm  ncon %.1f  fz_tail %.2f N '
              '  %s%s' %
              (a.engine, k, s, r['W_cube'], r['drift_mm'], r['z0'],
               r['final_z'], r['max_z'], r['pen_max_mm'],
               r['ncon_hand_cube_mean'], r['fz_tail_mean'],
               'ESCAPED' if r['escaped'] else
               ('clean' if r.get('clean') else 'drift'),
               '  (%.0f steps/s)' % r['steps_per_s'], ), flush=True)
        tag = '/tmp/allegro_%s_k%.2f_S%.2f.json' % (a.engine, k, s)
        json.dump(r, open(tag, 'w'), indent=1)
        print('  wrote', tag, flush=True)


if __name__ == '__main__':
    main()
