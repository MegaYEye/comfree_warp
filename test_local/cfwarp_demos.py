"""cfwarp_demos.py — run the remaining demo scenes of asu-iris/comfree_warp
(the ones test_viewer.py rotates through) with the neural z-force corrector:

  leap        benchmark/leap/env_leap_cube.xml   (LEAP hand + 7cm cube,
              no keyframe: cube drops onto the palm and settles - static hold)
  primitives  benchmark/test_data/primitives.xml (125 free bodies; the
              corrector hooks onto ONE resting sphere = virtual support)
  humanoid    benchmark/humanoid/humanoid.xml   (corrector on the ROOT
              freejoint z = levitation force; ctrl=0 so it crumples at k=0)
  probe       engine support: load every demo scene on mjwarp + comfree
  throughput  n_humanoid nworld=256 steps/s on both warp engines

Corrector (same accounting as the other runners in this study):
    qfrc_applied[body z] = k * (S*W - qfrc_constraint[body z])
live after settle; k = authority, S = prior scale.

Third-party repo READ-ONLY (self-contained portable copy; imports nothing
from the package). Run from the repo root or set COMFREE_WARP_ROOT:
  python test_local/cfwarp_demos.py <scene> [--k ..]
  (CPU mujoco; pass --engine mjwarp|comfree for GPU rows)
"""
import argparse
import os
import json
import time

import mujoco
import numpy as np

_ROOT = os.environ.get('COMFREE_WARP_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENES = {
    'leap': _ROOT + '/benchmark/leap/env_leap_cube.xml',
    'primitives': _ROOT + '/benchmark/test_data/primitives.xml',
    'humanoid': _ROOT + '/benchmark/humanoid/humanoid.xml',
    'collision': _ROOT + '/benchmark/test_data/collision.xml',
    'hfield': _ROOT + '/benchmark/test_data/hfield/hfield.xml',
    'floppy': _ROOT + '/benchmark/test_data/flex/floppy.xml',
}


def _obj_geoms_of(m, body):
    return set(int(g) for g in range(m.ngeom)
               if m.geom_bodyid[g] == body)


def _other_geoms(m, body):
    return set(int(g) for g in range(m.ngeom)
               if m.geom_bodyid[g] != 0 and m.geom_bodyid[g] != body)


def pen_on(m, d, obj_geoms, other_geoms):
    n = 0
    pm = 0.0
    px = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        i1, i2 = con.geom1 in obj_geoms, con.geom2 in obj_geoms
        if not (i1 or i2):
            continue
        other = con.geom2 if i1 else con.geom1
        if other not in other_geoms:
            continue
        n += 1
        pen = max(0.0, -con.dist)
        pm += pen
        px = max(px, pen)
    return n, pm / max(1, n) * 1e3, px * 1e3


class Demo:
    """One demo scene + the body the corrector hooks onto."""

    def __init__(self, name, target='auto'):
        self.name = name
        self.m = mujoco.MjModel.from_xml_path(SCENES[name])
        self.d = mujoco.MjData(self.m)
        m = self.m
        if name == 'leap':
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                  'cube_table')
            self.settle = 500
        elif name == 'primitives':
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'sphere00')
            self.settle = 1500
        elif name == 'humanoid':
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root')
            self.settle = 0
        else:
            raise ValueError(name)
        self.qadr = int(m.jnt_qposadr[j])
        self.dofz = int(m.jnt_dofadr[j]) + 2
        self.body = int(m.jnt_bodyid[j])
        self.W = float(m.body_mass[self.body]) * 9.81
        self.obj_geoms = _obj_geoms_of(m, self.body)
        self.other_geoms = _other_geoms(m, self.body)

    def reset(self):
        d, m = self.d, self.m
        if m.nkey:
            d.qpos[:] = m.key(0).qpos
            d.qvel[:] = 0
            d.ctrl[:] = m.key(0).ctrl
        else:
            # no keyframe: qpos0 IS the intended initial pose (leap hand
            # curled, cube at z=0.1) - keep it, zero velocities only
            mujoco.mj_resetData(m, d)
        d.qacc_warmstart[:] = 0
        d.qfrc_applied[:] = 0
        mujoco.mj_forward(m, d)

    def run(self, k, prior_scale, steps=3000, settle=None, rec_every=25):
        self.reset()
        m, d = self.m, self.d
        settle = self.settle if settle is None else settle
        z0 = float(d.qpos[self.qadr + 2])
        rec = []
        t0 = time.perf_counter()
        for step in range(steps):
            active = k > 0 and step >= settle
            mujoco.mj_step1(m, d)
            if active:
                fs = float(d.qfrc_constraint[self.dofz])
                d.qfrc_applied[self.dofz] = k * (prior_scale * self.W - fs)
            else:
                d.qfrc_applied[self.dofz] = 0.0
            mujoco.mj_step2(m, d)
            if step % rec_every == 0:
                n, pm, px = pen_on(m, d, self.obj_geoms, self.other_geoms)
                rec.append((float(d.qpos[self.qadr + 2]),
                            float(d.qfrc_applied[self.dofz]) +
                            float(d.qfrc_constraint[self.dofz]), pm, px,
                            n))
        wall = time.perf_counter() - t0
        arr = np.array(rec)
        z = arr[:, 0]
        tail = arr[len(arr) // 2:]
        fz = tail[:, 1]
        drift = float((z[-1] - z0) * 1e3)
        escaped = bool(z.max() > 1.0)     # demo scenes: 1 m above start
        return dict(scene=self.name, k=k, prior_scale=prior_scale,
                    W=self.W, steps=steps, z0=z0, final_z=float(z[-1]),
                    max_z=float(z.max()), drift_mm=drift,
                    escaped=escaped, pen_max_mm=float(arr[:, 3].max()),
                    ncon_mean=float(arr[:, 4].mean()),
                    fz_tail_mean=float(fz.mean()),
                    fz_tail_cv=float(fz.std() / (abs(fz.mean()) + 1e-6)),
                    steps_per_s=steps / wall, trace=rec[::4])


def _backend(engine):
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


def run_trial_warp(engine, name, k, prior_scale, steps=3000, nworld=1,
                   nconmax=200, njmax=1000):
    """Same trial on a warp backend (per-step host sync; qfrc_applied
    pushed manually - their sync does not copy it)."""
    demo = Demo(name)
    mjm, mjd_ref = demo.m, demo.d
    wp, put_model, put_data, step_fn, get_into = _backend(engine)
    wp.set_device('cuda:0')
    mjm.opt.ccd_iterations = 50
    m = put_model(mjm)
    d = put_data(mjm, mjd_ref, nworld=nworld, nconmax=nconmax,
                 njmax=njmax)
    qpos0 = (mjm.key(0).qpos if mjm.nkey else mjm.qpos0)
    wp.copy(d.qpos, wp.array(np.tile(qpos0, (nworld, 1)).astype(np.float32)))
    wp.copy(d.qvel, wp.array(np.zeros((nworld, mjm.nv), np.float32)))
    if mjm.nu:
        ctrl0 = (mjm.key(0).ctrl if mjm.nkey else np.zeros(mjm.nu))
        wp.copy(d.ctrl, wp.array(np.tile(ctrl0,
               (nworld, 1)).astype(np.float32)))
    settle = demo.settle
    z0 = float(qpos0[demo.qadr + 2])
    rec = []
    t0 = time.perf_counter()
    for step in range(steps):
        get_into(mjd_ref, mjm, d, world_id=0)
        active = k > 0 and step >= settle
        if active:
            fs = float(mjd_ref.qfrc_constraint[demo.dofz])
            mjd_ref.qfrc_applied[demo.dofz] = k * (prior_scale * demo.W -
                                                    fs)
        else:
            mjd_ref.qfrc_applied[demo.dofz] = 0.0
        wp.copy(d.qfrc_applied,
                wp.array(mjd_ref.qfrc_applied[None].astype(np.float32)))
        step_fn(m, d)
        wp.synchronize()
        if step % 25 == 0:
            n, pm, px = pen_on(mjm, mjd_ref, demo.obj_geoms,
                               demo.other_geoms)
            rec.append((float(mjd_ref.qpos[demo.qadr + 2]),
                        float(mjd_ref.qfrc_applied[demo.dofz]) +
                        float(mjd_ref.qfrc_constraint[demo.dofz]), px, n))
    wall = time.perf_counter() - t0
    arr = np.array(rec)
    z = arr[:, 0]
    return dict(scene=name, engine=engine, k=k, prior_scale=prior_scale,
                W=demo.W, steps=steps, z0=z0, final_z=float(z[-1]),
                max_z=float(z.max()),
                escaped=bool(z.max() > 1.0),
                pen_max_mm=float(arr[:, 2].max()),
                fz_tail_mean=float(arr[len(arr) // 2:, 1].mean()),
                steps_per_s=steps / wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene', choices=list(SCENES) + ['all'])
    ap.add_argument('--k', type=float, default=None)
    ap.add_argument('--prior-scale', type=float, default=1.3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--engine', choices=('cpu', 'mjwarp', 'comfree'),
                    default='cpu')
    a = ap.parse_args()

    names = list(SCENES) if a.scene == 'all' else [a.scene]
    for name in names:
        if a.engine == 'cpu' and name in ('hfield', 'floppy',
                                          'collision'):
            continue
        if a.engine != 'cpu':
            pairs = [(a.k, a.prior_scale)] if a.k is not None else \
                    [(0.0, 0.0), (1.0, 1.3)]
            for k, s in pairs:
                r = run_trial_warp(a.engine, name, k, s, steps=a.steps)
                print('[%s] %s  k=%.2f S=%.2f  W=%.2f N  z %.3f->%.3f '
                      '(max %.2f)  pen %.2f mm  fz_tail %.2f N  %s  '
                      '(%.0f steps/s)' %
                      (a.engine, name, k, s, r['W'], r['z0'],
                       r['final_z'], r['max_z'], r['pen_max_mm'],
                       r['fz_tail_mean'],
                       'ESCAPED' if r['escaped'] else 'held',
                       r['steps_per_s']), flush=True)
                json.dump(r, open('/tmp/demo_%s_%s_k%.2f_S%.2f.json' %
                                  (name, a.engine, k, s), 'w'), indent=1)
            continue
        demo = Demo(name)
        pairs = [(a.k, a.prior_scale)] if a.k is not None else \
                [(0.0, 0.0), (0.25, 1.3), (0.5, 1.3), (1.0, 1.0),
                 (1.0, 1.3)]
        print('=== %s  (W = %.2f N, settle %d steps, dt %g) ===' %
              (name, demo.W, demo.settle, demo.m.opt.timestep), flush=True)
        for k, s in pairs:
            r = demo.run(k, s, steps=a.steps)
            print('  k=%.2f S=%.2f  drift %+.1f mm  z %.3f->%.3f '
                  '(max %.2f)  pen %.2f mm  ncon %.1f  fz_tail %.2f N  '
                  '%s  (%.0f steps/s)' %
                  (k, s, r['drift_mm'], r['z0'], r['final_z'],
                   r['max_z'], r['pen_max_mm'], r['ncon_mean'],
                   r['fz_tail_mean'],
                   'ESCAPED' if r['escaped'] else 'held',
                   r['steps_per_s']), flush=True)
            json.dump(r, open('/tmp/demo_%s_k%.2f_S%.2f.json' %
                              (name, k, s), 'w'), indent=1)


if __name__ == '__main__':
    main()
