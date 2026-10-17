"""cfwarp_neural_allegro_train.py — port the NEURAL dynamics (learned-
residual force corrector) onto this repo's Allegro-hand benchmark and
test whether learning improves on the static prior.

Prior result (cfwarp_neural_allegro.py): at k=1 the static prior
S=1.3W ejects the cube to 43.7 m — once contact is lost the corrector
keeps pushing k*S*W forever. Here the target gets a learned residual:

    F_target = S*W + MLP(ncon, mean_pen_mm, max_pen_mm)     (3-32-32-1)

trained ON TASK at k=1 against the deliberately wrong prior (S=1.3) by
REINFORCE-style ES (antithetic, Adam) on the rollout cost

    J = cv_tail + 3*|bias_tail|/W + 0.02*|drift_mm| + 0.3*pen_max_tail

(the pen term is essential: in a STATIC grasp Newton forces fz_total =
W for ANY constant applied force, so bias alone cannot see a 'press
harder' cheat; the penetration cost can.)

Result: on-task ES (60+30+30 iters, ~10 s) turns the k=1 + wrong-prior
43.7 m divergence into a stable +3.5 mm hold (12 s verified). A
residual trained on a DIFFERENT scene does not transfer (zero-shot
still escapes) — see NEURAL_DYNAMICS.md.

Needs torch. CPU engine only (reference dynamics).
Usage:
  COMFREE_WARP_ROOT=/path/to/comfree_warp \
    python test_local/cfwarp_neural_allegro_train.py [--iters 60 ...]
Writes allegro_target.pt + allegro_neural_results.json next to this
file (or in $CFWARP_TRAIN_OUT if set).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('COMFREE_WARP_ROOT',
                       os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import json
import time
import argparse

import numpy as np
import mujoco
import torch
import torch.nn as nn
from cfwarp_neural_allegro import MODEL, pen_stats, hand_cube_geom_sets

torch.set_num_threads(8)
OUT = os.environ.get('CFWARP_TRAIN_OUT', _HERE)
ZERO_SHOT_CKPT = os.environ.get('CFWARP_ZERO_SHOT_CKPT', '')


class Target(nn.Module):
    """F_target(g) = prior + residual(g), prior = S*W (deliberately wrong)."""

    def __init__(self, W, prior_scale=1.3):
        super().__init__()
        self.prior = prior_scale * W
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def f(self, g):
        with torch.no_grad():
            r = float(self.net(torch.tensor(g, dtype=torch.float32)
                               .view(1, -1)))
        return self.prior + r


class Const:
    """Static prior only (the baseline corrector)."""

    def __init__(self, W, s):
        self.v = s * W

    def f(self, g):
        return self.v


class Scene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MODEL)
        self.d = mujoco.MjData(self.m)
        self.key = self.m.key(0)
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                              'cube_table')
        self.qadr = int(self.m.jnt_qposadr[j])
        self.dofz = int(self.m.jnt_dofadr[j]) + 2
        self.W = float(self.m.body_mass[self.m.body_jntadr[j]]) * 9.81
        self.obj, self.hand, _ = hand_cube_geom_sets(self.m)

    def reset(self):
        d = self.d
        d.qpos[:] = self.key.qpos
        d.qvel[:] = self.key.qvel
        d.ctrl[:] = self.key.ctrl
        d.qacc_warmstart[:] = 0
        d.qfrc_applied[:] = 0
        mujoco.mj_forward(self.m, d)

    def rollout(self, target, k, steps, settle=250, rec_every=25):
        """keyframe grasp -> settle (corrector off) -> live; corrector
        injected between mj_step1/mj_step2 (forces phase)."""
        self.reset()
        d = self.d
        z0 = float(d.qpos[self.qadr + 2])
        rec = []
        for step in range(steps):
            mujoco.mj_step1(self.m, d)
            if k > 0 and step >= settle:
                g = np.asarray(pen_stats(self.m, d, self.obj, self.hand),
                               np.float32)
                ft = target.f(g)
                fs = float(d.qfrc_constraint[self.dofz])
                d.qfrc_applied[self.dofz] = k * (ft - fs)
            else:
                d.qfrc_applied[self.dofz] = 0.0
            mujoco.mj_step2(self.m, d)
            if step % rec_every == 0:
                n, pm, px = pen_stats(self.m, d, self.obj, self.hand)
                rec.append((float(d.qpos[self.qadr + 2]),
                            float(d.qfrc_applied[self.dofz]) +
                            float(d.qfrc_constraint[self.dofz]), px, n))
        return np.array(rec), z0


def jcost(rec, z0, W):
    """Rollout cost (lower better). Tail = 2nd half of the samples."""
    tail = rec[len(rec) // 2:]
    fz = tail[:, 1]
    pen = tail[:, 2]
    cv = fz.std() / (abs(fz.mean()) + 1e-6)
    bias = fz.mean() - W
    drift = abs(rec[-1, 0] - z0) * 1e3
    return cv + 3 * abs(bias) / W + 0.02 * drift + 0.3 * pen.max()


def train(scene, iters, sig, lr, steps, k, theta0=None, seed=0,
          log_every=4):
    tgt = Target(scene.W, 1.3)
    if theta0 is not None:
        tgt.load_state_dict(theta0)
    else:
        # tiny noise on the last layer: zero-init + deterministic sim
        # would make J+ == J- exactly at iter 0 -> zero ES gradient
        with torch.no_grad():
            for p in tgt.net[-1].parameters():
                p.add_(1e-2 * torch.randn_like(p))
    opt = torch.optim.Adam(tgt.net.parameters(), lr=lr)
    names = [n for n, _ in tgt.named_parameters()]
    rng = np.random.default_rng(seed)
    best = (float('inf'), None)
    t0 = time.time()
    for it in range(iters):
        pmap = dict(tgt.named_parameters())
        center = {n: p.detach().clone() for n, p in pmap.items()}
        eps = {n: torch.randn_like(p) for n, p in pmap.items()}
        with torch.no_grad():
            for n, p in pmap.items():
                p.add_(sig * eps[n])
        rec1, z0 = scene.rollout(tgt, k, steps)
        J1 = jcost(rec1, z0, scene.W)
        with torch.no_grad():
            for n, p in pmap.items():
                p.sub_(2 * sig * eps[n])
        rec2, _ = scene.rollout(tgt, k, steps)
        J2 = jcost(rec2, z0, scene.W)
        with torch.no_grad():          # restore theta
            for n, p in pmap.items():
                p.add_(sig * eps[n])
        # save the params that were ACTUALLY evaluated (center +- sig*eps),
        # not the center -- hold/escape is bistable here, the center can
        # escape while a neighbour holds
        if J1 <= J2 and J1 < best[0]:
            best = (J1, {n: center[n] + sig * eps[n] for n in names})
        elif J2 < best[0]:
            best = (J2, {n: center[n] - sig * eps[n] for n in names})
        g = (J1 - J2) / (2 * sig)      # antithetic PG estimate of grad J
        opt.zero_grad()
        for n, p in pmap.items():
            p.grad = g * eps[n]
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            drift1 = abs(rec1[-1, 0] - z0) * 1e3
            print('  train %3d  J %8.2f / %8.2f  best %8.2f  '
                  'drift %+.1f mm  (%.0fs)' %
                  (it, J1, J2, best[0], drift1, time.time() - t0),
                  flush=True)
    tgt.load_state_dict(best[1])
    return tgt, best[0], best[1]


def eval_row(scene, target, k, steps, tag):
    rec, z0 = scene.rollout(target, k, steps)
    tail = rec[len(rec) // 2:]
    fz = tail[:, 1]
    z = rec[:, 0]
    cv = fz.std() / (abs(fz.mean()) + 1e-6)
    bias = fz.mean() - scene.W
    drift = (z[-1] - z0) * 1e3
    escaped = bool(z.max() > 0.3)
    r = dict(tag=tag, k=k, drift_mm=float(drift),
             max_z=float(z.max()), escaped=escaped,
             pen_max_mm=float(rec[:, 2].max()),
             fz_tail_mean=float(fz.mean()), fz_tail_cv=float(cv),
             bias_n=float(bias))
    print('%-28s drift %+.1f mm  max_z %.3f m  pen %.2f mm  '
          'fz_tail %.2f N (W %.2f) cv %.3f  %s' %
          (tag, drift, z.max(), rec[:, 2].max(), fz.mean(), scene.W, cv,
           'ESCAPED' if escaped else 'held'), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--sig', type=float, default=0.4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--train-steps', type=int, default=2500)
    ap.add_argument('--eval-steps', type=int, default=3000)
    a = ap.parse_args()

    scene = Scene()
    W = scene.W
    print('allegro neural-dynamics port  (W = %.3f N, prior 1.3W)' % W,
          flush=True)
    print('=== baselines (reference dynamics, same process) ===',
          flush=True)
    rows = []
    rows.append(eval_row(scene, None, 0.0, a.eval_steps, 'k=0 solver only'))
    rows.append(eval_row(scene, Const(W, 1.3), 1.0, a.eval_steps,
                         'k=1 static 1.3W (fails)'))
    rows.append(eval_row(scene, Const(W, 1.0), 1.0, a.eval_steps,
                         'k=1 static 1.0W (oracle prior)'))

    if ZERO_SHOT_CKPT and os.path.exists(ZERO_SHOT_CKPT):
        print('=== zero-shot transfer of a foreign-scene checkpoint ===',
              flush=True)
        tz = Target(W, 1.3)
        tz.load_state_dict(torch.load(ZERO_SHOT_CKPT, map_location='cpu'))
        rows.append(eval_row(scene, tz, 1.0, a.eval_steps,
                             'k=1 learned (foreign ckpt, zero-shot)'))

    print('=== train on task: prior 1.3W, k=1, antithetic ES ===',
          flush=True)
    tgt, bestJ, best_sd = train(scene, a.iters, a.sig, a.lr, a.train_steps,
                                k=1.0)
    # anneal: resume from the best with a smaller sigma, keep the winner
    for round_, (it2, sig2) in enumerate([(a.iters // 2, a.sig / 2),
                                          (a.iters // 2, a.sig / 4)]):
        t2, J2_, sd2 = train(scene, it2, sig2, a.lr, a.train_steps, k=1.0,
                             theta0=best_sd, seed=round_ + 1)
        if J2_ < bestJ:
            tgt, bestJ, best_sd = t2, J2_, sd2
        print('  anneal round %d: best J %.2f' % (round_, bestJ), flush=True)
    ckpt_path = os.path.join(OUT, 'allegro_target.pt')
    torch.save(tgt.state_dict(), ckpt_path)
    print('best train J %.2f  -> %s' % (bestJ, ckpt_path), flush=True)

    print('=== learned corrector, on-task ===', flush=True)
    rows.append(eval_row(scene, tgt, 1.0, a.eval_steps,
                         'k=1 learned (on-task)'))
    rows.append(eval_row(scene, tgt, 1.0, 2 * a.eval_steps,
                         'k=1 learned (on-task, 12 s hold)'))
    rows.append(eval_row(scene, tgt, 0.5, a.eval_steps,
                         'k=0.5 learned (on-task)'))

    print('=== what the residual learned (F_target = 1.3W + r(g)) ===',
          flush=True)
    probe = {}
    for g in [(0, 0, 0), (6, 0.7, 0.7), (6, 2.0, 2.0), (12, 1.0, 1.0)]:
        probe[str(g)] = dict(target=tgt.f(np.array(g, np.float32)),
                             residual=tgt.f(np.array(g, np.float32)) - 1.3 * W)
        print('  g %-16s F_target %7.3f N  (r %+.3f N)' %
              (str(g), tgt.f(np.array(g, np.float32)),
               tgt.f(np.array(g, np.float32)) - 1.3 * W), flush=True)

    res_path = os.path.join(OUT, 'allegro_neural_results.json')
    json.dump(dict(W=W, rows=rows, probe=probe, bestJ=bestJ,
                   iters=a.iters, sig=a.sig, lr=a.lr),
              open(res_path, 'w'), indent=1)
    print('wrote', res_path, flush=True)


if __name__ == '__main__':
    main()
