# Neural z-force corrector — test results on this repo's benchmarks

Two test cases from this repo (run headless) exercise a neural z-force
corrector of the form

    qfrc_applied[object z] = k * (S*W - qfrc_constraint[object z])

where `k` = authority dial (0 = pure solver, 1 = full model), `S` =
prior scale factor, `W` = object weight, and `qfrc_constraint[obj z]` =
the contact force the dynamics engine applies to the object. The
corrector is an instance of the "hybrid force" architecture: the solver
keeps contact geometry, the model reshapes the force.

Plumbing note (verified in the sources of both backends):
`sync_backend_from_mujoco` does NOT copy `qfrc_applied`, so the
corrector's force must be pushed to the device explicitly after the
sync; both engines then consume it (mujoco_warp adds it in
`_qfrc_smooth`, forward.py:953; comfree reuses that module).

## Test 1 — Franka cube grasp (`test_local/cfwarp_neural_grasp.py`)

Headless port of `test_local/test_franka_grasp.py` (phases
approach/pregrasp/descend/settle/squeeze/lift/hold/perturb, 3320 steps)
with the corrector live from squeeze. Cube 0.15 kg (W = 1.472 N).

CPU mujoco (reference dynamics):

| k    | S    | final h (m) | outcome |
|------|------|-------------|---------|
| 0    | –    | 0.2781      | success (lift 0.1450→0.2911, perturb hold) |
| 0.25 | 1.0W | 0.2781      | success, indistinguishable from k=0 |
| 0.25 | 1.3W | 0.2800      | success, +2 mm vs baseline |
| 1.0  | 1.0W | 0.1450      | **FAIL — corrector cancels the dynamic grip W+m·a during LIFT, cube falls back** |
| 1.0  | 1.3W | 27.52       | **runaway ejection** |

Cross-engine runaway (k=1, S=1.3, final cube height): mujoco 27.5227 /
mjwarp 27.5229 / comfree 27.5294 — same divergence within 7 mm across
all three dynamics engines.

## Test 2 — Allegro hand + cube (`test_local/cfwarp_neural_allegro.py`)

The scene of `test_local/test_throuput_hand.py`
(`benchmark/test_data/allegro/env_allegro_cube.xml`), headless. Allegro
right hand (16 position actuators, kp=0.1), cube 0.343 kg
(W = 3.365 N), keyframe starts in a closed grasp, palm welded to world.
No squeeze/lift phases — isolates the STATIC-HOLD failure mode.

CPU mujoco (3000 steps = 6 s):

| k    | S    | drift        | pen max | outcome |
|------|------|--------------|---------|---------|
| 0    | –    | +0.15 mm     | 0.69 mm | clean (ncon 6) |
| 0.25 | 1.3W | +0.6 mm      | 0.69 mm | clean |
| 0.5  | 1.3W | +1.3 mm      | 0.69 mm | clean |
| 1.0  | 1.0W | +0.2 mm      | 0.69 mm | clean (no lift phase to sabotage — confirms the S=1.0 failure above is phase-dependent) |
| 1.0  | 1.3W | +43 648 mm   | 0.69 mm | **ESCAPED to 43.7 m** |

Cross-engine escape (k=1, S=1.3, final cube height): mujoco 43.7018 /
mjwarp 43.6703 / comfree 43.6816 — same divergence within 32 mm.
Mechanism verified numerically: contact lost early (ncon 6 → 0.6) ⇒
`qfrc_constraint = 0` ⇒ applied = k·S·W = 4.37 N constant ⇒ net
+1.01 N on 0.343 kg = +2.94 m/s² ⇒ ½·2.94·(6−0.7)² ≈ 43.7 m.

comfree baseline (k=0): drift −6.4 mm, penetration 6.83 mm — its
analytical contact model is softer than MuJoCo's (same finding as
test 1: looser grip); its corrector rows match its own baseline.
mjwarp k=0 matches CPU within 0.02 mm. GPU nworld=1 throughput
~225–260 steps/s is host↔device sync bound (per-step Python loop).

## Conclusions

1. **A static, imperfect target (S≠1) at full authority (k=1) is
   unstable in every engine and every grasp** — a constant surplus
   force ejects a loosely-held object; once contact is lost the error
   is unbounded (27.5 m / 43.7 m). Low k (≤0.5) bounds the error
   proportionally.
2. **A correct target (S=1) at k=1 fails exactly where acceleration
   makes the static target wrong** (lift phase: grip must supply
   W+m·a) and is a benign no-op where the object is static.
3. Both failure modes are the corrector faithfully enforcing an
   incomplete model — the fix is in the TARGET (inertia term, learned
   residual), not the engine. Same conclusion as the bowl-grasp study
   in bigrasp_dp (see NOTES_cfwarp_corrector.md there).

## Repro

    pip install warp-lang==1.14.0   # 1.16/1.17 breaks the vendored kernel codegen; 1.18 needs CUDA 13
    export COMFREE_WARP_ROOT=$(pwd)
    python test_local/cfwarp_neural_grasp.py --engine mujoco --k 1.0 --prior-scale 1.3
    python test_local/cfwarp_neural_allegro.py --engine mjwarp   # full CPU matrix by default

(These runners import the repo's benchmark logic read-only; nothing in
the repo itself is modified by a run.)
