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

## Test 3 — Learned residual on the Allegro scene (`test_local/cfwarp_neural_allegro_train.py`, needs torch)

Does learning improve on the static prior? The target becomes

    F_target = S*W + MLP(ncon, mean_pen_mm, max_pen_mm)     (3-32-32-1)

trained ON TASK at k=1 against the deliberately wrong prior (S=1.3) by
antithetic REINFORCE ES (Adam 2e-3, sigma 0.4 -> 0.2 -> 0.1, 60+30+30
iterations, ~10 s on CPU) on the rollout cost

    J = cv_tail + 3*|bias_tail|/W + 0.02*|drift_mm| + 0.3*pen_max_tail

The penetration term is essential here: in a STATIC grasp Newton
forces fz_total = W for ANY constant applied force, so the bias term
alone cannot distinguish the correct target from a "press harder"
cheat; penalizing penetration depth can.

| target                        | drift      | outcome |
|-------------------------------|------------|---------|
| k=0 solver only               | +0.1 mm    | held |
| k=1 static 1.3W               | +43 649 mm | ESCAPED 43.702 m |
| k=1 static 1.0W (oracle)      | +0.2 mm    | held |
| k=1 learned, foreign checkpoint (zero-shot) | +43 578 mm | ESCAPED — the residual is task-specific |
| k=1 learned (on-task)         | +3.5 mm    | held, fz 3.37 N (W 3.365), cv 0.006 |
| k=1 learned (on-task, 12 s)   | +3.5 mm    | held, cv 0.010 |
| k=0.5 learned (on-task)       | +0.6 mm    | held |

What the residual learned (F_target = 1.3W + r(g)):
- during contact (ncon 6, pen 0.7 mm): r = -0.78 N -> target 3.595 N
  ~= W — it cancels the 0.3W surplus that causes the escape;
- deeper penetration -> lower target (2.876 N at pen 2 mm): negative
  feedback on squeeze depth, the "press harder" cheat closed off;
- contact-loss gate direction learned but weak (r = -0.20 at ncon=0):
  the exact mechanism of the 43.7 m runaway is a target that keeps
  pushing after contact is lost; here the gate never triggers because
  the surplus is cancelled before contact can drop.

Implementation notes for anyone reproducing:
- best-tracking must save the params ACTUALLY evaluated (theta +-
  sigma*eps), not the center — hold/escape is bistable and the center
  can escape while a scored neighbour holds;
- zero-init last layer + deterministic sim makes J+ == J- exactly at
  iteration 0 (zero ES gradient) — initialize the last layer with
  ~1e-2 noise to break the tie.

## Test 4 — The remaining demo scenes (`test_local/cfwarp_demos.py`)

The scenes test_viewer.py rotates through, run headless with the same
corrector. One run per scene family:

**LEAP hand + cube** (no keyframe: the cube free-falls from z=0.1 onto
the palm and settles there in 0.4 s, 10 contacts, W = 1.373 N):

| k    | S    | outcome |
|------|------|---------|
| 0    | –    | settles on palm, fz_tail = W, held |
| 0.25 / 0.5 / 1.0 | 1.3 / 1.0 | identical to k=0 (static hold) |
| 1.0  | 1.3  | **ESCAPED to 36.07 m** |

Cross-engine escape (final z): mujoco 36.07 / mjwarp 36.04 / comfree
36.02 — same divergence within 50 mm. The learned residual (test-3
recipe, ~5 s of ES) holds k=1 at fz = W but in a bounded oscillation
(z within ±3 cm over 18 s, cv 0.097) and without a contact-loss gate
(it never needed one — its surplus cancellation keeps contact).

**primitives** (125 free bodies; corrector on one sphere resting on
the table, W = 5.14 N — no hand at all, the simplest setting):
k=0 rests on the table; k=1 S=1.0 held (no-op); k=1 S=1.3
**ESCAPED to 13.33 m** (sphere launched off the table).

**humanoid — architecture mismatch, N/A.** With ctrl=0 the rig
crumples and the joint springs (knee stiffness 10, damping 5) press
the body into the floor: steady-state qfrc_constraint[root z] ~=
400 N = 7×W. The corrector's static target W is then structurally
wrong: k=1, S=1.3 applies 1.3·57.4 − 400 ≈ **−326 N — it drives the
humanoid INTO the floor**. The corrector presumes the target equals
the task's true static support, which equals W only for a
freely-suspended object; full-body control needs a different target
model. Reported as a negative result, not an engine bug.

**Engine support probe** — every viewer scene loads and steps on BOTH
warp engines at nworld=1 (primitives, collision, floppy/FEM-flex,
hfield, humanoid, n_humanoid, leap, allegro). comfree is consistently
20-30% faster than mjwarp at nworld=1 (humanoid 715 vs 512 steps/s;
primitives 408 vs 23 — mjwarp struggles with 125 free bodies).

**n_humanoid throughput** (48 bodies, device-resident, 100 steps):

| nworld | mjwarp | comfree |
|--------|--------|---------|
| 64     | 15 769 world-steps/s | 25 339 |
| 256    | 60 871 | **102 637 (1.7x)** |

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
4. **A learned residual fixes it, but only on-task** (test 3): 10 s of
   ES on the same scene turns the k=1 + wrong-prior 43.7 m divergence
   into a stable hold; a residual trained on a different scene does
   not transfer (zero-shot still escapes).
5. **The same physics in every demo scene** (test 4): LEAP 36 m,
   primitives 13 m — a constant surplus force at full authority
   ejects any loosely-supported object, hand or no hand. The one
   scene where the architecture does NOT map is full-body control
   (humanoid): the static target W is structurally wrong when the
   task's real support force is not W (spring pre-load, 7×W).

## Repro

    pip install warp-lang==1.14.0   # 1.16/1.17 breaks the vendored kernel codegen; 1.18 needs CUDA 13
    export COMFREE_WARP_ROOT=$(pwd)
    python test_local/cfwarp_neural_grasp.py --engine mujoco --k 1.0 --prior-scale 1.3
    python test_local/cfwarp_neural_allegro.py --engine mjwarp   # full CPU matrix by default
    # test 3 (needs torch):
    python test_local/cfwarp_neural_allegro_train.py
    # test 4 (demo scenes):
    python test_local/cfwarp_demos.py leap          # full k x S matrix
    python test_local/cfwarp_demos.py leap --engine comfree

(These runners import the repo's benchmark logic read-only; nothing in
the repo itself is modified by a run.)
