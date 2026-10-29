# PROJECT.md

## Current: latent-variant screen and flat_fine6 30k run (2026-10-29)

30k run: out/flat-fine6-long-30k-20261028T005516Z (comparison.json/.csv).
flat_fine6 scratch WSD30000 (warmup500/stable25500/cosine4000; Muon 3e-3->3e-5,
AdamW 1e-4->1e-6, EMA .999), seed20261023 with identical initial tensors and
40000-microbatch prefix as the 10k seed23 run; SDE500 tau.01 seed0 K2,
15 targets/13 entries. Post-cooldown 30k AA/CA/TM .3203/.6234/.5711 versus
three-seed 10k flat_fine6 mean AA .1530 (SD .0115): dAA +.167 (14.5 SD),
dCA +.047, dTM +.056; paired minus seed23 10k AA +.154 (15/0/0). All 15
targets improve from this run's 10k to 30k checkpoint. Pre-cooldown
diagnostics: 10k AA .1447 (-.72 SD, inside the three-seed range), 20k .2815.
This corrects the 5k-versus-10k section: a mid-stable peak-LR checkpoint is
not near-zero AA here; at 10k it matched cooled 10k runs. AA still rises
20k->30k (+.039), so quality is not saturated. Cost 1270 ms/update, 10.7 h
end-to-end on one RTX6000Ada, 18.5 GiB allocated, 35.3 s/structure.
Limits: one seed, reused panel, 3x updates and a 2x longer cooldown
(4000 versus 2000) mixed together.

Latent-variant screen: out/latent-variants-scratch-wsd-20261028T052328Z
(comparison.json/.csv). Six scratch WSD10k seed20261023 arms versus the reused
flat_fine6 s23 baseline (AA/CA/TM .1661/.5763/.5224); same sampler/panel.

| arm | comp | d_latent | layers | params | ms/upd | s/struct | AA/CA/TM | dAA |
|---|---|---|---|---|---|---|---|---|
| flat_fine6 | 4 | 512 | 6 | 269.1M | 1280 | 34.7 | .1661/.5763/.5224 | - |
| r4_w1024 | 4 | 1024 | 6 | 376.9M | 1325 | 34.9 | .1553/.5819/.5259 | -.011 |
| r8_w1024 | 8 | 1024 | 6 | 381.1M | 1329 | 34.7 | .1699/.5871/.5174 | +.004 |
| r16_w2048 | 16 | 2048 | 6 | 839.5M | 1660 | 35.0 | .1619/.5827/.5234 | -.004 |
| r8_w1024_pm | 8 | 1024 | 1 | 261.4M | 997 | 23.9 | .1539/.5770/.5197 | -.012 |
| hier_wide (coarse 1024) | 4 | 512 | 6 | 321.2M | 1385 | 39.6 | .1600/.5865/.5262 | -.006 |
| fine_deep | 4 | 512 | 12 | 309.2M | 1597 | 47.8 | .1535/.5802/.5106 | -.013 |

All |dAA| < .025 (seed-noise band): no winner, flat_fine6 retained, and the
winner-replicate step is closed. r8_w1024_pm (only parameter-matched arm) is
-22% ms/update and -31% s/structure at dAA -.012 within noise: a cheaper
candidate, not selected. The first attempt,
out/latent-variants-scratch-wsd-20261028T011116Z, failed GPU preflight
(eval_velocity differs) from first-process GPU nondeterminism in the old-code
re-run; its new-code probe bitwise-matched the sealed historical probe. Fixed
by gating a second current-code probe against the sealed probe after a
discarded warm-up.

Implication: training length dominates latent architecture at this scale;
10k screen rankings may not transfer to long schedules.

Batch-confound probe: out/batch-confound-ddp4-20261028T172456Z
(comparison.json/.csv). flat_fine6 scratch WSD10k seed20261023, torchrun 4 ranks
x (1 protein x 4 copies, accum 1) = 4 distinct proteins / 16 samples per update
versus 1 protein x 16; per-rank training-draw streams (rank 0 unchanged), no LR
scaling. AA/CA/TM .1477/.5943/.5358 versus three-seed 10k mean: dAA -.005
(-.46 SD, inside range; target wins 6/15), dCA +.018 (+6.0 SD), dTM +.021
(+2.3 SD); CA and TM beat every seed paired. 395 ms/update (3.2x faster wall),
4519 s torchrun, 5.02 train GPU-h versus 3.6 per 1-GPU seed (1.4x GPU-h).
Limits: one seed; arm also changes accum 4->1, adds all-reduce and per-rank
streams, length-similar proteins per update, single-process input/LR audits
not recorded. Reading: distinct proteins help backbone/global fold, not AA at
10k. Long eight-GPU production still needs explicit user confirmation.

CI: .github/workflows/check.yml runs syntax-only `python -m compileall -q
scripts src benchmarks tests` on push/PR (seconds). The test suite needs
torch/CUDA and runs only as Slurm jobs on the cluster, never in CI.

## Current: flat_fine6 selected by seed replicates (2026-10-28)

The user authorized five scratch WSD10k single-GPU runs to test whether
flat_fine6's all-atom advantage exceeds seed noise and reproduces under the
current code. Training run: out/seed-replicate-scratch-wsd-20261027T063211Z
(array112836; all five trained 10000 applied/0 skipped/40000 microbatches).
The three flat arms were marked FAILED only by a presence-based admission check
that rejected trainer-default keys recycle_policy=legacy/atom_recycling=none;
the check is now value-based. A follow-up run admitted all five trainings
read-only and evaluated them: out/seed-replicate-eval-20261027T192020Z
(jobs113121-113124 completed0:0; 377 CPU tests). Same panel (15 targets/13
entries), SDE500 tau.01 seed0, K2, one RTX6000Ada for all five new models.

AA/CA/TM chain means: flat_fine6 seeds23/24/25 .1661/.5763/.5224,
.1441/.5737/.5179, .1489/.5796/.5053 (mean AA .1530, SD .0115);
fixed2 seeds23(reused)/24/25 .1271/.5732/.5067, .1313/.5725/.5005,
.1034/.5665/.5059 (mean AA .1206, SD .0150). Paired flat-minus-fixed2 AA
+.039 (13/2 target wins/losses), +.013 (8/7), +.046 (12/3); CA +.003/+.001/
+.013; TM +.016/+.017/-.001. Pooled seed SD .0134; mean paired gap about 2.2x.
New-code flat seed23 reproduces the old run (AA -.0008) with identical initial
tensors and 40000-input stream. flat_fine6 costs 1280 versus 1440-1504 ms per
update and 34.7 versus 41.5 s per structure.

Decision: flat_fine6 is the selected architecture under the pre-declared rule
(AA wins every paired seed, gap above seed noise, CA/TM loss within .01).
Limits: three seeds, a repeatedly used 15-target panel, a 10k schedule and one
protein per update; transfer to longer training or an eight-protein DDP batch
is unmeasured. Next authorized step: one single-GPU flat_fine6 30k run with
the same recipe. Long eight-GPU production remains retired and needs explicit
user confirmation.

## Current: 5k versus 10k checkpoint curve (2026-10-27)

The user authorized re-evaluating the existing own5k EMA checkpoints with the
10k sampler. Run: out/recycle-checkpoint-curve-20261027T033829Z (evaluator
benchmarks/evaluate_recycle_checkpoint_curve.py). Jobs112709/112710/112711
completed0:0;221 CPU tests passed;45 new SDE500 predictions on the same
15 targets/13 entries; 10k rows reused from the admitted three-way run.

AA/CA/TM at5k -> 10k: fixed2K2 .0008/.5033/.4323 -> .1271/.5732/.5067;
sampled16_tail2K2 .0001/.5105/.4388 -> .0622/.5751/.4923;
fixed4K4 .0012/.5212/.4435 -> .1255/.5938/.5184.
fixed2-minus-sampled AA gap: +.0007 at5k (4/11/0 wins/ties/losses) versus
+.0649 at10k (15/0/0). fixed4-minus-fixed2 CA gap: +.0179 at5k, +.0206 at10k.
All models produce near-zero all-atom lDDT at the stable-phase peak learning
rate; the all-atom gap appears between5k and10k. 5k->10k mixes training length
with LR annealing, single seed; it does not prove long-run recovery or cause.
Screens must include the cooldown to rank all-atom quality.
Three-way10k (out/recycle-full4-scratch-wsd-20261026T031310Z, jobs111916-111918
completed0:0): fixed4K4 AA/CA/TM .1255/.5938/.5184 at77.3s/structure and
2382ms/update versus fixed2K2 .1271/.5732/.5067 at41.9s and1441ms; fixed models
collapse off their training K.

## Current: handoff snapshot prepared (2026-10-26)

The user requested a transfer summary. Read out/current-handoff.json for the
portable TXT/JSON packet, including architecture, full15-target sampler results,
historical comparisons, checkpoint/source references and authorized next steps.
The packet is timestamped; refresh live accounting before continuing. Preserve
111916 and its already-queued evaluation111917/111918. Additional5k evaluation,
longer training and narrow-K/full-BPTT candidate arms are proposals, not new jobs.
The workspace contains uncommitted/untracked code and ignored out artifacts;
Git HEAD alone does not reconstruct this state. No new training was submitted
while preparing this handoff.

## Current: provisional scores from completed predictions (2026-10-26)

The user requested a rough comparison immediately. Preserve all running jobs
and frozen sources. Snapshot the intersection of completed target predictions
across available SDE50/500, fixed2K2 and sampledK1/2/4/8; use identical targets
for all ten conditions without selecting on scores. Verify prediction/reference
hashes, checkpoint/EMA/runtime identity and recorded calls, then score the
regular-file copies with the same canonical OST on cpu_only. Limit concurrent
scorers to the allocated four CPUs. Report provisional AA/Cα/TM and measured
sampler cost with the exact common cohort. This does not admit or replace
unfinished full evaluations.

Interim112027 completed0:0:110 canonical structures, common11 targets/9entries,
all ten SDE50/500 model/depth conditions. Additional112028 completed0:0 scores
fixed2K2 atSDE100 on the same11 targets. Combined provisional table:
out/recycle-step-interim-20261026T061834Z/comparison_with100.csv.
SDE100 fixed2 AA/CA/TM=.119364/.548182/.470545 at8.43286s versus500
.111091/.565909/.497636 at41.4430s. CA and TM each fall on9/11 targets;
SDE50 loses substantial quality. Sampled-tail2 deep inference loses AA quality.
Full-panel and remaining100/200 conditions continue; no sampler is selected.

## Current: compare SDE50/100/200 against the preserved SDE500 study (2026-10-26)

The user authorized supplementary sampler-step evaluation with the same own10k
fixed2 and sampled16_tail2 EMA checkpoints. Preserve running500 inference111959
and score111960, K4 training111916 and combined evaluation111917/111918. Retain
fixed2 inferenceK2 and sampled inferenceK1/2/4/8, the admitted15-target/13-entry
panel, SDE tau.01, target-derived seed0, model source, runtime and checkpoint
identities. Generate225 new structures:75 for each of50/100/200 sampler steps.
Do not retrain or repeat the full500-step inference. Reuse its admitted scores
once available, through its original frozen verifier, with exact checkpoint,
panel, source and sampler provenance. Score new steps independently; the final
comparison report waits for both new scoring and existing111960 success.

Make the existing inference helper and audit accept a positive integer n_steps,
default500, and check exactly n_steps+1 velocity calls and K times that many
recurrent core calls. Keep old frozen sources and historical defaults intact.
Run all three new step counts and all model/depth combinations on one physical
RTX6000Ada. Existing500 timing can come from another physical GPU of the same
class; record both bindings and qualify speed ratios when UUIDs differ. Report
canonical AA-lDDT, bb_lddt/Cα and TM, paired target-wise changes against500,
inference time, speedup and allocated/reserved memory, retaining zero scores.
Same initial-noise seed does not couple the entire SDE noise path across
different grids. This remains a single-seed exploratory study, without an
automatic sampler/architecture selection. Freeze source/config before jobs;
all ML/checkpoint/test work stays in Slurm, CPU-only work uses cpu_only.

Step-study phase: out/recycle-sampler-steps-20261026T055517Z. CPU111970
completed0:0 with185 tests and no skips; source/model/panel identity passed.
New inference111973, canonical scores111974 and comparison111975 are submitted.
Comparison waits afterok111974:111960 and reuses the admitted500 report.
All166 static files (159 code) are sealed by source manifest
a0ecb8b86c0e246bf78982796342d9257f107864e3c2a18aa96863f97a08bdca.

## Current: evaluate the completed recycle pair immediately (2026-10-26)

The user authorized a separate evaluation of the two completed own10000-update
EMA checkpoints while preserving K4 training111916 and combined evaluation
111917/111918. Evaluate fixed2 at inferenceK2 and sampled16_tail2 atK1/2/4/8,
75 predictions on the unchanged eligible15-target/13-entry panel. Reuse its
sealed sequence-homology admission, packed inputs and references in place;
verify the original frozen pair source, training receipts, exact40000 delivered
inputs, identical fresh initialization and runtime before prediction. No new
training, teacher, checkpoint transfer, sampler change or architecture change.

Freeze this supplementary evaluator and protocol separately. Use the same
SDE500/tau.01/seed0, input-derived noise seeds, canonical stereocheck-enabled
OST and one physical RTX6000Ada for all five model/depth combinations. Report
AA-lDDT, canonical bb_lddt as Cα, TM, geometry, recorded training cost and
measured inference time/allocated/reserved memory. Compare the two models
directly atK2, and sampled depths againstK1. This is a single-seed exploratory
panel, not FLOPs-matched training or an automatic architecture selection.
Keep all ML/checkpoint/test work in Slurm and CPU-only jobs on cpu_only.

Supplementary phase: out/recycle-completed-pair-eval-20261026T045801Z.
CPU111957 completed0:0 with85 tests and no skips; original panel validated.
Inference111959 is independent of K4; CPU scoring/report111960 follows it.
All163 static files (157 code) are sealed by source manifest
70d51b97bb31a1512a3793ba761972e0d5a37bb2f112c9a41653c62399822926.

## Current: add a matched K4 full-BPTT arm (2026-10-26)

The user explicitly authorized a third scratch WSD10000-applied-update arm
while preserving both running111798 training tasks and their frozen source.
Add `recycle_policy=fixed4` to the existing enum. Keep the exact parameter
topology, random initialization, prelude3, shared residue6/fine6/r4, atom carry,
live R0+PLM boundary, original PLM injection indices5/10, data recipe and Muon
settings of the current pair. Keep nominal construction latent_loops2/n_trunk15
and execute four passes through the runtime policy; changing nominal n_trunk
would also move PLM injection positions and is not an isolated depth change.
The fixed4 default runtime budget is4, with no new parameters or RNG draws.

All four recurrent states retain their autograd graph. LiFT supervises u1..u4
with the original complete randomized depth grid, mean-over-four trajectory
loss, .01 live u0 auxiliary, one final-based GT alignment and final structural
loss. Accept explicit full-trajectory or valid tail-only supervision metadata,
while rejecting malformed/orphan/prefix-graph metadata. Preserve all older
policy arithmetic, checkpoint behavior and default legacy path. Verify K4
final-only gradients through early recurrent residue/atom states and initial
encoder/prelude/PLM paths, full-loss gradients and native max-shape memory.

Use the same fresh folding initialization (no donor/teacher/resume), seed,
40000 ordered delivered microbatches, all-trainable Muon .003/AdamW .0001,
WSD500/7500/2000, copies4/accum4, own EMA and exact10000 applied-update cap.
Require frozen CPU/tests/panel gates and zero-update native backwards before
formal training. All ML/checkpoint/tests stay in Slurm; CPU-only uses cpu_only.

Compare all three own10k EMA models on the unchanged eligible15-target/13-entry
panel at inference K1/2/4/8/16 with the same SDE500/tau.01/seed0 and one physical
GPU. Preserve the original pair training/admission artifacts and reuse its
pending evaluation through a combined common-GPU stage if possible, avoiding
duplicate old-pair inference. Only pending evaluation scheduling may be rewired
for this authorized three-way comparison; do not stop/restart either training.
Report canonical OST AA/Ca/TM, geometry entries, training/inference speed and
allocated/reserved memory. Raw LiFT loss allocation differs across policies;
same10k/data is not matched FLOPs. No automatic architecture selection.

## Current: variable recycling depth with a two-pass gradient tail (2026-10-26)

The user authorized continuing the proposed next comparison. Preserve all prior
campaigns, checkpoints and scores. The completed atom-feedback pair improves
carry over reset (AA .144133 vs .132333, CA .583333 vs .576533, TM .519867 vs
.512000), but carry does not beat the earlier fine6 AA .166867 and has more
stereochemical violations. Carry is an exploratory recurrence base, not a
selected final architecture.

Prepare a separate matched, fully trainable scratch WSD10000-applied-update
pair: nominal K2/prelude3/shared residue6 interleaved residue3 -> r4/fine6 ->
residue3, with hidden atom carry in both arms. One optional `recycle_policy`
enum retains `legacy` by default and adds `fixed2` / `sampled16_tail2`. Both new
policies use the same masked boundary RMSNorm(previous residue) + live initial
residue/PLM representation, including the existing PLM reinjection projection,
without new parameters. This preserves encoder, prelude and PLM learning when
the carried history is detached. The fresh fixed2 control is required because
this common boundary changes the previous model's computation.

For sampled16_tail2, draw integer K1..16 with P(K) proportional to 1/K using a
deterministic optimizer-step seed shared by ranks and accumulated microbatches.
Compute one complete randomized training depth grid ending at1. Run its first
K-min(K,2) passes without activation gradients, detach carried residue AND atom
states at the boundary, then differentiate the final min(K,2) passes. Retain
global u0 and the original grid coordinates; only the selected suffix readouts
contribute LiFT trajectory loss, averaged over that suffix. Prefix velocities
are diagnostics only. Keep the original .01 u0 auxiliary, one final-target
alignment and final all-atom structural loss. This intentionally changes the
gradient/objective allocation, not just memory; neither improvement nor exact
original-LiFT equivalence is assumed. Coordinates/time remain fixed inside each
velocity call. No teacher, checkpoint transfer, extra geometry loss or cross-call
state cache is introduced. Muon/AdamW and WSD/data settings match the prior pair.

Require CPU contract/tests/homology admission and native K1/2/16 zero-update
backwards plus restored-state semantic probes before formal training. Verify
the gradient cutoff and live initialization/PLM paths, actual sampled budgets,
matching fresh weights and40000 ordered input microbatches. Freeze source/config
before submission. Evaluate each own10k EMA at K1/2/4/8/16 with the same fixed
sampler, eligible15-target/13-entry panel and one physical inference GPU. Report
canonical OST AA/CA/TM, geometry entries, train/inference speed and memory, with
single-seed/data/compute/loss-allocation limits and no automatic selection.
All ML/testing/checkpoint work remains in Slurm; CPU-only jobs use cpu_only.

## Current: matched atom-state recycling comparison (2026-10-25)

The user prioritizes structural quality and authorized a separate scratch WSD10k
comparison of atom feedback, preserving campaign111447 and its artifacts. That
campaign completed both10k runs and canonical15-target/13-entry evaluation:
hierarchy AA/CA/TM=.153733/.580333/.522667; fine6=.166867/.580067/.515800.
These are descriptive single-seed results, with no automatic architecture choice.

New phase: out/atom-feedback-scratch-wsd-20261025T113452Z. Use the active
MambaFoldAllAtom path with one optional enum `atom_recycling=none/reset/carry`;
default none preserves the old model. Both new arms use the same modules,
parameter count and seeded random initial tensors: prelude3, shared residue6
interleaved as residue3 -> r4/fine6 latent -> residue3, K2, full BPTT. Each pass
re-mixes and pools atom features, supplies residual atom-to-residue feedback,
then updates atom features through the decoder body. Reset starts each pass
from the initial encoder atom features; carry uses the preceding decoded atom
features. This isolates hidden atom-state carry, rather than adding capacity
only to one arm. Original FM coordinates/time remain fixed within a velocity
call; prediction heads remain readout branches. Carry never crosses sampler
calls. No K sampling, gradient truncation, teacher, donor checkpoint or loss
change is included. All weights train; Muon/AdamW and WSD settings match111447.

Require CPU source/config/test and whole-chain/EU homology gates; native8
backwards with zero updates; temporary nonzero-head semantic probes with exact
state/RNG restoration; fresh formal initialization; actual40000-input equality;
own10k EMA; and same-physical-GPU paired inference followed by canonical OST.
Recheck the same corpus and require the same15 eligible targets/13 entries.
Record actual atom mixer/decoder/head calls; the recurrent decoder body executes
K times at inference while its velocity head runs only at the final pass.
All tests/model/checkpoint work runs in Slurm; CPU-only work uses cpu_only.
The parent owns admission and submissions. Neither run preparation nor a
finite native probe constitutes evidence of improved structural quality.

## Current: all-scratch WSD architecture screen (2026-10-24)

User explicitly revised the comparison: all12 architectures freshlyrandom
initialized,no10k/70k donor transfer,no teacher/distillation,max10000applied
updates performal model,WSD500warmup/7500stable/2000cosine decay. Native
preflight performs forward/backward only withZEROoptimizer updates,so it
doesnot exceedthe10k cap. Allparameters useLRscale1;Muonpeak3e-3,AdamW
fallbackpeak1e-4,min1e-6;cachedESMC6b inputs remain common pretrainedfeatures.
No scratch architecture isselected yet. Formercontrolr4K2 choice isonly
archived warm-start/adaptation evidence,not thecurrent global choice.
Prepared phase:out/architecture-scratch-wsd-20261024T013403Z.
All12configs andisolated scratch source frozen;CPU110910/main+batch completed0:0
in44s with150tests passed,noskips,12parserroundtrips/408typedmodel fields.
GPUcontrol110911 started2026-10-24 11:05:07KST. Its8nativebackwards passwith
zerooptimizerupdates,model/EMAunchanged,u0routing/detachmentverified;loss1.819–2.143,
gradnorm0.380–1.109. Freshformalinitialhash andfirst8inputsmatchnative.
Control600updates/zero skips parentadmitted:100actualpeakLRupdates atMuon3e-3
/AdamW1e-4;lastwindowloss0.79226,preclipgradnorm9.76134(clip2),
mean1.39741s/update,peakallocated22.11708GiB,datawait2.024ms. Earlystability,
notfull10k orquality evidence. Other11 GPUarray110912 released2026-10-24
11:46KST withthrottle7,max8total. Initiallycontrol/r8/shared running;remaining
9pendingAdaresources. At11:55KST all3formallytraining;control1kwindowloss
0.69382,preclipgradnorm4.25522,1.40549s/update,22.13957GiB. Allfour gradient
groups connected/positivenorm;latent242/242connected,clippedgroupnorm0.001948.
This doesnotprove everyindividualgradientnonzero orfunctionalqualitybenefit.
Allthree arms subsequentlypassed2500appliedupdates;eachof4gradientgroups is
fullyconnectedwithpositivegroupnorm. First10000common deliveredmicrobatches
matchbyteexact across3;full40k/all12 validation remainsrequired. Allthree also
passed600actualLRchecks including100peakLRupdates,zero skips. These areearly
numerical/gradientchecks,notstructuralquality. LiFT finaldirectGTFM coefficient
is1/K;sampledK1:2:1 hasE[1/K]=7/12 versusfixedK2's1/2,soK comparison also
includeslossallocation/conditioning/PLM differences. Seephase
first3_2500_admission.json anddepth_loss_interpretation.json. CPUscorers110913/110914
andaggregate110925 follow. Source/configmanifestSHA
4c16dae2599928eb2e437f85d6b1fb4cd440e99efd2ede61d7938da7e5de4616.
At2026-10-24 13:42KST control/shared ownfresh5k checkpoints arepublished.
Parent independentlyverified5000appliedupdates,zero skips,allactualusedLRs,
fourconnectedpositivegradientgroups and20000byteexact deliveredmicrobatches.
FiveprimaryGPUtasks arenowlive;hierarchy/shared9p3 nativezero-update and
formal freshinitialization alsoverified,withfirst100updatewindowsfinite/noskips.
Thesechecks do notmeasurestructurallDDT. Seephase first2_5000_observed.json,
first2_5000_parent_verified.json andhierarchy_shared9p3_initialization_verified.json.
Atlaunch,supplementary control/shared own5k K2 inference110968 usedtest/A5000;
CPUadapteradmission110967 completed0:0in14s. Allsources independentlyreviewed.
Primaryarray temporarilythrottled6,so1control+6array+1supplement remains<=8.
CPU110969 afterany110968 restores7 onlyafterauxallocationterminal,then scores
onlyonsuccess. Supplement isoutsideprimaryranking;seephase
diagnostics/early-control5k/launch_plan.json andsubmission.json.
Allfiveformal arms subsequentlypassed600appliedupdates/zero skips,including
100actualpeakLRupdates each;first2400deliveredmicrobatches matchbyteexact.
Control/r8/shared eachpassed5000;all50windows,5000actualusedLRrows,four
gradientgroups andcommon20000inputs verified. Ownnumbered5k checkpoints
arepublished. Seephase first5_600_admission.json andfirst3_5000_prefix_verified.json.
At2026-10-24 15:05KST allfiveformalmodels passed2500appliedupdates,
zero skips,allactualusedLRs,fourfullyconnectedpositivegradientgroups,and
10000byteexactcommonmicrobatches. Control/shared passed8000updates with
32000byteexactcommoninputs;shared's first500cooldownupdates alsoverified.
Theirnumbered8k checkpoints arepublished(statonly,no bodyreadinthischeck).
Shared7p4's sixthGPUallocation passedzero-update nativeadmission andfresh
formalinitialization,parentverified. First100appliedupdates/400commoninputs
andactualwarmupLRs verified;6armsawaitresources. Seephase
shared7p4_initialization_verified.json andshared7p4_first100_verified.json.
Seephase first5_2500_admission.json andfirst2_8000_parent_verified.json.
Allfirst3control/r8/shared subsequentlypassed8000updates and32000byteexact
commoninputs;allsixformalmodels passed600updates/zero skips,including100actual
peakLRupdates each. Seephase first3_8000_parent_verified.json and
first6_600_admission.json.
At2026-10-24 15:29:50KST sharedpublisheditsfresh10k completionreceipt.
Parentreusedthefrozen training/native/LRverifiers:10000appliedupdates,
40000finite microbatches,zero skips,allactualWSDLRs,K2exactly10000times,
allfourraw-studentgroupdigests changed. Meanstep1.28300s,peakallocated18.00110GiB,
datawait2.11765ms. Gradientsdirectlyrecordedat5karefullyconnected/positivenorm;
thereisNOdirect10kgradientreceipt. Parentdidnotreadthecheckpointbodyorclaim
all10ktensorsfinite. Seephase shared_10000_training_parent_verified.json.
Primarysharedown5k K2 samplingthencompletedall12structures;PDB/referencehashes,
32752finiteatomcoordinates and501network/decoder+1002corecalls eachverified.
GPUmanifestbindsthesame5kcheckpointbodyasthefiniteauxiliaryadmission.
ActualAdasamplermean30.73307s,peakallocated1.51655GiB;canonicalqualitystillpending.
GPU110912_1continues8k/10kinference;CPU110914_1 followsallocationcompletion.
Seephase primary_shared5000_inference_parent_verified.json.
Control subsequentlycompletedfresh10k;parentfrozenverifiers passeditsfullbudget,
actualWSDLRs andallfourgroupdigestchanges. Its40000actualdeliveredmicrobatches
matchsharedbyteexact. Meanstep1.51150s,peakallocated22.14128GiB. Seephase
control_10000_training_parent_verified.json andfirst2_10000_input_parent_verified.json.
Controlr8 subsequentlycompletedfresh10k;parentandindependentfrozenhelper
audits verified10000actualWSDLRs,40000finite microbatches,zeroskips,fresh
nativebindingandallfourgroupdigestchanges. All40000actualinputsarebyteexact
acrosscontrol/r8/shared. R8mean1.523672s/update,peak21.933403GiB,datawait
2.155899ms. Itsdirectgradientreceiptendsat5k;no10kbody/finiteness/quality
claim. Seephase control_r8_10000_training_parent_verified.json and
first3_10000_input_parent_verified.json. CPU111135 nowoverlapscompletedcontrol
5k/8k canonicalstageswithitsfinalGPUinference;main110913retainsafterok110911
andwaitsafterany111135,verifiedbeforeearlyCPUrelease. Sibling110914_0unchanged.
ThisaddsnoGPUanddoesnotchangefrozenmodel/recipe/scoringcriteria.
CPU111135/main+batch completed0:0in3m34s. Parentverifiedbothcontrol
stages:5k/8k defaultallatom0/0.08025,CA0.592417/0.679333,TM0.60525/0.69875.
All24rawscores/sidecars,48PDBhashes,65504finitepredictedatomcoordinates,
bothcheckpointEMAfiniteandretainedsnapshotswerevalidated. At8ksharedminus
controlallatomis+0.044,CA+0.02625(both11/12chainsimprove),TM+0.03525(9/12).
ThisisnominalK2intermediateevidenceonmatchedfreshdata/updatebudgets,not
10k/globalarchitectureselection. Main110913subsequentlycompleted0:0in4m29s,
with24RESUME/24SCORE;all24earlyrawscoresand24sidecarsremainedbyteidentical.
Seephase diagnostics/incremental-control-canonical-score/main_cache_reuse_verified.json and
diagnostics/incremental-control-canonical-score/parent_verified.json.
Firstfivearmsnowhave5000actualLRchecks/20000byteexactcommoninputsand
fourfullyconnectedpositivegradientgroupsat5k. SharedK1passedzero-update
nativeadmissionand600actualupdates/100peakLRupdateswithnoskips;allseven
armsfirst2400inputsarebyteexact. K1andK2sharedinitialrandomtensorhashes
areidentical,butexecution/lossallocationdiffer. Theseareprefixadmissions,
notfullK1/hierarchyqualityproofs. Seephase first5_5000_admission.json and
first7_600_admission.json.
CPU111062/main+batch completed0:0in2m12s,overlappingcanonicalshared5k/8k
scoringwithGPU10kinference. All24rawscores/identities/means andbothEMAfinite
checks passed. Defaultallatom lDDT0→0.12425;CA-only0.62292→0.70558;TM0.637→0.734.
Stereochemistrycriteriaunchanged. Sourceadapter112lines independentlyreviewed;
mainCPUchild110914_1(actual110916) wasserializedafterGPU110942successandearlyCPU
termination,beforereleasing111062. Siblingdependencyunchanged. Main110914_1
subsequentlycompleted0:0in2m48s,with24RESUME/24SCORE;all24earlyrawscores
and24sidecars remainedbyteidentical. Earlysummary/rawsnapshots areretainedbecause
maincanonicalrerunsrewritesummarytimestamps. Otherarmsfullquality and
common-devicefinalselection remainpending. Seephase
diagnostics/incremental-canonical-score/parent_verified.json andsubmission.json.
Shared ownfresh10k EMA nowhascompletecanonicalK1/K2 results:allatom
0.0201667/0.2904167,CA-only0.688583/0.730083,TM0.729917/0.760083.
AllatomandCAimproveon12/12chains;TMimproves8/12. K2minusK1allatomgainis
0.27025. Samplermeans19.72343/30.84349s,allocatedpeaks1.524435/1.525229GiB.
Parentverifiedall48rawscores/sidecars,96PDBhashes,131008finitepredictedatom
coordinates,summaryarithmetic,calls,checkpoint/protocol/sourcebindingsand
GPU/CPUmain+batch0:0. Checkpointbodies werevalidatedbyCPUscorer,notreloaded
onlogin;thisdoesnotclaimall10ktensorsfinite. ExecutionKchangesconditioning
andPLMinjectionaswellasrecurrence,so thisisapolicybenefit,nopurecausal
recurrenceclaim. Onearm/seed/developmentpanel cannotselecttheglobalarchitecture.
Seephase shared_10000_quality_parent_verified.json and
diagnostics/incremental-canonical-score/main_cache_reuse_verified.json.
Controlownfresh10kK1/K2canonicalresultsare0.269333/0.260583allatom,
0.715/0.713CA,0.743667/0.741583TM. Parentverifiedall48rawscores/sidecars,
96PDBhashes,131008finitepredictedatomcoordinatesandall24resultbindings.
ControlK2minusK1allatomis-0.00875(3better/1tie/8worse). SharedK2minus
controlK2is+0.029833chainmean(8better/4worse);equal-entrymean+0.016833,
8-entrybootstrap95[-0.013543,+0.048625]. SharedK2sampler30.84349svs
control37.56867s,andallocated1.525229vs2.438416GiB,aredistinctAdaallocation
measurements,notyetcommonGPUconfirmation. Only2/12primaryarmscomplete;
globalarchitectureunselected. Seephase control_10000_quality_parent_verified.json
andfirst2_10000_comparison_parent_verified.json.
Controlr8GPU110912_0andCPU110914_0/main+batchalsofinished0:0. Itsownfresh
10kK1/K2allatomis0.270917/0.269333,CA0.714167/0.712750,TM0.738667/0.736583.
Parentverifiedall48rawscores/sidecars,96PDBhashes,131008finitepredictedatoms,
24resultbindingsand16u0inputs/flags/means. Thus3/12primaryarmsarecomplete,
144/600canonicaloutputsverified. R8K2minusK1allatom=-0.001583;R8K2minus
controlK2=+0.00875. SharedK2minusR8K2chainmean=+0.021083,butequal-entry
mean=+0.000292/bootstrap95[-0.035834,+0.034500]. Noarchitectureselected.
Seephase control_r8_10000_quality_parent_verified.json and
first3_10000_comparison_parent_verified.json.
Matched16-inputown10ku0probesalsoverified:controlprelude15FM0.054777→
final0.055452(1/16improve),sharedprelude3FM0.152196→0.057234(16/16improve).
Meanrelativevelocityupdate1.2661%/21.2175%. Theseareown-modelaligned,
different-prelude,train-modeK2diagnostics;firstK1readoutisnotinferenceK1.
Seephase u0_control_shared_10000_parent_verified.json.
ControlGPU110911/main+batchcompleted0:0;post-inferenceCPU/NFSwaitingresolved
withoutintervention.Exactblockingphaseunconfirmed;source/runtimedidnotchange.
AftercontrolandauxGPUswereverifiedterminal,parentraisedarray110912throttle
7→8withschedulerreadback,keepingtheauthorizedmax8GPUs.Frozenprotocoland
recipesunchanged.Seephase array_throttle_after_control.json and
diagnostics/postinfer-control-wait/trace.json.
Tenactualformalinitializationsnowverified:allsourcecheckpointnull,
nativezerooptimizerupdates,34typedmodelfieldsandfirst8formalinputsbound.
SharedfixedK1/K2/K3/K4initialrandomtensorhashesareidentical,whileexecution/
PLMschedule/objectivechange. Twoarmsstillawaitresources.Seephase
first10_initialization_parent_verified.json.
Bothcontrol/shared spikeonthesame8200inputwindow(loss0.75088/0.70798),
withoutLRreset/increaseorskips;shared recovers8400onward. Commoninput-related
FMdifficulty isaleadinghypothesis,notfullyisolatedcausality. Bothmodels
subsequentlyreturntoabout0.4lossat8400onthesameinputstream;preservefrozenrecipes.
Seediagnostics/late-loss-spike/trace.json andrecovery.json.
Supplementary control/shared own5k K2 GPU110968 completed0:0;CPUrecovery110997
completed0:0 after initialpost110969 failed compute-node accounting AFTERrestoring
arraythrottle7. Both12-target canonical outputs,parentbindings andmeans verified:
allatom lDDT0/0;CA-onlybb_lDDT0.59233/0.62308;TM0.60550/0.63667. Noarchitecture
is selected. SamePDB one-target stereoON/OFFprobe111021 completed0:0:default
lddt0 versusOFF0.607,CA0.639unchanged. OFFisdiagnosticonly,notranking.
Root inactivegeometrybond/angleindexing had anextrabroadcastdimension;one-line
repair passedCPU111022/main+batch0:0 with40tests/0skips after red4fail/32pass.
Onlyindexing isO(L);clash remainspairwise. Currentphysicalweights0andfrozen
experimentsunchanged,so thisrepair doesnotexplain orfixcurrentpredictions.
Seephase diagnostics/early-control5k/canonical_results_parent_verified.json,
diagnostics/early-control5k/stereo-counterfactual/receipt.json and
diagnostics/geometry-topology-repair/verified.json.
Preserve70k checkpoint andretired108918.
NewsameGPUconfirmation adapter isseparatelysealed(9files). CPU110935/main+batch
completed0:0in5s,43tests passed,noskips. Independent1198line/50091byte source
reviewpassed;GPU110936 parentreleasedafterCPUgate,stillafter110925all12aggregate.
Slurm removedthefulfilled110935dependency;sealedCPUreceiptguard remains.
FinalCPU110937 follows110936. No architecture
chosen. Seephase README_KO.md,goal_audit.json andconfirmation/submission.json.
Offline winner-export adapter isseparatelyprepared:CPU110957/main+batch0:0,
51tests;actualNFS110958/main+batch0:0,6selectedtests. A realNFS renameat2
failure wasrepaired usingexclusive mkdir/payloadhardlinks/manifest-last commit.
Independentreview andparenthash/JUnitchecks passed. Actualexport stillawaits
all12qualityresults andparentrevieweddecision/report/admission SHAvalues.
Seephase export_review.json andinput_rng_review.json. Trainingrecipesunchanged.
Original goal staysactive;same48-hour horizon andmaximum8concurrent newGPUs.

## Archived warm-start architecture campaign (2026-10-23)

Historical decision: selected control after all12 matched additional10k policies
and common-device confirmation:15-layer residue trunk,512-wide residue3 +
fine-latent4 refiner,compression4,fixedK2,GT LiFT;atom recurrence/Ksampling off.
Long eight-GPU production remains retired. This is a warm-start engineering
selection,not a scratch-convergence or generalization claim.

Final selection observed2026-10-24 10:16KST: GPU110531/main+batch completed0:0
in4h11m23s;CPU110532/main+batch completed0:0 in10m52s. All312 canonical quality
rows and72 extra timing rows pass exact execution/PDB identity verification.
Independent review reproduces12 costs,26 rows,36 effects andselection.
Control/r8 allatom=.719083/.724167,mean target-median37.25639/37.18376s,
allocated2.434870/2.442502GiB. Both pass.01chain/.05entry quality margins
and5% time tie;control wins with419382490 versus420433114 parameters.
Freeze artifact:out/architecture-confirmation-20261023T082124Z/selected_architecture/manifest.json.
Its34-field model.yaml andsame-inode EMA-evaluated checkpoint arebound to
actual config/initializer/frozen source/lock/runtime/decision/report identities.
Root preset already matches all34model fields;no default/model-source change.
Final Korean report:out/architecture-screen-48h-20261023T075200Z/ARCHITECTURE_REVIEW_KO.md.
Source/recipe andselection rules remain immutable;historical export/report audit passed.
This archivedselection issuperseded forglobal choice bythe all-scratch WSD phase.
The12 policies comprise8 parameter structures,not12 distinct weight layouts.
Single-seed/development-panel/nonpreserving-warmstart/mixed-training-GPU and
post-outcome input-tolerance limits remain explicit. No production restart.

Observed2026-10-24 05:48KST: twelve arms completed10000 applied updates and40000 finite
microbatches,zero skips,all four parameter groups changed and full ordered
input/native-prefix bindings passed: control,control_r8,shared,hierarchy,shared9p3,
shared7p4,shared_k1,shared_k3,shared_k4,shared_r8,shared_fine2,shared_sampled. All twelve arms completed
formal scoring. Fine2/K4 Ada training means are1.05035/1.82995s/step,
peaks17.541/26.850GiB. Hierarchy's completion appeared04:56:34KST;all40000
actual ordered inputs match the certified retained stream,and native32 matches
the fresh-training prefix. Independent review confirms all four group digests
differ from initial. K2 was applied10000 times,with zero skips. H100 steady
mean1.61067s/update,peak allocated18.384GiB;end-to-end16442.153s includes
startup/checkpoint writes. The step timer includes loader/H2D,full loss/backward,
clipping,optimizer and EMA,but excludes checkpoint writes. See
diagnostics/observations/completed-training-input-bindings.json.
Hierarchy GPU110537_2,CPU110517_2 scorer andCPU110832 coarse-parameter diagnostic
and their batch steps completed0:0. Final hierarchy10k EMA
probe passed16-row routing/detachment and arithmetic checks,with exact initial
input/time/atom coverage. Prelude/K1/K2 endpoint FM means changed from
.574696/.280236/.283883 initially to.219948/.085879/.029289 at10k.
Each model uses its own final-based target alignment;these are probe losses,
not structure scores. All60 hierarchy structures passed the parent coarse-exact
verifier;independent raw/sidecar inspection found no metric or identity discrepancy.
Final K0/K1/K2 all-atom lDDT=0/0/.374417,backbone=.163833/.770000/.846333.
At2k/5k K2,all-atom=0/.086583 andbackbone=.681750/.795000. The CPU scorer
verified actual checkpoint bytes and EMA tensor identities. Coarse72/72 tensors
changed in both raw student and EMA,covering all15444096 fresh parameters;
endpoint changes do not prove functional contribution. See
diagnostics/observations/hierarchy-u0-progress.json. Initial u0/native
admission is recorded in diagnostics/observations/hierarchy-admission.json.
SampledK original and supplementary K1/2/3 inference/scoring are complete.
No long8-GPU job is running.

Observed2026-10-24 06:41KST: common-device control,control_r8 andhierarchy
completed72 K1/K2 inference outputs;parent verified checkpoint/EMA/sampler,
exact residue/latent/coarse counts andprediction/reference hashes. Independent
artifact review passes forr8 andhierarchy. First-pass hierarchy K1/K2 means
21.96561/35.40448s,peak1.610827/1.611621GiB;these are not final replicated
timing orcanonical quality. Full selection-source review found no blocking
method defect;parent must still verify actual calls because report() does not.
GPU110531 remains RUNNING;CPU110532 andsecondary coarse110871/72 wait.
See diagnostics/observations/{freeze-readiness,common-selection-method-review}.json.

Observed2026-10-24 06:56KST: common inference control/r8/hierarchy/shared
has96 verified outputs. Bounded early canonical CPU110874 andbatch completed
0:0 in2m30,scoring72 structures fromthe firstthree arms. Before itsrelease,
110532's original afterok110531 waspreserved andAND afterany110874 readback
passed;runtime writer guard passed. That fulfilled early dependency isnow
removed bySlurm,leaving the originalGPU gate. Canonical raw caches are reusable
bythe unchanged finalreport;summarytimestamps willberewritten. K1/K2 allatom:
control .719667/.719083,r8 .722500/.724167,hierarchy0/.372167.
Control K2-K1=-.000583,chain-weighted entry-cluster95[-.004625,.005700];r8
+.001667,interval[-.001875,.004250]. The small recurrence gains areunresolved
onthis panel. R8-control K2 chain+.005083,interval[-.001375,.010643].
These arepartial3-of12 common-device results,not afrozen architecture choice;
remaining recipes/repeated timing/finalCPU report remainrequired. See
diagnostics/common-early-score-v1 anddiagnostics/observations/common-first-three-results.json.

Observed2026-10-24 07:13KST: common-device first-pass outputs forfive arms
total120 verified structures;shared7p4 exact execution/identity/PDB checks and
independent artifact review pass. Its K1/K2 means21.63581/30.57834s andpeaks
1.786469/1.787262GiB. Shared3+6 and7+4 bothapply15 full-width residue blocks
and8 fine blocks atK2;first means30.57342/30.57834s,with7+4 using1.18712x
parameters and1.17891x allocatedmemory. These arearchitecture-derived counts
andfirst-pass timings,not measured FLOPs orfinal replicated timing.
Final export preparation bindsall12 actual34-field configurations andexternal
initialization receipts tothe same original10kEMA. Rootpreset resolves tothe
testedcontrol model fields;14 currentmodel/loss/optimizer/runtime/config/lock
files matchthe frozen implementation byte-for-byte. FixedK training usesrandom
LiFT interiorcoordinates;evaluation usesuniform coordinates. Preserve
constructionK before changing inference loops,as sharedprelude layout depends
onconstructionK. No architecture selected orproduction launch/default change.
See diagnostics/observations/{common-compute-cost-progress,final-architecture-export-contract}.json.

Subsequent shared9p3 K1/K2 completion raises common verification to144 structures
across six recipes. Parent and independent checks pass all24 new rows and48 PDB
hashes. First means22.71746/30.59461s,peak1.922127/1.922921GiB. Its K2 also
applies15 full-width residue and8 fine blocks;the3+6/7+4/9+3 first-pass costs
are30.57342/30.57834/30.59461s.
Fine2 then adds24 outputs,total168 across seven recipes,with parent and
independent execution/identity/PDB checks passed. Its first K1/K2 means are
17.33476/26.16583s,peak1.439056/1.440613GiB. Canonical common scores remain72
structures from the first three arms;remaining recipes and replicated timing
are in progress. These first-pass times do not establish final timing precision.

All12 final scored receipts,actual ordered40000 inputs and probe/native prefixes
now pass the admitted input-equivalence consumer;all24 producer/scorer jobs and
batch steps completed0:0. There are744 canonical scored structures,excluding
the36 supplementary sampled-depth outputs. The159-file report stage seal passed.
See diagnostics/observations/all-arms-final-bindings.json.
Sealed CPU aggregate110868 is released;confirmation110531 was rewired to it
before cancelling obsolete held110529. CPU110532 remains after110531. The first
submission created no job because old completed dependencies had expired from
the controller;retry uses independently verified readiness. New report completed
0:0 in12s;common-device GPU110531 started2026-10-24 05:47:44KST. CPU110532
awaits it;architecture is unselected. No frozen recipe or
selection criterion changed. See diagnostics/input-report-transition.json.

The frozen aggregate scope text still says all new arms usedAda. Actual per-row
hardware and resource receipts show control,retained shared andhierarchy onH100;
the other arms usedAda. Preserve the running confirmation's comparison input
SHA5cfb5db490734cd7663f308d3fcc1352ca87a07c9530964f929019483eb679dd.
Use actual hardware records and this correction in the final decision;
same-device inference does not remove the mixed-training-hardware limitation.

Secondary fixed-checkpoint coarse attribution is admitted separately at
diagnostics/hierarchy-coarse-ablation. The310-line external wrapper reuses frozen
inference andzeroes only finite coarse_up output before tanh,with an identity
hook in the fresh full baseline. No weights,training recipe orprimary ranking
change. Syntax checks andfull read-only source review passed;CPU110870 andbatch
completed0:0 in8s,covering hook identity/zero/cleanup/state invariance,three
rejections andwrong-seal rejection. GPU110871 started2026-10-24 07:15:58KST;
CPU110872 follows oncpu_only. Allnine native cases passed:identity velocity,
four-fine-stack full/partial-mask parity andtrueK0 errors areexactlyzero;
complete modelstate andEMA/checkpoint bindings areunchanged. Parent and
independent review passall13 source/receipt seals. Coarse removal changes
velocity onallnine cases,RMS.00882250–.01660189 inmodelvelocity units.
Paired24 SDE500 structures arebeinggenerated;actual quality attribution
remains pending. See diagnostics/observations/hierarchy-coarse-native.json.
Coarse computation stays active,so its timing is not a coarse-skip speed saving.
ProtocolSHA c4b7cae60d9851ef9dbbd1485c679ebefb1b674aef8dda093e13a1658ab1dee0.
This diagnostic runs independently ofcommon-device confirmation110531.

Observed2026-10-24 07:52KST: shared_k1 common K1/K2 adds24 parent-verified
outputs,total192 across eight recipes;independent review passes24 rows and48
actual PDB hashes. First-pass K1/K2 means19.54153/30.58192s andpeaks
1.515749/1.516543GiB. Shared_k3 isbeinggenerated;remaining recipes,timing
rounds andfinal canonical report stillprevent primary architecture freeze.
Secondary coarse GPU110871 andCPU110872 main/batch completed0:0,producing24
canonical paired structures. Freshfull12 predictions arebyte-identical tothe
common hierarchyK2 baseline. Coarsefull versuszero allatom=.372167/.381167,
backbone=.846083/.845917. Full-minus-zero chainmean=-.009,entry-cluster95
[-.013600,-.002700] usingthe unchanged common effects method(seed20261023).
The original secondary equal-entry estimate is-.007,95[-.012083,-.001833]
usingits originalseed0;these aredifferent weighted estimands/methods.
Coarseoutput affectsvelocity butslightly worsenscanonical allatom onthis
fixed10kEMA,12-chain/eight-entry single-seed panel. This doesnotestablish a
retrained coarse-free architecture orremove thelarge gap tocontrol.
CPU rawauditor110882 failedbefore rawparsing becausecompute-node sacct could
notreach itsaccounting database. Preservedv1;reviewed immutablev2 usesa sealed
parent-fetched completed-job receipt. CPU110883 andbatch completed0:0 in2s,
verifyingall24raw/sidecar/pair bindings,144metrics andsummary arithmetic;
independent small-receipt review passesall25source/receipt seals and60aggregate
fields. No training/inference wasrepeated. Bothconditions stillcompute coarse;
no compute-removal speedclaim orprimary selection change. See
diagnostics/observations/hierarchy-coarse-quality.json and
diagnostics/canonical-raw-audit-v2. Production remainsretired.

Subsequent common shared_k3 K1/K2/K3 adds36 verified outputs,total228 across
nine recipes. Parent andindependent review pass36 rows,72actual PDB hashes,
checkpoint/EMA/panel identity andexact501*K residue/latent passes. Its original
construction remainsK3 withprelude3. First-pass means19.54476/30.57957/41.61918s,
maxallocated1.520523/1.521317/1.521317GiB;quality/final timing remainpending.
All12 architecture-only config fragments areprepared under theconfirmation's
prepared_model_configs directory. All408typed values matchactual configs and
roundtrip theexisting parser;no rootpreset/default change ormodel/checkpoint
load. Themanifest separates thenew-trial andretained-shared initialization
driver/helper identities,as their frozen training helpers differ. These fragments
omit training/data/resource settings anddonotreproduce a training recipe orEMA
initialization bythemselves. Final selected YAML/nominalK/checkpoint/EMA and
decision bindings stillawaitthecomplete common-device report andparent freeze.

Common shared_k4 now adds36 verified K1/K2/K4 outputs,total264 across ten
recipes. Parent and independent checks pass all36 rows and72 actual PDB hashes.
Means are19.51482/30.53533/52.48944s;allocated peaks are
1.520523/1.521805/1.521805GiB. These are first measurements;canonical quality
and repeat timing remain pending. ARCHITECTURE_REVIEW_KO.md summarizes all12
original trials and the72 early common canonical scores. Its independent audit
found no numerical or selection-gate blocker. The review explicitly distinguishes
the common EMA donor from unequal initial functions:only control preserves the
initial function. Shared conversion and r8 regrouping change it. No architecture
selected or production launch;the frozen comparison and criteria are unchanged.

Common shared_r8 adds24 verified K1/K2 outputs,total288 across eleven recipes.
Parent and independent review pass24 rows and48 actual PDB hashes. Means
19.58634/30.63765s,allocated peaks1.520987/1.521630GiB. Source and admission
agree that r8 adds1,050,624 parameters through the wider slot-concat projection
and offset embedding. Shared K2 first means for r4/r8 are30.57342/30.63765s;
final timing and canonical quality remain pending. The sampled recipe is the
last full-panel candidate before the two repeat timing rounds.

Common shared_sampled adds thelast24 verified first-pass outputs,total312
across all12 recipes. Parent andindependent artifact audits pass24 rows and48
actual PDB hashes;K1/K2 first means19.57278/30.62442s,allocatedpeaks
1.515236/1.516030GiB. The full-panel CPU canonical cacheprefill110898 was
submittedheld,with original110531 GPU gate preserved andAND afterany110898
readback verified before release. Its immutable12-source/26-directory protocol
and rejection controls pass parent andindependent review. It scores all312
first-pass structures while110531 completes72 extra timing outputs;itdoesnot
change frozen training/selection orrestart production. Final110532 follows
both originalGPU andthiscachewriter. Primary architecture remainsunselected.

Common full-panel CPU110898 andbatch completed0:0 in9m50,verifyingall312
canonical raw/sidecar/PDB bindings and156metricstatisticrecords. Parent reproduced
all780summary statisticfields and36paired-effect records. Same-GPU nominal
allatom:control_r8 .724167,control .719083,9+3 .578083,7+4 .528583,
hierarchy .372167,shared_r8 .371667,fine2 .365667,K3 .358250,shared .338917,
K1 .327333,sampled .307417,K4 .224167. Independent full-panel review passes29files/9215lines/266346bytes,all1872
finite metrics and156statisticrecords/780fields. Independent NumPy arithmetic
reproduces36paired-effect records/936numbers,maxdifference1.11e-16. The first21
repeated timing rows pass parent andindependent exact-call/42actualPDB checks;
frozen timing/quality selection hasnotrun yet.
No architecture selected orproduction restarted. See
diagnostics/observations/common-full-panel-results.json.

All36 round1 timing outputs now pass parent verification;round2 isinprogress.
The prepared checkpointidentity companion binds all12 trainedEMA models,original
donors,actualconfigs andinitializerreceipts with39smallsource seals. Independent
280-line review andparent decisive source checks pass. Both training-family
probes strict-load EMA anddigest state_dict beforeanyforward;their fullhelper
versions differ butprobe/digest semantics match. No checkpointbody wasread on
thelogin node. The Korean report now includes a source-verified forwarddiagram:
atom slotMLPs thenbackbone-stream mixing;15fullresidue blocks;K2 sharednarrow3
plusfine4 latent passes;one finaldecoder at inference andthree LiFT readouts
intraining. Final architecture selection andproduction remainunchanged.

Final-evidence audit2026-10-24 03:05KST checked all12 actual training configs,
initialization receipts and raw/native initial audits. All share33 core training
settings and the same original10k EMA identity;fresh model digests match the
pre-native initial and all parameters are trainable. Native updates were not
carried into these fresh runs. Receipted parameter counts:control419382490,
control_r8 420433114,shared255779866,hierarchy271223962,9+3 327573658,
7+4 303642394,fine2 242436506,shared_r8 256830490. FixedK1/K2/K3/K4 and
sampledK share255779866 parameters;different nominal depths do not add a new
core copy. R8 uses slightly more parameters despite fewer latent positions.
These checks compare existing GPU-produced initial digests,not a new checkpoint
load/hash on login. Final hierarchy quality/common-device efficiency remain
pending. See diagnostics/observations/initialization-training-contract.json.

Delivered architecture audit distinguishes8 parameter structures from12
execution/training recipes. All34 architecture fields in every actual config
match the frozen baseline plus declared arm changes. Independent read-only
review checked12 definitions/initial structures,11 completed training receipts,
11 final sampling files and35 final K manifests(420 rows). Parent rechecked
all132 nominal-K final inference rows,checkpoint/u0 identities and actual
501*K residue/latent passes with501 network/decoder calls. These hooks measure
stack passes,not individual block calls. No architecture mismatch was found;
hierarchy training,u0 and all60 inference/score checks have since passed. See
diagnostics/observations/delivered-architecture-matrix.json.

Parent completed-arm verification now also checks exact coarse stack calls.
The prior parent helper checked residue/latent passes but ignored coarse;
v2 admits501 network/decoder calls,501*K residue/latent forK>0 and501*K
coarse only for hierarchy. K0 must omit all branch keys. Saved35 manifests/
420 nonhierarchical rows(K0..K4),five synthetic hierarchy positives andnine
rejection controls pass;both auxiliary scripts compile. Read-only review of
the full86-line verifier and decisive frozen hooks/sampler/coarse implementation
found no blocking issue. These are parent-only checks;synthetic controls do
not prove future hierarchy execution or learning. Original model/training/
selection sources remain unchanged. Use diagnostics/completed-arm-verifier-v2.py
for the final hierarchy receipt;source/control hashes and fixture bindings are
in diagnostics/completed-arm-verifier-v2-controls.json.

Hierarchy coarse evidence audit found only401/401 connected latent tensors
and a whole-group gradient norm;this does not prove nonzero coarse updates.
Separate checkpoint-only CPU diagnostic now compares all72 fresh coarse tensors
between initial raw student,final student and final EMA,by six components.
CPU110831 and batch completed0:0 in3s;three positive/readout controls and four
rejection controls passed,including scalar hashing and source/protocol seals.
CPU110832 completed0:0 in16s on cpu_only. All72 tensors changed in raw student
andEMA,with zero EMA-only endpoint differences. Independent inspection of the
full1302-line result reproduced all six group counts,L2 andRMS;all11 source seals
and receipts pass. CPU verified the actual10k checkpoint SHA against its u0 and
original K2 manifest before loading. Net endpoint changes do not prove gradient magnitude,output contribution
or quality;an unchanged final student can have changed in between. This secondary
diagnostic does not alter the twelve-arm selection. See
diagnostics/hierarchy-coarse-learning/{protocol,submission}.json.
Shared9p3 and shared7p4 GPU/CPU jobs and batch steps are COMPLETED0:0;each has60 canonical
structures with prediction/reference/raw and executed-depth bindings checked.
Their10k K2 all-atom lDDT is.57583/.52858,backbone.86383/.86108;retained
shared3+6 K2 is.34975/.83725. These are warm-start adaptation results,not an
intrinsic architecture ranking;all twelve arms and common-device confirmation
remain required. Ada training mean1.20294/1.20184s,peak19.218/18.808GiB.
See diagnostics/{shared9p3,shared7p4}-10000-results.json and
diagnostics/observations/completed-training-input-bindings.json.

Control_r8 GPU110516_1/CPU110517_1 and batch steps completed0:0;all60
canonical structures,executed-depth counts and full input/budget bindings
are verified. Additional10k K0/K1/K2 all-atom lDDT=.72725/.72250/.72417,
backbone=.87433/.87375/.87383. Ada inference mean20.947/28.955/36.892s,
peak2.347/2.446/2.447GiB. Nominal K2 is strongest among the eleven scored
recipes,but other candidates and common-device confirmation remain pending.
Same-checkpoint true branch bypass K0 has similar quality and lower latency,consistent with
the full15-block trunk dominating on this panel;this is a secondary ablation,
not a declared architecture/depth winner or evidence of statistical equality.
The common-device nominal-K selection protocol is unchanged. See
diagnostics/control_r8-10000-results.json.

Shared_k3 GPU110516_6/CPU110517_6 and batch steps completed0:0;72 structures
and full delivered-input/budget/execution bindings verified.10k K1/K2/K3
all-atom lDDT=0/.04408/.35683,backbone=.69650/.82767/.84433.
Ada K2/K3 inference mean30.343/41.293s,peak1.517GiB. Nominal K3 is only
.00708 above retained K2-trained3+6(.34975),with mixed training hardware and
different PLM re-injection schedules;this is not an isolated depth effect.
Early2k/5k checkpoints were evaluated at fixed K2,not their nominal K3.
See diagnostics/shared_k3-10000-results.json.

Shared_k4 GPU110516_7/CPU110517_7 and batch steps completed0:0;all72 canonical
structures,full40000 input/budget/probe/native and executed-depth counts passed.
10k K1/K2/K4 all-atom lDDT=0/.000083/.22417,backbone=.56467/.79142/.81750.
Ada K4 inference mean52.721s,peak1.517GiB,versus K3 recipe41.293s/.35683.
Under these recipes,deeper training did not improve10k quality. PLM schedules,
LiFT grids and final-readout loss fractions also differ,so this is not an
isolated recurrence-depth effect. Early2k/5k K2 all-atom values are both0;
these are fixedK2 diagnostics,not nominal K4 learning curves. GPU/CPU outcomes
were reconciled with typed OMS tools. See diagnostics/shared_k4-10000-results.json.

Shared_r8 GPU110516_8/CPU110517_8 and batch steps completed0:0;60 structures
and full delivered-input/budget/execution bindings verified.10k K2 all-atom
lDDT=.37508,backbone=.84467,Ada inference mean30.466s,peak1.524GiB.
Training zero skips,steady1.17877s/step,peak17.799GiB. The r4 reference runs
on H100,so speed/memory compression comparisons await common-device inference.
Control_r4 GPU110537_0/CPU110517_0 and batch steps completed0:0;all60 canonical
structures and full input/budget/executed-depth bindings are verified. Additional
10k K0/K1/K2 all-atom lDDT=.72342/.72025/.72150,backbone=.87358/.87300/.87292.
Same H100 inference mean19.865/26.750/33.979s,peak2.360/2.460/2.459GiB.
Its K0 bypass is close in quality and41.5% faster than K2 on this panel;
this remains a secondary ablation,not a final depth choice. R8 nominal K2
exceeds r4 by.00267,but different training/evaluation GPUs and one seed prevent
an isolated compression-effect claim;common-device inference remains required.
H100 train mean1.63350s,peak22.180GiB;zero skips and all four groups changed.
Stream membership is checked by actual ordered digest,not arm or machine
labels;no input-helper change was needed. See completed-training bindings
and diagnostics/{control,shared_r8}-10000-results.json.

Shared_fine2 GPU110516_9/CPU110517_9 and batch steps completed0:0;all60 canonical
structures,full40000 ordered input/budget/probe/native bindings and executed
depths verified. Additional10k K0/K1/K2 all-atom lDDT=0/0/.36175,backbone=
.16775/.78125/.84467. Ada K2 inference mean26.223s,peak1.444GiB. Its2k/5k K2
all-atom values are0/.09283. Nominal10k quality is.01200 above retained
shared fine4(.34975),with mixed training hardware and one seed;this does not
isolate latent-layer count or establish superiority. Both remain well below
matched control_r4.72150/control_r8.72417. Same-device efficiency confirmation
is still pending. See diagnostics/shared_fine2-10000-results.json.

SampledK completed10000 applied updates/40000 finite microbatches,zero skips,
all four groups changed and full actual/native input binding passed. Actual
K1/K2/K3 counts2549/4961/2490 match independent stdlib reproduction of all10k
draws;all100 logged window means match exactly. MeanK1.9941,total19941 core
passes versus fixedK2's20000. All40000 GT endpoint checks bind4x per-depth
update counts. Ada train mean1.15883s/step,peak22.387GiB. GPU110516_10/CPU110517_10
and batch steps completed0:0;all60 canonical structures and full input/budget/
executed-depth bindings verified.10k K0/K1/K2 all-atom lDDT=0/.09850/.30742,
backbone=.11333/.82367/.83725;Ada K1/K2 mean19.400/30.341s,peak1.517GiB.
Nominal K2 is.04233 below retained fixedK2.34975,with mixed training hardware;
this does not establish isolated sampling causation.2k/5k K2 values0/.01833.
Supplemental GPU110779/CPU110780 and batch steps completed0:0;all36 canonical
same-model K1/2/3 results,source/runtime/score bindings and raw hashes pass.
K1/K2/K3 all-atom lDDT=.09950/.30625/.31267,backbone=.82375/.83725/.83425;
Ada mean20.071/31.327/42.581s,peak1.516/1.517/1.517GiB. K3-K2 chain mean
is+.00642,but equal-entry mean is-.00388,entry-bootstrap95[-.02313,+.01962],
with4 chains better,7 worse,1 tied and35.93% greater mean latency. Repeated
K2 differs from original by-.00117;do not silently substitute it for the
original nominal score. These results do not support promoting sampledK3.
See diagnostics/observations/sampled-range-results.json.
See diagnostics/shared_sampled-10000-results.json.
This25/50/25 prior is fixed training-depth sampling,not adaptive inference
halting. Actual per-step order is not stored,and window endpoint metrics are
not K-conditional means. Native8 admission updates were discarded. See
diagnostics/observations/{sampled-admission,sampled-training-policy}.json.

Supplemental coverage audit found sampledK3 absent from both original screen
and common-device nominal-K confirmation. Completed pairedK1/2/3 inference
fills this gap without more training or changing the twelve-arm criterion.
CPU110778 and batch completed0:0:original frozen source/checkpoint/panel gate,
real36-row completed fixture and6 rejection controls passed. v2 binds exact
sampling/manifest rows,finite positive timing/memory,all36 scored outputs and
runtime protocol/checkpoint/original-manifest identity. Read-only review found
no remaining mandatory fix. GPU110779/CPU110780 and batches completed0:0 in
22m15s/4m37s. Runtime sampled10k EMA SHA binds original u0 and K2 manifest;
all10 source bindings,36 canonical raws and paired means were rechecked.
First GPU submit rejected explicit CPU/memory
flags;corrected job uses the cluster's automatic6CPU/25G allocation.
v1 CPU110777 evidence is preserved and had no GPU submission. See
diagnostics/sampled-depth-range-v2/{protocol,submission}.json.

Separate production60k evaluation110411/12 completed0:0;all36 canonical
structures and source/reference bindings were independently reviewed.
Under the original production/A5000 protocol,K1/K2/K3 all-atom lDDT is
.78733/.78833/.78942. K2 improves over production10k.72808 by.06025 on
average(all12 chains improve),but K2-K1 is only+.00100 versus+.00125 at10k.
The entry-bootstrap95 interval for the60k depth gain is[-.00100,+.00475],
while paired median K2 latency costs25.72% more than K1. Overall learning
improved;increased recurrence benefit is not established. This separate
60k result is not a matched additional10k architecture comparison and does
not evaluate the preserved70k checkpoint. Production remains retired.
See diagnostics/observations/production60k-results.json.

CPU-only geometry recovery diagnostic110757 and its batch completed0:0 in6s.
All144 already scored predictions(12 groups x12 chains),450 input bindings
and3 source bindings are verified unchanged. Reference-observed N-CA/CA-C/
C-O samples are3971 per group,CA-CB3669,identical across groups. Shared3+6
bad-bond totals at2k/5k/10k are22906/5449/2348;9+3 totals6511/1623/821;
7+4 totals13800/2336/1084. Historical original10k has95 bad bonds.
Local geometry recovers alongside canonical quality,but this descriptive
comparison does not isolate adaptation budget,capacity or optimization.
Nominal K1-trained/K1-inference still has2710 bad bonds:train/inference depth
mismatch and malformed interior LiFT targets are not necessary explanations
of nominal damage(K1 uses endpoint s=1). Prelude auxiliary and path changes
remain possible contributors. Aligned FM target source parity with the pinned
SimpleFold reference still holds;no model/loss change is justified by algebra
alone. Canonical metrics and twelve frozen recipes are unchanged. See
diagnostics/geometry-recovery-grid/{verified,summary,trace}.json.

Observed21:23KST: full CPU input replay110579/80 and both batch steps
COMPLETED0:0. Parent admitted sealed certificate2be08b509335beb33a735f47d5662a6800978d97594420baa935db25c956e28a:
all40000 ordered inputs per role,exact non-coordinate tensors,max coordinate
error7.6294e-6 normalized(.00012207A),below the fixed.0016A bound.
Both completed/scored arms(shared/shared_k1) bind their full40000 actual
inputs,budgets and probe/native prefixes to the certified streams. Remaining
arms still require final bindings;aggregate stays held and amended report
undeployed. See diagnostics/full-input-replay-long/admission.json.

K1 GPU110516_5 and scorer110517_5 completed0:0. All60 structures/raw receipts
are verified. At10k,K0/K1/K2 all-atom lDDT=0/.32733/0 and backbone=
.06842/.82867/.60458;K1 is positive on12/12 chains,K2 zero on12/12.
For this K1-trained model,Ada inference K1/K2 mean19.62/30.66s,allocated
1.517/1.517GiB. Extra untrained recurrence does not improve it;depth grids
also differ. This does not select between K1-trained and K2-trained recipes
or establish adaptive-depth performance. Historical initial.73 is stronger,
but matched additional10k control and common-device confirmation remain.
Evidence:diagnostics/shared-k1-10000-results.json. Fine2 passed native8/32
with zero skips and initial16-input u0 routing/detachment;fresh10k is running.

Controlled CPU diagnostic110667 completed0:0 in53s on the same24 K1/K2
predictions with OST2.9.1. Turning off stereochecks changes K1 all-atom
.32733->.77067 and extraK2 0->.57517;backbone scores are unchanged.
All102 canonical inputs and all source SHA bindings remain unchanged.
This isolates a large filtering contribution to the metric,not the training
cause or a repair of invalid geometry. Canonical scores/selection rules stay
unchanged. See diagnostics/shared-k1-10000-stereo/{verified,trace}.json.

The copied-core high-LR hypothesis is refuted: all_atom.py177 explicitly marks
latent_residual.residue_stack as backbone;optim.py33 excludes it from refiner
groups. Actual shared/control role_lrs agree at Muon3e-4/AdamW1e-5 for
backbone. Namespace-based initial audit counts are not optimizer role counts.
Correct routing does not establish optimal LR;do not change classification
on this disproven hypothesis. The geometry training cause remains unresolved.

Frozen-source path audit is recorded in diagnostics/architecture-paths.json:
control executes full15 then[narrow3+fine4]x2;shared executes full3 then
[full6+fine4]x2. Shared removes the separate narrow residue stack. SampledK
keeps PLM indices5/10,but K1 reaches only index5;fixedK1 uses3/6. These
recipes therefore do not isolate recurrence count alone. u0 also changes
with prelude depth. No source,recipe or predeclared selection rule changed.

Conditioning audit confirms ESM is concatenated into the residue input before
block0 in every recipe,including3+6 u0. Additional PLM re-injections differ:
7+4/9+3 cross index5 before u0,while3+6 does not;fixedK1 index3 is in the
core after u0,not in prelude3. Missing ESM is therefore not demonstrated.
Eight update-free EMA probes have exact time lists;the six new-arm input
lists are identical,retained inputs satisfy the admitted coordinate bound.
At10k,u0/final FM is.22055/.02965 for3+6,.14815/.02283 for7+4,
.11097/.02171 for9+3,.20325/.03287 forfixedK1. Each probe aligns its target
using its own final output;these are not errors to one shared target frame.
No isolated PLM effect or immediate new variant follows. See
diagnostics/observations/u0-conditioning-audit.json.

Observed20:54KST: shared_k1 completed all10000 applied updates/40000 finite
microbatches with zero skips;the full ordered input audit and rolling digest
match its completion receipt,and all four parameter groups changed.
Steady training mean=.82096s/step,peak allocated13.598GiB on Ada;do not rank
this against H100 timings. At20:54,post-training probe/inference was still
running and no actual shared_k1 lDDT score existed. All eight new arms'
recorded training windows had finite loss/grad norm and update_applied=1.
Evidence:diagnostics/observations/training-health-20261023T205439+0900.json
under the screen run. Shared3+6 and K1 have completed training;the other
matched trials,full input proof and common-device confirmation remained
required at that observation;the21:23 record supersedes its completion state.

Full-input CPU proof waiting was diagnosed separately from GPU training.
Read-only directory visibility refresh in the existing two CPU allocations
measured17.40 microbatches/s,then1.78/s after withdrawal;reapplication again
measured17.40/s over301.7s in the consumer. Both helpers subsequently ended
COMPLETED0:0 at39520 and the original proof completed40000;whole-helper
consumer mean was15.39/s. Original replay sources/protocol,40000 checks and bounds
are unchanged;the precise NFS cache mechanism is not isolated. Sidecars only
read directory/progress metadata,with1800s/39500 polling stop targets.
Receipts:diagnostics/input-visibility-probe/{reapplication,refresh-complete}.json. This improves
CPU proof throughput,not measured model training speed or proof coverage.

Historical initial references.7280833 and.7300000 use the same original
108918 step10000 EMA weights and panel,but separate production/A5000 and
screen/H100 inference runs with differing model source. Their small score
difference cannot be assigned to GPU or source alone. Neither reference is
the matched additional10k control. See diagnostics/baseline_identity.json.

Pending Ada tasks110516_7/_8/_9/_10 were moved from short to normal QOS after
the verified short eight-job limit blocked them alongside two CPU replays.
Resources,source/configs,10k budgets,deadline and maximum eight concurrent
new GPU trials are unchanged;readback now reports resource waiting. Receipt:
`out/architecture-screen-48h-20261023T075200Z/pending_qos_amendment.json`.

Optional sealed input-equivalence reporting is implemented separately from
the frozen original reporter. CPU admission110586 completed0:0:22 tests pass,
including truncated full audits,changed role receipts,default unequal-stream
rejection and explicit certificate/seal binding. The tolerance is an amendment
frozen before full replay,after shared scores already existed. Full40000 proof
has since passed and was parent-admitted;remaining per-arm final bindings
are still required. Aggregate/confirmation dependencies remain unchanged.
See diagnostics/input-report-admit-v2 under the screen run. Do not deploy
the amended reporter before all twelve completed-arm bindings pass.

Original pending aggregate110529 is now held,with its scoring and confirmation
dependencies unchanged,to prevent the known byte-exact mismatch from failing
and cancelling downstream confirmation. Candidate training is unaffected.
Prepared report consumer diagnostics/input-report-stage-v2 passed CPU110589
COMPLETED0:0:wrong-seal rejection under optimized Python,source integrity,
original frozen-base verification and exact relocated imports all pass.
First wrapper110587 caught cwd-shadow imports;v2 fixes cwd and replaces
operational asserts with explicit exceptions.110588 had a mistyped submit
seal;unchanged v2 sources passed with the programmatically verified seal.
All155 code entries remain identical to22-test admission110586. The stage is
not deployed:every remaining finished-arm binding is still required before
the safe aggregate/confirmation transition;full CPU proof is now admitted.
K4/r8 initial16-input u0 routing/detachment checks passed;fresh10k phases began.

Shared110361 and formal scorer110528 completed0:0;all60 structures verified.
At additional10k,K0/K1/K2 all-atom lDDT=0/0/.34975 and backbone=.16842/
.76875/.83725. K2 improves every chain over K1 but remains below historical
original control.73 on every chain. FixedK2 LiFT first-pass targets are interior
interpolations;K1 inference uses grid[1],K2[.5,1],so this is not an isolated
generic recurrence benefit. See scores/shared.json and diagnostics/shared_recovery.json.
Early input audit caught different raw streams between retained/new hosts.
CPU replays110568/69 and comparison110570 reproduced all16 recorded batches;
noise/time/ESM/discrete tensors are exact,only x_clean/x_t differ by at most
1.91e-6/4.77e-7 normalized in that initial subset. Full40000-microbatch CPU
replays110579/80 completed0:0 with10h/8h allocations,after initial3h110577/78 were cancelled
with partial proof records preserved. Slurm denied owner time-limit increase;
the8h consumer limit permits backfill before higher-priority reservations.
Replay/test code and tolerances are unchanged;5 admission tests passed on
both initial hosts and the replacement producer. Protocol fixes atol1e-4,
rtol0(.0016A per component),with exact non-coordinate checks;this is an
engineering equivalence bound,not identical gradients/trajectories. Original
byte-matching aggregate gate remains unsatisfied and unmodified. The sealed
certificate amendment requires every completed arm's40000 ordered hashes/
budget to bind before aggregate/confirmation transition.
Frozen current proof source/protocol:diagnostics/full-input-replay-long under
the screen run;the initial full-input-replay directory remains preserved.

Follow-up diagnostic110564 completed0:0 on the same12 shared3+6 additional5k
K2 structures:all-atom lDDT=.09108 (12/12 positive),backbone=.79833,TM=.72950.
From2k to5k,bad-bond counts fall22906->5449 and bad-angle counts15486->2891;
N-CA median recovers1.173->1.430A,while C-O remains1.136A versus1.234A in
reference. Recovery is real but still far below historical original control
all-atom.73,which was measured in an earlier run. Matched10k/common-device
results remain necessary. Shared10k actualK0/K1/K2 inference has completed;
its16-microbatch probe passes,with u0 FM=.22055,K1=.08590,K2=.02965.
New control_r8/shared9p3 completed8 native updates/32 finite microbatches each
with zero skips,and their16-microbatch initial u0 routing/gradient probes pass.
Other arms remain running or resource-pending. Recovery evidence:
`out/architecture-screen-48h-20261023T075200Z/diagnostics/shared_recovery.json`.

Early diagnostic110552 completed0:0 on12 already-generated shared3+6
additional2k K2 structures. Canonical all-atom lDDT is0 on every chain;
backbone mean=.68017 and TM mean=.66950. All raw OST results are SUCCESS,
with unchanged2.9.1/reference hashes/standard stereochemistry. Independent
parsing confirms every model residue has a backbone stereochemical issue,
so OST removes all residues before all-atom scoring;backbone scoring uses
unfiltered coordinates. Median matched N-CA/CA-C/C-O distances are
1.173/1.212/.848A versus reference1.458/1.524/1.234A;original control is
1.458/1.526/1.218A. This is real local-geometry distortion,not a missing-score
fallback;its training cause and10k persistence remain unknown. Preserve the
canonical metric and matched recipes;do not select architecture from this2k
diagnostic. Evidence:`diagnostics/shared-2000-K2/{verified,geometry_observation,
residue_filtering}.json` under the frozen screen's output directory.
The paired16-microbatch5k probe now exists: u0 FM.57470->.24209 from initial;
K1/K2 endpoint FM=.09468/.03906,with actualK0 routing/detached-gradient checks
passing. These velocity MSEs do not establish sampled structure quality.

Execution-resource amendment at17:43KST: control/hierarchy are now pending in
H100 array110537 with indices0,2 and concurrency1,using the unchanged frozen
scripts/configs. Their original pending Ada tasks110516_0/_2 were held and
cancelled only after both existing CPU scoring dependencies110517_0/_2 were
rewritten to aftercorr:110537 and read back. Other nine Ada trials retain
array110516,now concurrency7;maximum8 concurrent new trials is unchanged.
GPU resource edits are prohibited by cluster policy,so cancel/resubmit was
required. Completed CPU admission110515 remains the verified admission receipt
and frozen train.sh still requires its marker;its aged controller dependency
could not be reused for submission. Aggregate110529 and confirmation110531/32
retain their original scoring dependencies. Detailed before/after receipts:
`out/architecture-screen-48h-20261023T075200Z/primary_resource_replacement.json`.
Shared110361 has saved its10k checkpoint;post-training probes/structure
inference are still running. Ada arms control_r8 and shared9p3 have started
frozen-source/native admission;no new-arm native pass or structure score is
claimed yet. No additional trial,training budget or recipe changed.

Common-device confirmation is now frozen separately at
`out/architecture-confirmation-20261023T082124Z/README.md`. CPU admission
110530 completed0:0 with20 tests passed,including cached-source rejection,
panel/scorer binding,coverage and inclusive selection-boundary checks. GPU
110531 runs after aggregate110529 andCPU110530;CPU scoring/decision110532
follows it. All12 nominal-K recipes are evaluated on one physical Ada GPU in
one process,with K1/K2 andconfiguredK3/K4:312 scored structures. Two further
nominal-K rounds on shortest/median/longest targets give72 timing outputs;
50 unmeasured sampler warmups make434 SDE500 invocations total. Capture
physical UUID/process/order/runtime;latency includes counting hooks. Three
timing observations exist for those three targets,one for the other nine;
do not claim5% timing precision. GPU walltime10h/deadlineOct25 14:34KST leaves
two hours for CPU scoring before the user horizon. No quality shortlist from
mixed-device preliminary scores is used. Existing snapshots remain immutable.

Before any new scores appear,the engineering selection preference is fixed:
reference=highest common-device nominal-K chain-mean all-atom lDDT;eligible
models lose at most.01 in chain mean and.05 on every PDB-entry mean. Within
5% of fastest eligible mean target-median latency,prefer fewer parameters,
then lower allocated memory,then arm name. These are engineering tolerances,
not statistical noninferiority. Show chain-mean entry-cluster bootstrap and
equal-entry mean/interval separately,including versus-control effects and
K2-minus-K1. One-seed/development-panel/model-selection/mixed-training-GPU/
nonpreserving-warm-start limits remain. Parent must review the recommendation
before architecture freeze;neither scripts nor elapsed48h declare completion.

User authorized preserving70k, retiring108918, then >=10 distinct architecture
trials at additional10k each before architecture selection. Goal horizon ends
2026-10-25 16:34 KST. Full70k checkpoint was hardlinked and independently
verified in cpu_only110513: step70000,finite EMA and optimizer/scheduler/RNG/
data state present,SHA25640802f577a6609d2425c6205de374aae1487a3646dce462be511b2fc43b3e64d.
Parent cancelled108918 after verification;sacct confirms CANCELLED. Preservation
receipt:`out/retired-108918-70k-20261023T073500Z/verified.json`.

The frozen12-arm screen uses original unadapted10k EMA and the same additional
10000 applied updates,seed20261023,batch16,all-backbone training,GT LiFT,
Muon3e-3/3e-4 and AdamW fallback1e-4/1e-5,500warmup,no cooldown,EMA.999.
Arms:current control,control r8,shared3+6,hierarchical3+6,shared9+3,shared7+4,
shared3+6 K1/K3/K4,shared3+6 r8,fine2,and K1/2/3 sampled25/50/25%.
Retain running3+6 direct trial110361;replace pending hierarchy110362 only
after new CPU admission with identical-recipe Ada trial to avoid H100 queue.
Preserve old frozen runs;do not edit their sources or restart main training.

New benchmark runner explicitly validates donor transforms,all applied updates,
each microbatch's actual LiFT endpoint and aggregate objective,actual student
u0 bypass/target detachment,selected-K counts and delivered-input hashes.
No teacher or initial structure-quality gate. r8 repeats source r4 projection
slots/offsets without rescaling;fine2 drops source fine blocks2/3. Shared cores
copy the first suffix group,discard later full-width blocks and old narrow core.
These are warm-start adaptation comparisons,not function-preserving conversions.
K1/3/4 keep prelude3 with nominal depth9/21/27,also changing PLM injection
positions;sampled K keeps K2 placement. Document this confound rather than
claim isolated recurrence gains. No model runtime/loss math changes are needed.

Score actual12-chain/eight-entry development SDE500 structures at2k/5k K2
and10k K0/1/2 plus configured deeperK. Compare nominal-K quality and commonK2,
K2-minus-K1,paired entry uncertainty,initial damage/recovery and same-GPU speed/
memory. All targets have training homologs;one seed and10k do not establish
generalization or final convergence. Existing shared H100 timing cannot be
ranked against Ada;collect common-device measurements for contenders before
efficiency-based selection. Native admission is per candidate before its fresh
10k phase. Maximum8 concurrent single-GPU new trials,each12h walltime;CPU-only
admission/scoring exclusively cpu_only. Architecture choice awaits verified
quality/efficiency evidence;48h elapsed alone does not prove completion.

CPU preparation/admission110515 completed0:0 with135 tests passed andone CUDA
skip;24 configs parsed and185 frozen source artifacts verified. New11-arm
GPU array110516 is resource-pending,with a Slurm deadline at the48h horizon.
CPU scoringarray110517 follows matching GPU tasks using aftercorr;retained
shared110361 is scored separately by110528. Aggregate110529 requires both
scoring paths to complete successfully. Pending old hierarchy110362/report
110363 were cancelled after new CPU admission. Full source/protocol/config
snapshot:`out/architecture-screen-48h-20261023T075200Z/README.md`.
Parent checked generated scripts;read-only review verified the complete
manifest and common-recipe equivalence. Actual new-arm native admission and
10k structure quality remain outstanding;submission is not goal completion.

## Production108918 continuation-value evaluation (2026-10-23)

User requested loss review and actual inference versus the preserved10k model;
production108918 remains unchanged. New frozen evaluation:
`out/lift-depth-60k-108918-20261023T050600Z/README.md`.
The numbered60k EMA was immediately hardlinked before production retention;
CPU admission110410 completed0:0, validates finite EMA, exact60k/r4/K2/LiFT
configuration and unchanged frozen production/panel/evaluation sources. Seven
paired-report checks passed, baseline conditions match and Ruff/Python/Bash
syntax checks passed. Same12 chains/eight PDB entries,seed0,SDE500/tau.01,
length-bin64,full warmups and rotating K1/2/3; original production code only.
Inference110411 is pending test/A5000 resources, then CPU-only actual
OpenStructure scoring/report110412 runs afterok. At14:08 KST the scheduler
estimated inference start2026-10-24 07:06 KST; this is not a reservation.
Inference walltime was reduced from2h to1h for backfill: historical same-panel
sampling took~40min plus warmup, with CPU scoring separate. Do not duplicate
or claim scores before all36 predictions/scores and checkpoint/PDB hashes pass.

Through64k,320 logged windows have update_applied=1. The5–10k versus60–64k
training total-loss median improves.30046->.28397, endpoint FM median
.03419->.02838 and soft-lDDT loss mean.27702->.26453. Total-loss mean instead
.30036->.30733 because occasional FM spikes inflate the recent window;
e.g.60.8k total.65351/FM.53268. No spike cause is established. Maximum
window-averaged grad norm1.4935 is not an instantaneous per-step bound.
These are changing training examples/noise, not matched validation or actual
generated-structure lDDT. Baseline10k K2 actual all-atom lDDT=.72808 and
backbone=.87467;60k quality and change in K2-minus-K1 remain pending.
Continuation to300k is not justified by loss alone; retain main pending actual
structure evidence rather than asserting saturation or stopping it unilaterally.
Reports:training_review.md,training_trend.json,submission.json in the new run.

## Direct ground-truth3+6/hierarchy10k comparison (2026-10-23)

The user replaced teacher adaptation with two teacher-free10000-update trials,
retaining production108918. New frozen protocol/source/config:
`out/direct-gt-3p6-10k-20261023T041000Z/README.md`. Both candidates initialize
from the original unadapted10k EMA with the same explicit3+6 donor map. No
teacher-adapted weights, teacher model or loss override is used. Both retain
r4,K2,all-backbone learning,Muon3e-3 refiner/3e-4 backbone,AdamW fallback,
batch16,seed20261023,warmup500 and no cooldown. The hierarchy alone adds the
initially zero-output fine2/coarse2/fine2 branch.

Source reading and LiFT v1 Section3.1 confirm that u0 is the student's shared
decoder on its own3-block prelude BEFORE either core pass. Existing model/loss
math already implements that boundary correctly. Preserve mean-over-K path
regression to(1-s)*stopgrad(student_u0)+s*aligned_GT_velocity, plus0.01 GT
prelude regression outside the K average, plus final soft-lDDT. The target
anchor is detached; u0's auxiliary and later-state gradients train the student.
Moving u0 after pass1 at totalK2 would remove interior trajectory supervision;
no such change or extra core execution is introduced. Method reference:
https://arxiv.org/html/2610.05538v1#S3.SS1 .

There is no initial lDDT/u0-quality gate: direct GT training must get its full
authorized10k even with a weak initial prelude. Admission checks only config,
finite native loss/gradients,zero skips,exact delivered inputs and actual u0
routing/target detachment. The new probe validates that u0 equals a real core
bypass and that its isolated direct gradient equals only the0.01 GT auxiliary.
All three readouts must receive finite nonzero gradients. Initial/5k/10k probes
replay16 microbatches/four protein examples/64 noise copies, with exact input
and EMA identities. Native8-step weights are discarded before fresh10k runs.

Stored2k/5k EMA checkpoints receive K2 SDE500 structure evaluation after that
candidate completes10k; the10k EMA receives K0/1/2. CPU-only scoring compares
actual all-atom/backbone lDDT,paired entry effects,u0 trajectory diagnostics,
training throughput and peak memory. Frozen original10k baseline is contextual,
not a matched continued-training control. The12-chain development panel has
training homologs; do not make generalization or final-convergence claims.

Local affected checks:113 passed,one CUDA-environment skip, plus Ruff/Python/
Bash and generated Python admission syntax. The two native student-gradient
integration checks cover both shared and hierarchical3+6. Previous immutable
teacher/source runs remain preserved; the superseded teacher hierarchy pipeline
was replaced after CPU admission110359 completed0:0 (105 passed,one CUDA
skip). Teacher hierarchy110311's durable1000-step checkpoint is pinned under
the old run's preserved/ directory. Its requested USR1 checkpoint was not
observed within a bounded wait, so the saved1000-step state was preserved
before cancelling only that superseded job. Pending teacher gate110312,
continuation110313 andreport110314 were also cancelled. Production108918
was never signalled or changed. Native direct admission110360,matched10k
jobs110361/110362 andCPU report110363 are submitted with afterok dependencies;
no initial quality gate is present. Receipts:submission.json,replacement.json.
Native110360 completed0:0. Both candidates applied8/8 actual GT updates with
zero skips,changed all four model groups,andpassed16 real input u0-bypass and
target-only gradient checks each. Training/probe input hashes match. Shared
direct10k110361 is RUNNING(startup);hierarchy110362 waits for the same H100.
The initial-quality guard is absent; both full10k budgets remain authorized.
Native short-warmup checks do not prove structural quality or peak-LR stability.

## Teacher-adapted3+6 failed recovery;hierarchy continues (2026-10-23)

Adaptation110308 completed0:0 with5000 applied updates,zero skips and all
student groups changed; frozen teacher weights remained unchanged. Actual
SDE500/OpenStructure evaluation completed24/24 structures on the same12-chain
panel. All-atom lDDT K0/K2=0.0000/0.1995 versus teacher.73142/.73000;
backbone lDDT=.27967/.79683 versus teacher.87525/.87458. The K2 result
improved from the direct transplant's0 all-atom/.06958 backbone, but remains
far below the declared all-atom admission threshold. K0 is still0 on all12
chains; this is a scored result, not the disabled training soft-lDDT field.

Paired u0 ground-truth FM MSE improved.48679 ->.10213, still3.01x teacher
.03393, above the2x guard. This probe replays one protein with16 noise copies
over four microbatches and uses per-model final-based alignment; actual K0/K2
structure scores independently confirm failed recovery. Gate110309 failed1:0
intentionally, anddependent10k110310 was cancelled before starting. No guard
was relaxed and no replacement compute was launched. Finite velocity training
does not establish correct all-atom generation;5k distillation was insufficient
here, without proving the architecture intrinsically untrainable.

At12:49 KST hierarchy110311 remains running, latest logged600/5000 with
600 applied updates, KD loss.18334,grad norm2.100. Its independent gate110312
andconditional10k110313 remain pending. Production108918 is unchanged and
running at latest logged61000,loss.28186,grad norm.29752,update fraction1.
Evidence: `out/teacher-adapt-3p6-20261023T000200Z/shared_recovery_review.json`.
Prediction/reference hashes were independently verified for all24 structures,
coverage/finite score checks passed andall12 K2 raw score statuses are SUCCESS.
The development panel/one-seed limits remain; no source/runtime changes were
made for this status review.

## Teacher adaptation before3+6 admission (2026-10-23)

The user authorized frozen-teacher adaptation of3+6 and3+6 with hierarchical
latent, then an additional10k LiFT trial for each recovered candidate.
Production108918 stays unchanged. New immutable source/config/protocol:
`out/teacher-adapt-3p6-20261023T000200Z/README.md`.

Teacher is the preserved production10k EMA, evaluated frozen/no_grad on the
same corrupted inputs as the student. A branch input hook plus one extra atom
decoder read obtains the actual teacher u0 and final velocity in one trunk
pass. The bounded5k adaptation objective is0.5*MSE(student_u0,teacher_u0)
plus0.5*mean_s MSE(student_us,(1-s)*teacher_u0+s*teacher_final). Targets are
detached and already share the input frame; neither a broken student anchor
nor independent target alignment is used. Ground-truth FM and soft-lDDT are
disabled during adaptation only. All student backbone parameters train, with
the established Muon3e-4 backbone/3e-3 refiner split and AdamW fallback.
Teacher identity/unchanged weights/no gradients, actual delivered inputs,
exact applied updates/skips and memory/time are audited.

After5k, each candidate independently requires actual SDE500 K0 AND K2 mean
all-atom lDDT within0.05 of the frozen teacher, and paired four-microbatch u0
FM error at most2x teacher. These are predeclared exploratory damage guards,
not statistical noninferiority margins. Teacher baseline is the previously
verified same-H10012-chain panel:K0=.73142,K2=.73000,u0 FM MSE=.03393.
The u0 FM probe retains per-model final-based alignment; structural K0/K2
scores guard against treating a favorable probe frame as structure recovery.
No equality between separately adapted hierarchy/shared outputs is required.
Failed quality admission cancels only that candidate's dependent10k trial.

Admitted EMA weights initialize an identical-layout student BEFORE optimizer
and training EMA construction; the original transplant is never repeated.
The additional10k phase restores ordinary LiFT without teacher, fresh
optimizer/EMA/data and a distinct phase seed. Both candidates share seeds,
batch16, K2/r4, all-backbone training, LR schedule and loss. Actual structure
evaluation covers2k/5k K2 and10k K0/1/2, with loader/update/gradient/throughput
and peak-memory audits. CPU scoring runs separately in cpu_only; GPU stages
use one H100 without changing the production8-GPU allocation. The two
candidates receive5k extra teacher updates; comparison to the frozen teacher
is not a matched continued-training control or a generalization result.

Local affected tests77/77 passed; an additional same-layout continuation
regression passed in the eight-test teacher suite after freezing (the three
runner source hashes still match the snapshot). Ruff, Python/Bash syntax and embedded
admission JSON round-trip passed. CPU admission110306 completed0:0 with77
frozen checks; native teacher sanity110307 completed0:0. Both candidates
applied8/8 updates with zero skips on identical delivered inputs; teachers
remained unchanged with no gradients. Shared/hierarchy peak allocated memory
on these sampled batches was11.89/12.20GiB, and short-warmup step time was
2.04/2.25s (not peak-LR or long-run stability measurements). The8-update sanity
weights are discarded; no5k recovery or10k quality result exists yet.
Job identities and dependencies are in the new run's submission.json.
Queued independent pipelines:3+6 adaptation110308 ->CPU recovery gate110309
->conditional10k110310;hierarchy adaptation110311 ->CPU gate110312
->conditional10k110313. CPU report110314 runs after both continuation jobs
finish/cancel. All GPU phases use one H100 and stay separate from108918.
The3+6 adaptation110308 is running; hierarchy110311 waits for the same H100.
Earlier unsubmitted preparation directory ending000500Z was superseded after
fixing a generated JSON newline; it launched no jobs. Prior failed110172
screen and production sources/checkpoints remain untouched.

## Initial3+6 admission failed;10k never started (2026-10-22)

Native GPU admission110172 completed0:0. CPU quality gate110173 completed
FAILED1:0 by the predeclared0.05 mean-lDDT damage guard; dependent10k job110174
andreport110175 auto-cancelled. Production108918 remains running unchanged;
latest observed step27600,loss0.28950,grad norm0.24853,applied fraction1.

All72 initial structures scored successfully. On the same H100/12-chain panel,
control K0/K2 all-atom lDDT=.73142/.73000;both3+6 andhierarchy=0/0,with K2
backbone lDDT=.06958 versuscontrol.87458. Paired update-free u0 FM MSE rose
from.03393 to.48679 (14.35x),final endpoint MSE from.03411 to.23643 (6.93x).
Native8-step sanity passed8updates/zero skips in all arms but does not establish
structure quality. Thirty copied norm vectors match source exactly; new coarse
restoration is zero, and both new arms' initial noisy-input diagnostics match.
The direct conversion changes the learned function before training; which
sharing/placement/deletion component dominates remains unisolated. Do not
attribute this initial failure to short training or later LiFT optimization.

Review and independently verified source/score/input identities:
`out/architecture-screen-3p6-10k-20261022T121102Z/initial_admission_review.md`
and`.json`. Small shared/hierarchy cross-process SDE coordinate differences
also make the queued strict PDB-bitwise identity gate a future review item;
that gate did not cause the current quality rejection. No frozen source was
changed,no replacement compute was launched,andno10k result exists.

## Replace9+3 screen with3+6; include initial LiFT u0 diagnosis (2026-10-22)

The user explicitly requested control,3+6,3+6 plus hierarchical latent,
each with10k additional updates from the preserved production10k EMA.
Production108918 remains untouched. Prior9+3 GPU/dependent jobs110162-110165
were all pending and were cancelled; completed CPU admission110161 and the
old immutable run remain preserved. New run:
`out/architecture-screen-3p6-10k-20261022T121102Z/README.md`.

Config sets shared_trunk_layers=6,n_trunk=15,K2, yielding independent prelude3.
Donors are original blocks3..8, copied intact and reused twice; blocks9..14
and the old narrow residue stack are excluded explicitly. Original norm/FM
conditioning and low backbone LR remain intact. PLM injections occur at
executed positions5/10 (local2 in pass1,local1 in pass2). Hierarchy retains
fine4 and inserts coarse2 with one initially zero restoration projection.

Initial evaluation now includes true K0/u0 and K2 structures for ALL arms,
plus an update-free probe of the same first4 audited training microbatches:
u0/final RMS,delta/relative change/cosine,prelude and endpoint FM errors,
trajectory loss. Probe identities bind input hashes, actual initial model hash
and checkpoint SHA. FM alignment follows actual per-model final-based LiFT
semantics, not a common cross-model target frame. u0's own structure quality
is diagnostic, not a promotion condition; retain the predeclared0.05 K2
initial damage guard. u0 report is saved before that guard can reject training.

All three architecture trials remain fixed K2 with the same short-run training
and evaluation budgets. The user's follow-up about deeper/sampled recurrence
is a subsequent experiment: K1/2/3 at25/50/25% retains expected K2 but raises
maximum memory. In3+6 this executes9/15/21 full-width layers. Do not conflate
that experiment with the current architecture comparison or raise K by default.
CPU jobs must use cpu_only; native GPU stages request one H100 in heavy,
with all three training arms sharing one allocation. Existing8-GPU production
is not replaced to obtain these resources.
Replacement chain submitted:CPU admission110171 completed0:0 with16 frozen
checks;GPU admission110172 started on an H100;CPU quality gate110173,
three-arm10k comparison110174 andCPU report110175 wait on afterok dependencies.
Local affected checks69/69 passed, plus Ruff/Python/Bash syntax. Actual initial
u0/lDDT and10k comparison results remain pending; native admission has started,
not the full10k comparison. Source/checkpoint/panel hashes verified on the GPU
node at startup. Submission receipts are in the new run directory.

## Bounded10k architecture screening implemented (2026-10-22)

The user authorized short architecture trials, including the fine/coarse/fine
hierarchy, while retaining production108918. Run preparation and protocol:
`out/architecture-screen-10k-20261022T110000Z/README.md`. Three arms start from
the same pinned production10k EMA: unchanged control;9 independent full-width
blocks plus shared3x2 and the existing fine latent4; that shared design plus
two initially zero-output coarse L/16 blocks between fine blocks2/3.
Each arm has a hard additional10k-update budget; future variants are not queued.

New opt-in model settings are `shared_trunk_layers` (default0) and
`latent_hierarchy` (defaultfalse). With sharing, `n_trunk` is nominal executed
full-width depth at configured K; prelude=n_trunk-K*shared_trunk_layers.
The full-width core keeps original block normalization/FM conditioning; depth
conditioning stays on the latent correction. Copied core weights retain low
backbone LR despite being under the latent_residual namespace. PLM reinjection
follows original executed positions5/10. Hierarchical pooling equally weights
valid fine tokens; it is not weighted by original residue mass in partial groups.

The runner initializes from source EMA before normal trainer EMA/optimizer
construction and records donor maps/exclusions, source/runtime identities,
actual initial weights, delivered-batch hashes and exact applied-update budgets.
GPU8-step/initial structure checks precede CPU-only scoring/admission; initial
mean lDDT loss over0.05 stops the dependent10k job. This is a damage guard,
not an established noninferiority margin. GPU training uses one allocation for
three sequential10k arms and2k/5k/10k inference, followed by CPU-only scoring.
K0 truly skips the core; actual sampler calls and structure hashes are audited.

The screen uses matched single-GPU batch16,500-step warmup and no cooldown,
not production's8-GPU batch128. All backbone weights remain trainable; LiFT,
r4,K2 and the validated Muon LR split remain fixed. The small development panel
and one seed support early trend selection, not final/generalization claims.
Control/shared comparison changes several architectural aspects including u0;
shared/hierarchy starts from the same function to isolate the coarse addition.
Submission/state receipts live in the run directory; no change to production.
Submitted chain:CPU admission110161 completed0:0 with10 frozen tests passing;
GPU admission110162 is waiting for resources;CPU lDDT gate110163,three-arm
10k job110164 andCPU report110165 depend on the preceding stage succeeding.
Local verification covered204 passing CPU checks (one test-name error repaired)
and two GPU-environment skips, plus syntax/Ruff/Bash checks. Native GPU training
and structure-quality results remain pending. Do not report the screen as started
or completed merely because its jobs were accepted.

## Architecture debate: shared trunk before second latent level (2026-10-22)

Three read-only design perspectives completed independent openings and one
common rebuttal round. The analysis, summarized positions and arithmetic are
under `out/architecture-hierarchy-debate-20261022T102604Z/`. Production108918
is retained unchanged; no new candidate or GPU experiment was launched.

The preferred FIRST warm-start prototype is now9 independent1024-wide prelude
blocks plus a shared3-block1024-wide residue core repeated K2, retaining the
single r4 latent path and replacing the extra three512-wide residue blocks.
This revises the earlier3+6 preference on migration-risk grounds, not measured
performance.3+6 remains a stronger, higher-risk test of prelude dominance.
Neither should be promoted before true K0/path/profiling diagnostics and an
initial structure-quality audit; K1 already includes a correction pass.

A second latent level L->ceil(L/4)->ceil(L/16) is a later isolated proposal.
For quality expansion, keep four fine blocks and add two coarse blocks through
one initially zero-output branch with fine/full-residue/atom skips. For an
efficiency test, redistribute the four blocks as fine1/coarse2/fine1 instead.
The former adds capacity; the latter changes the learned function. They answer
different questions. Use only one outer K2, without nested per-level recurrence
or extra velocity heads. Mamba already has a global scan; hierarchy is a
representation/allocation hypothesis, not newly global communication.

Estimated base sizes are327.57M for9+3 single-level,343.02M with additive
hierarchy,329.67M with redistribution, before new core depth conditioning.
These are not instantiated models or measured speed/memory/quality results.
Conversion must audit donor mapping, learned norms/residual semantics, original
low-LR parameter roles, PLM reinjection and the changed LiFT u0 boundary.
Evaluate absolute lDDT versus measured time with uncertainty, not K2-K1 alone.

## Shared-trunk architecture analysis;108918 unchanged (2026-10-22)

The user asked whether the full15-block prelude leaves little useful work for
the appended recurrent refiner. Analysis and provenance are under
`out/trunk-sharing-analysis-20261022T080200Z/analysis.md` and `analysis.json`.
Production108918 remains running with its frozen architecture/configuration.
No candidate model was built/trained or additional GPU evaluation launched.

CPU metadata inventory110062 completed0:0:358,968,960 of419,382,490 parameters
(85.59%) are in the original trunk;48,901,824 (11.66%) are in the appended core.
The total excludes42 fixed Fourier buffer values; an earlier assertion failure
was corrected by identifying those buffers in source. Capacity concentration
does not establish where accuracy comes from. Current K1/2/3 results do not
include a complete core bypass K0, so a strong prelude and a useful first
correction followed by little second-pass gain remain competing explanations.

The preferred next candidate is3 original-width prelude blocks followed by a
shared6-block1024-wide residue core plus the existing4-block512-wide/r4 latent
branch, repeated K2, and the existing atom decoder. Remove the separate three
512-wide residue correction blocks. Full-width executed depth stays15, unique
full-width blocks fall to9, and base parameter arithmetic gives255.78M before
new depth conditioning. This is not a measured accuracy/speed/memory result.
Do not interpret the39% parameter reduction as a39% FLOP or peak-memory saving.

Conversion must preserve learned norms, residual semantics and original-block
LR roles. Current prefix-based optimizer grouping would assign a10x higher LR
to copied trunk blocks if moved under `latent_residual.` without changing the
grouping. PLM reinjection positions5/10 and the decoder's earlier prelude input
also need explicit handling. Existing output-hook bypass still computes the
core, so it cannot provide a true K0 timing. The minimal next validation is
same-EMA K0 versus K1/K2; an architecture comparison should use shared versus
untied cloned core groups with the same initial K2 function, matched data and
compute accounting, and fully trainable backbone. Stronger K2-minus-K1 alone
is insufficient if K1 was simply weakened. LiFT/RRT/Huginn are supporting
image/language evidence, not validation of this protein proposal; scoped source
coverage and remaining unknowns are recorded in the report.

## Matched production108918 step10000 evaluation completed (2026-10-22)

Production108918 continues unchanged on eight GPUs. At16:02KST its latest
logged step is12,000, loss0.29697 and preclip gradient norm0.20839; all logged
scalar metrics are finite and all recorded applied-update fractions are1.
No training/model/optimizer/configuration or allocation changes were made.

The new10k checkpoint was saved at15:05KST and hard-linked under
`out/lift-depth-10k-108918-20261022T052200Z/input/`; SHA256 begins7e3bc7ae62baf0fd.
CPU admission109981 completed0:0 at15:06, test/A5000 inference109982 at15:59
and CPU-only OpenStructure2.9.1 scoring/report109984 at16:01:41. All three jobs
exited0:0; no GPU was held while awaiting a checkpoint or scoring on CPU.
Seven pairing/cluster-weighting/depth-gain tests passed in0.057s. Run-local
Python syntax, Ruff and Bash syntax passed before submission.

Every shared protocol field matches5k:12 chains/eight PDB entries, EMA, seed0,
SDE500/tau0.01, references/embeddings, length binning, rotated K order, warmup
and timing scope. All36 new predictions scored successfully with standard
stereochemistry checks. Source/protocol/baseline and prediction/reference
hashes, sampler/EMA manifests and actual recurrence counts K/K/1 were checked.
Report means and paired gain-change arithmetic were independently recomputed;
receipts are in `verification.json`. Existing5k artifacts remain unchanged.

| K |5k all-atom lDDT|10k all-atom lDDT|10k backbone lDDT|Mean s/structure|Peak allocated GiB|
| --- | ---: | ---: | ---: | ---: | ---: |
|1|0.71458|0.72683|0.87417|52.40|2.432|
|2|0.71667|0.72808|0.87467|65.89|2.433|
|3|0.71658|0.72758|0.87483|79.29|2.432|

Overall quality improves at every K:10k-minus5k chain-mean lDDT increases are
0.01225/0.01142/0.01100. Second-iteration chain-mean benefit K2-minus-K1 is
0.00125 at10k versus0.002083 at5k, a change of-0.000833. The change in the
equal-PDB-entry mean benefit is-0.00050, paired entry-bootstrap95% interval
[-0.005375,0.004500]; increasing benefit is not established, and these data do
not establish a true decrease either. At10k eight chains improve with K2,
one worsens and three tie at scorer precision. The one decline,8vyf_ch1,
is-0.017, so more improving chains do not imply a larger mean benefit.
K2's own equal-entry gain interval[-0.000333,0.005125] also includes zero.
K2 costs25.6% more sampler time than K1 with effectively unchanged allocation;
K3 costs20.4% more than K2 and lowers chain-mean all-atom lDDT by0.00050.
Production training stays K2; these measurements do not support increasing K.

`comparison.json`/`comparison.md` contain10k depth comparisons;
`progress.json`/`progress.md` compare5k/10k quality and depth-gain changes.
Paired observational checkpoints cannot isolate the cause of training progress.
All targets have known training homologs, with one sampling seed and eight
correlated PDB entries; no generalization or eventual convergence claim follows.
Changing K also changes the uniform depth grid; K3 exceeds training depth.
Timing is within one A5000 process per checkpoint; cross-checkpoint timings
may vary with device/load. Memory is PyTorch peak allocated, not physical usage.

## Production108918 step5000 depth evaluation completed (2026-10-22)

Production108918 started on October22 at10:31KST and remains running on eight
RTX6000Ada GPUs. At14:02KST its latest logged step is7,400, with finite scalar
metrics and applied-update fraction1 in all37 logged intervals. Its source,
configuration, optimizer and allocation are unchanged. Separate evaluation is under
`out/lift-depth-5k-108918-20261022T034900Z/` with a frozen protocol and source hashes.
CPU-only109840 passed five report-pairing/cluster-weighting tests, hard-linked this
run's exact `trainer/ckpt_0005000.pt`, checked step/configuration and finite EMA,
and completed at13:00KST. Its SHA256 beginscb15a3c803f07ca0; the full hash is in
`admission.json`. Test/A5000 inference109842 completed at13:47 and CPU-only
OpenStructure2.9.1 scoring/report109844 at13:49; all three jobs exited0:0.
The initialization checkpoint was not substituted. No GPU was used for checkpoint
waiting or scoring.

The comparison uses12 fixed development chains from eight PDB entries, EMA,
sequence-only inputs, matched target-derived seed0 and SDE500/tau0.01 at K1/2/3.
All36 predictions scored successfully with standard stereochemistry checks.
Whole-SDE warmup at each depth and rotated K order preceded timed measurements
on one GPU/process. Each forward's residue/latent/decoder call counts were
verified as K/K/1.

| K | Coverage | all-atom lDDT | backbone lDDT | Mean s/structure | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
|1|12/12|0.71458|0.86667|53.00|2.432|
|2|12/12|0.71667|0.86700|66.52|2.433|
|3|12/12|0.71658|0.86683|80.02|2.432|

Primary K2 minus K1 improves chain-mean all-atom lDDT by0.00208:6 chains improve,
1 worsens and5 tie at scorer precision. Equal-PDB-entry mean delta is0.00304,
with a paired entry-bootstrap95% interval[-0.00025,0.007375], so reliable
second-iteration benefit is not established. One chain,8iec_ch1, supplies0.016
of the0.025 summed chain gain. Median paired runtime increases25.6%; allocated
memory is effectively unchanged. K3 minus K2 averages-0.000083 while requiring
20.3% more time. These results do not support increasing inference depth now;
training remains fixed K2 as requested. Reports are `comparison.json` and
`comparison.md`; independently checked coverage, hashes and mean calculations
are recorded in `verification.json`.

All targets have known training homologs; one seed on this development panel
cannot establish generalization. Changing K also changes the uniform inference
depth grid, and K3 exceeds training depth. Memory is PyTorch peak allocation,
not total physical device usage; inference timing excludes model/input loading,
PDB output and scoring, but includes final coordinate transfer to CPU.
Run-local Python syntax, Ruff and Bash syntax checks pass; five CPU tests ran
in0.612s. Training/convergence tests are not repeated because this evaluation
does not modify training or model code.

## LiFT LR controls completed; eight-GPU restart submitted (2026-10-19)

All three 108742 arms completed 5,000 applied updates, 40,000 finite-loss
microbatches and zero skips. Initial model hashes and complete input streams
match; all original/refiner modules remained trainable. CPU-only 108743 scored
12/12 identical development chains per arm with OpenStructure2.9.1 standard
stereochemistry, EMA, K2/SDE500/tau0.01/seed0 on the same A5000 GPU family.

| Setting | all-atom lDDT | Delta from initial | backbone lDDT |
| --- | ---: | ---: | ---: |
| Initial20k-derived identity refiner |0.68017|0|0.83617|
| Original-LR LiFT |0.56167|-0.11850|0.81150|
| Backbone-LR0.1 LiFT |0.69958|+0.01942|0.86125|
| Original-LR final-only |0.56983|-0.11033|0.81617|

Reducing both optimizer rates outside the new refiner improves all-atom lDDT
by0.13792 against matched original-LR LiFT, and beats initialization on10/12
chains. Removing LiFT alone recovers only0.00817 and still regresses on12/12.
This isolates an excessive original-module LR as the dominant repairable cause
in this restart protocol. It does not prove Muon alone caused it: the experiment
scales both Muon and AdamW. Low-LR final-only remains untested; no LiFT-specific
advantage, useful extra inference depth or clean generalization is established.
This is one seed and a correlated development panel with training homologs.
Evidence: `out/refinement-regression-diagnosis-20261019T022804Z/controls_report.json`.
LiFT/low-LR LiFT steady wall times are4.409/4.411s per update; final-only4.068s,
with peak allocated14.681/14.681/13.049GiB at copies2/accumulation8 on A5000.

The user authorized an eight-GPU production restart. Production108331 was kept
running during preparation, then saved and retired with explicit user approval
as recorded below. `configs/residue_latent_refiner.yaml` now selects the admitted
backbone LR scale0.1, fixed K2, compression4, no atom recurrence and LiFT.
Prelude preserves the full original atom encoder and15-block trunk; the shared
width512 correction has three full-resolution residue and four latent blocks.
Both original-module and refiner parameters train. The existing stable additive
residual boundary is retained; it differs from LiFT's image-model boundary
`RMSNorm(h)+h0`. No new normalization is imposed on learned trunk features.

Re-reading [LiFT v1§3/Algorithm1](https://arxiv.org/html/2610.05538v1#S3)
and appendices B/C/D confirms the current objective: fixed training K with
per-example sorted uniform interior depth coordinates and endpoint1; FP64
draw/separation before FP32 depth embedding; target-only `stopgrad(u0)`;
uniform1/K trajectory average plus0.01 prelude regression outside that average;
full gradients through every recurrent state and shared decoder. Inference
uses a uniform grid and final-only decoder. K sampling is a different experiment
and stays off. The paper's random-grid ablation supports depth robustness, not
a guaranteed gain at the training depth; neither protein transfer nor this
expensive atom coda was evaluated there. Author implementation access was not
available, so this audit is against the paper, not inspected official code.

Protein adaptation retains one detached final alignment frame for all velocity
targets and endpoint-only structural losses. Training uses three shared atom
decoder readouts at K2; inference uses one. No per-depth clean-structure loss,
surrogate decoder, frozen anchor network or new loss weight is introduced.
Detached endpoint MSE and velocity-delta diagnostics for each readout now expose
whether the recurrence changes/improves predictions. Console logs show both
backbone/refiner optimizer rates; rank0 also appends scalar `metrics.jsonl`,
including the applied-update fraction. These diagnostics do not alter gradients.

The new run is `out/lift-residue-r4-k2-lowbb-8gpu-20261019T094354Z/`.
It starts from the complete evaluated low-LR5k EMA, including the learned refiner,
using strict weights-only loading and fresh optimizer/scheduler/EMA/data stream.
This is an explicit1→8-rank restart, not exact continuation. Backbone peaks are
Muon3e-4/AdamW1e-5; refiner peaks3e-3/1e-4, with no world-size LR multiplication.
One protein/rank, copies4×accumulation4 retains8 distinct proteins and128 noise
copies per global update. Budget is300k additional updates,5k warmup,50k cosine
cooldown, clip2, EMA0.999 and crop1024. Retain early/mid/final milestones plus
the last three checkpoints. W&B stays offline; frozen source/runtime/config
and the initial checkpoint are bound by hashes. No existing run directory is reused.

Submission is gated by CPU-only affected tests, launch signal/failure checks and
finite strict-layout initial EMA admission. Inside the eight-GPU allocation,
NCCL, two discarded crop1024 full-loss/Muon/EMA updates per rank and32 normal
mixed-corpus trainer updates per rank must all pass before production starts.
The preflight validates all16 binned lengths, applied updates and role LRs;
it is not a new convergence or sustained-throughput result. Pending jobs and
future GPU admission must not be reported as executed training. Source/test
receipts and submission state are recorded in that run directory.

CPU-only108916 completed0:0 in2m14s:219 tests passed,3 CUDA-only cases skipped;
launch failure/signal checks, source/runtime/config admission and finite full-EMA
layout/hash checks passed. The frozen package differs from the quality-validated
source only in detached diagnostics/logging; the trainer additionally records
scalar metrics and applied-update fraction. Model architecture and objective
computations are unchanged. Ruff, Bash syntax and diff-whitespace checks pass.
Initial5k checkpoint SHA256 begins91f8917d8e158988; its complete hash and EMA
tensor hash are in `initial_admission.json`.

Eight-GPU job108918 was submitted with an after-success dependency on108916,
now satisfied. At2026-10-19 18:52KST it is pending `QOSMaxGRESPerUser` while
production108331 used eight GPUs. The user subsequently approved saving and
replacing108331. At19:42KST it saved a full checkpoint at135,753 steps and was
cancelled; a user hold prevented its checkpoint handler from restarting it.
The135k fallback and new135753 checkpoint are hard-linked under the old run's
`archive/replaced-by-108918/`. CPU-only108972 completed0:0, verifying the saved
step, finite complete model/EMA, optimizer/scheduler and8-rank RNG/data state.
The saved file SHA256 beginsbde1e38a7a012f20; complete provenance and hash are
in the new run's `replacement.json` and `retired_checkpoint_verification.json`.
The expected batch termination signal is15; Slurm's job summary is CANCELLED.
No existing checkpoint was removed.108918 remains pending; its native GPU
admission and training have not run. Other queued users' jobs occupied the
released node, so replacing the old job does not guarantee immediate allocation.

## Backbone regression isolated; LR/loss controls running (2026-10-19)

Production108331 remains untouched. Read-only2x2 component transplants isolate
where the completed fixed5k regression is carried. Here "backbone" includes
all original modules outside `latent_residual`: atom encoder/decoder, residue
trunk, conditioning, embeddings and coordinate head. "Initial" means the
shared20k EMA-derived model before additional training, with its identity
refiner; it does not mean random initialization or current production weights.

All four combinations were regenerated on the same A5000 GPU family, covering
12/12 chains each, K2/SDE500/tau0.01/seed0, sequence-only inputs, EMA and
OpenStructure2.9.1 with standard stereochemistry checks. The initial score is
therefore remeasured rather than borrowed from the earlier Blackwell run.

| Backbone weights | Refiner weights | all-atom lDDT | backbone lDDT |
| --- | --- | ---: | ---: |
| Initial | Initial |0.68017|0.83617|
| Initial | Fixed5k |0.68008|0.84175|
| Fixed5k | Initial |0.58433|0.81100|
| Fixed5k | Fixed5k |0.58550|0.81633|

Restoring the backbone while keeping the learned refiner recovers+0.09458
all-atom lDDT by chain and+0.09021 with equal PDB-entry weighting. Removing the
refiner from the trained model does not restore accuracy. Backbone-only changes
worsen all12 chains against initialization. The learned refiner on the initial
backbone has essentially zero mean all-atom gain (-0.00008); these results do
not yet show a useful all-atom improvement from recurrence. Cross-swaps can
break coadaptation, and this one-seed development panel has training homologs.
The result localizes the regression to original-module updates, not its exact
optimizer/objective cause or a single layer within the backbone.

Raw geometry supports the same attribution: initial/full5k/restored-backbone
bad-bond totals are213/518/210, with clashes1,344/1,900/1,410. Read-only matched
input gradients on four predetermined length strata at t0.5/depths0.5/1 also
show all four endpoint FM and soft-lDDT losses worsen after training. Mean FM
MSE rises0.02185→0.02435 and soft-lDDT loss0.31656→0.32931. EMA relative L2
parameter changes are26.8% in the residue trunk and25.9% in the atom decoder.
LiFT versus final-only total trunk-gradient cosines are0.997–0.999 on these
inputs, not evidence of a detached or reversed backbone gradient.

The source audit found no demonstrated LiFT detach/averaging bug: detached
trajectory target anchor, mean over recurrent readouts and0.01 prelude loss
match [LiFT v1§3.1/Algorithm1](https://arxiv.org/html/2610.05538v1#S3.SS1).
The aligned FM target also matches the pinned
[SimpleFold reference](https://github.com/apple-aiml-research/ml-simplefold/blob/c7a5570a6be9f5c695126e27c804e77567209934/src/simplefold/model/simplefold.py#L368).
At identity, K2 LiFT's shared-backbone/decoder FM gradient
relative to final-only FM is `(s+1)/2 + .01`, expected0.76, while endpoint
structure loss retains full weight. Scalar loss ratio is instead
`(s*s+1)/2 + .01`, expected0.67667. A shared-parameter test checks this
intentional weighting difference. The source objective and trained
normalization are preserved while separating its effects from learning rate.

`backbone_lr_scale` is one new optimizer control, default1 for unchanged legacy
groups/checkpoints. Scaling below1 reduces both Muon and AdamW rates everywhere
outside the new refiner without freezing. The candidate0.1 uses backbone
Muon3e-4/AdamW1e-5 and refiner Muon3e-3/AdamW1e-4, with the same scheduler
multiplier. Split-group layouts/scales are checked before restoration; metrics
expose actual rates for each role. No reduced rate or new checkpoint has been
promoted to the reusable recipe or production before quality validation.

The isolated workbench is `out/refinement-regression-diagnosis-20261019T022804Z/`.
GPU108729 and108737 generated48 structures; CPU-only108745/108738 completed
standard scoring with complete coverage, successful raw statuses and unchanged
geometry settings. `causal_report.json`, `causal_analysis.json`,
`gradient_summary.json` and all raw scores preserve the evidence. Frozen source
manifests and the original panel manifest were rechecked.

Gradient job108731 was intentionally cancelled after its float32 flattened
reductions produced impossible cosines above1. A diagnostic-only fix uses
blocked float64 inner products;108740 reran all eight inputs successfully.
Model precision and loss code were not changed. A new regression test covers
the blocked reduction. Affected checks total64 passing cases across optimizer,
DDP-wrapper, loss and attribution suites (108739/108734/108741); one CUDA-only
loss test was skipped on CPU. Earlier exact-continuation fixture failures
(maximum2.9e-11) are reproduced by sharing optimizer state tensors between the
two paths. The corrected serialization-isolated fixture passes20 exact
continuations in108749; both intentionally aliased controls reproduce that
failure. No tolerance was relaxed. Ruff, Python syntax, Bash and whitespace
checks pass.

Native GPU preflight108736 applies8/8 updates per control at scale1 and0.1,
zero skips, identical complete initial model and all64 microbatch inputs;
every original/refiner module group changes. Both warm all16 shapes through
crop1024. Full comparison108742 is now running three5,000-update arms:
original-LR LiFT, backbone-LR0.1 LiFT, and original-LR final-only as a causal
loss control. All419,382,490 parameters remain trainable; K2/r4, no atom
recurrence, full mixed498,546-chain data, seed20261018, EMA0.999,1k warmup and
flat rates afterward are common. On24GB hardware, two copies/eight accumulated
microbatches preserve16 noise copies/update. This differs from the older
8-copy/two-microbatch stream, so a new matched original-LR control is included.
Complete initialization matches the older run; complete model initialization,
CPU input hashes and applied-update counts match across all three arms through
step200 (`early_controls_admission.json`). First50 LiFT means are loss0.29972/0.29995,
gradient norm0.374/0.369; this is warmup health, not full-rate quality evidence.

Dependent CPU-only108743 scores the final5k EMAs on the same12-chain panel and
writes `controls_report.json` only after admitting complete initialization,
input streams, applied updates, configuration differences and score coverage.
Passive one-second wall observers capture1000→5000 loading/GPU/checkpoint IO;
concurrent node load remains a timing limitation. The report separates the
LR-only and loss-only effects and records whether reduced-LR LiFT beats its
matched control without regressing against the initial panel mean. Do not
claim the lower LR repairs accuracy before that result. Optimizer restart and
distinct-protein batch remain unseparated: these single-GPU runs use one
protein/update, while production uses eight. Production's latest103.4k log point
is loss0.2854, gradient norm0.51, Muon3e-3, with zero gradient-guard/OOM-NaN
skips or Python tracebacks in a full-log scan. Its7-day time limit was verified.

## Full-backbone depth comparison completed (2026-10-19)

The 400-update, frozen-backbone pilots below establish connectivity and repair
behavior, not whether fixed or sampled recurrence learns better. At the user's
request, production108331 remains untouched while a separate paired comparison
trains every parameter, including the existing atom encoder/decoder and complete
residue trunk. Atom recurrence stays disabled in both arms. This changes the
experimental training scope, not the running production recipe.

`benchmarks/compare_full_refinement.py` instruments the normal `scripts/train.py`
entrypoint without replacing its loader, objective, optimizer, EMA or checkpoint
implementation. Both arms start from the same20k checkpoint EMA backbone plus
the same fresh identity refiner, use LiFT/r4/Muon3e-3/AdamW1e-4, crop1024,
eight copies and two accumulated microbatches on the allocated96GB GPU. This
retains16 noise copies/update; the paired preflight uses four copies/four
microbatches. Before either full run starts, the larger microbatch was admitted
using the earlier full-gradient optimizer/EMA K3 capacity result:43.67GiB
allocated/44.68GiB reserved at crop1024 with atom recurrence enabled. Both longer
arms disable that adapter. They train5,000 updates each with
1,000 warmup steps and a flat rate afterward. Fixed K2 is compared with stateless
K1/2/3 sampling at25/50/25%. The full admitted RCSB/AFDB mixed corpus replaces
the128-chain pilot subset. One same GPU runs the arms sequentially; this is not
a new DDP scaling measurement or a production replacement.

A dedicated identically seeded loader generator and capture/restore of prewarm
RNG separate worker inputs from model depth draws. Every delivered CPU tensor
field, including ESM, masks, crop positions, coordinates, noise and FM time, is
hashed. Initial complete model/trainable sets and the complete input stream must
match across arms. An8-update paired preflight precedes the longer runs.
Nonfinite loss/gradient-guard failure stops this experiment; it does not silently
count skipped updates as learning. Persistent worker RNG is not checkpointed by
the normal trainer, so this paired protocol requires uninterrupted runs rather
than claiming corruption-exact resume. Normal full checkpoints remain available.

The predeclared endpoint is the final5k EMA;2.5k EMA is a learning-curve check,
not an invitation to select a favorable checkpoint. The existing frozen panel
expands evaluation from4 to all12 chains across8 PDB entries. Initial/2.5k
checkpoints use inference K2; both final EMAs use K1/2/3, SDE500/tau0.01/seed0,
with the standard OpenStructure stereochemical filters. Expected coverage is
108 structures. All targets have known training homologs; one training seed and
correlated development targets cannot establish generalization or convergence.
Compare actual standard lDDT rather than equating differently averaged LiFT
training losses. Input/initialization checks gate the paired score report.

Artifacts and frozen source/config/protocol:
`out/full-refinement-k2-vs-sampled-20261018T124151Z/`.
CPU audit/variable-depth tests108612 passed26 tests in37.11s;108616 passed
16 audit tests in1.87s, and108618 passed17 audit tests in2.02s. Added coverage
includes six paired-report admission cases and a complete synthetic score report
that distinguishes correlated chain means from equal-PDB-entry means. There are
33 distinct affected tests across the variable-depth and expanded audit suites.
Both GPU preflights completed8/8 applied updates, zero skips and exactly
matching32-microbatch input hashes and complete initial model hashes. All four
parameter groups changed, including the original atom encoder/decoder and trunk.
Preflight sampled depths were seven K2 and one K3; this tiny draw is admission,
not a depth-distribution or quality estimate. Multiprocessing shutdown printed
EBUSY errors while removing `.nfs` temporary files; both stages returned normally.
CPU job108617 reloaded the5.04GB fixed checkpoint and verified every model/EMA
tensor finite, step8, scheduler/data cursor and optimizer/runtime receipts.
Those cleanup tracebacks are preserved and distinguished from training failure;
ownership of the remaining NFS handles was not established or changed.

GPU108613 completed exit0:0 in3h54m09s, followed by CPU-only scoring108614,
exit0:0 in4m43s, finishing2026-10-19 01:43:20 KST. Both arms applied5,000/5,000
updates with10,000 finite-loss microbatches and zero skips. All419,382,490
parameters were trainable; all four monitored groups changed. Complete initial
model hashes and complete input streams match. Actual sampled counts are
K1/K2/K3=1,274/2,499/1,227 (mean1.9906). Frozen source and all panel hashes were
rechecked after completion. The model and normal training entrypoint were unchanged.

All108 structures and nine12/12 score sets are present, without scoring failures.
Frozen inference explicitly loads EMA and uses sequence-only inputs. Every raw
score preserves OpenStructure2.9.1 standard stereochemistry checks; all individual
all-atom lDDT values are positive. The initial model is the shared pre-finetuning
identity refiner on the20k EMA backbone, not the current production checkpoint.

| Stage, inference K2 | all-atom lDDT | backbone lDDT |
| --- | ---: | ---: |
| Initial, before additional training |0.67558|0.83600|
| Fixed K2,2.5k EMA |0.61192|0.82033|
| Sampled K,2.5k EMA |0.59908|0.81908|
| Fixed K2,5k EMA |0.58542|0.81617|
| Sampled K,5k EMA |0.56908|0.80533|

The primary sampled-minus-fixed K2 difference is-0.01633 by chain and-0.01325
with equal PDB-entry weighting; sampling improves2/12 chains and worsens10/12.
Both arms worsen all12 chains against initialization, at both2.5k and5k. Thus
the result does not support promoting either new5k checkpoint. Final all-atom
lDDT at inference K1/K2/K3 is0.58567/0.58542/0.58092 for fixed training and
0.56817/0.56908/0.56875 for sampled training; extra inference loops show no
consistent quality gain on this panel.

The complete1000→5000 wall interval averages891.55ms/update fixed and893.16ms
sampled (+0.18%, effectively equal in this one run). Peak allocated memory is
36.896/42.051GiB (+13.97% sampled); peak reserved42.717/46.240GiB. The wall
observer includes completed GPU optimizer/EMA work, loading, input auditing,
logging and intermediate checkpoint IO; it excludes initialization/final writes,
with one-second boundary observation. End-to-end training times including those
costs are4,668.04/4,674.31s. At SDE500, final K1/K2/K3 mean inference times are
27.12/34.09/41.30s fixed and27.25/34.19/41.30s sampled, with about2.434GiB peak
allocated memory. These are same-GPU measurements, not DDP scaling claims.

The prepared wall-report helper was not present in the executed scoring output;
it was applied afterward to the completed observer data. The supplemented
`comparison.json` and `comparison.md` contain that timing scope and full window.
The source/panel hashes, complete coverage, EMA readout and geometry settings
were verified before supplementing the report; no score or checkpoint was changed.

Observed geometry deterioration accompanies the loss of lDDT: total clashes
across the12 K2 predictions increase1,350→1,897/2,055 (initial→fixed/sampled5k),
and bad bonds205→520/488. This does not isolate the cause of the learned changes.
There is no control separating backbone learning rate/optimizer restart, smaller
distinct-protein batch, LiFT objective, or architecture. Numerical finiteness and
zero skipped updates do not establish accuracy preservation. Do not infer from
this comparison that longer training alone will repair the regression.

Recommendation: keep production108331 unchanged, do not add sampled K or promote
either experimental5k checkpoint, and diagnose the common fine-tuning regression
before another promotion. Production is still running at the99.2k log point,
Muon3e-3, latest loss0.2897/gradient norm0.48; a full11,582-line scan has zero
gradient-guard skips, OOM/NaN skip lines or Python tracebacks. Recent2k-step means
span loss0.2848–0.3119 and gradient norm0.48–0.76. This is a training-health
observation, not a new production lDDT measurement.

Final artifacts include `comparison.md`, `comparison.json`, `paired_audit.json`,
`final_analysis.json`, complete scoring/raw geometry outputs and both5k EMA/full
checkpoints under the run directory above. The single-seed, correlated development
panel has known training homologs; generalization remains unmeasured. Syntax,
affected Ruff, launcher Bash and diff-whitespace checks passed; no new test run
was needed for this result/documentation update.

## Variable-depth and atom recurrence comparison (2026-10-18)

Production 108331 retains its frozen source, configuration and checkpoints.
The separate `configs/residue_latent_refiner.yaml` now enables training-only
`latent_loop_sampling`: a stateless run-seed/optimizer-step draw selects
K=1/2/3 with probabilities 25/50/25%. Every rank and accumulated microbatch
shares that step budget. Every selected LiFT trajectory ends at depth 1;
inference still uses `latent_loops` (default 2). No learned halting or adaptive
outer SDE integration is introduced. Legacy configs default to fixed depth.
The normal entrypoint prewarms K3 network forward/backward and fails early if
its configured network shapes do not fit. It backpropagates all LiFT readouts,
releases their graphs before the next shape, and includes a capped, non-binned
maximum length. This is not a substitute for full loss/optimizer/EMA capacity
measurement; the latter is recorded below.

Optional `atom_recurrence` adds a shared local width-64 adapter after every
residue+latent update. It reads updated residue context and carried atom hidden
states, mixes canonical slots, updates atoms, and returns masked pooled residue
feedback. Its two parallel output projections start at zero; neither is behind
another zero gate. All new parameters live under `latent_residual.*`, so strict
backbone migration and explicit refiner-only diagnostic training remain valid.
Initialization additionally rejects nonzero atom readouts. Atom encoder/trunk,
their normalization and PLM injection remain unchanged; coordinates/FM time
stay fixed inside the hidden-state loop. Atom recurrence defaults off.

`benchmarks/compare_variable_depth.py` predeclares three arms: fixed K2,
sampled K, and sampled K plus atom recurrence. It uses the same EMA backbone,
initial residue/latent weights, 128 admitted training chains, corruption inputs,
400 optimizer updates, Muon3e-3/AdamW1e-4, LiFT, crop256/two copies and frozen
backbone. It verifies initial equality and unchanged backbone weights, records
actual depth work, and evaluates every arm at K1/2/3 with SDE500/seed0 and
standard OpenStructure stereochemistry on the same four development chains.
All three400-update pilots completed without skipped updates. Each passes108
initial exact-output comparisons and three initial LiFT readouts; every learned
backbone tensor stays unchanged. Common initial residue/latent parameters and
all corruption inputs match by hash. Actual pilot overrides are crop256, two
noise copies, one microbatch/update,50 LR-warmup steps and no EMA. This differs
from the source recipe's full-training data/schedule settings.

Coverage is4/4 chains for every row/depth below, from three PDB entries with
known training homologs. Standard OpenStructure2.9.1 stereochemical handling,
SDE500/tau0.01/seed0, one RTX PRO6000 Blackwell Max-Q:

| Training | K1 all-atom lDDT | K2 all-atom lDDT | K3 all-atom lDDT |
| --- | ---: | ---: | ---: |
| fixed K2 |0.66250|0.65375|0.65750|
| sampled K1/2/3 |0.65450|0.65800|0.66125|
| sampled + atom recurrence |0.65850|0.65625|0.65025|

K sampling changes the quality/depth tradeoff but does not improve every depth.
Atom recurrence adds189,828 parameters and is active: removing it changes
velocity RMS0.571%, resetting carried atom history0.397%, and suppressing its
residue feedback0.248% across12 fixed t0.5 cases. These effects do not translate
to a consistent lDDT gain in this short pilot. Its default stays off.

Mean inference seconds at K1/2/3 are27.66/34.87/42.04 (fixed training),
28.06/35.27/42.71 (sampled), and28.28/36.06/43.88 (sampled+atom). Peak allocated
inference memory is about2.434/2.438/2.445GiB per arm, essentially independent
of K because inference keeps only the current paired state and decodes once.
K1 saves about21% time versus K2; no inference-memory saving is established.

Initial short training profiles showed a large execution-order effect and are
superseded by two opposite arm orders,15 warmup +30 measured updates per arm
per order. All six input hashes match. Setup GC roots are frozen after warmup.
These profiles train every parameter and include full loss/backward/clipping,
Muon+AdamW and EMA, at crop256/two copies; loading and DDP are excluded:

| Training | Observed mean K | Mean ms/update | Two round means ms | Peak allocated GiB |
| --- | ---: | ---: | --- | ---: |
| fixed K2 |2.000|348.9|357.3 /340.4|7.441|
| sampled |2.133|351.0|357.1 /344.8|7.773|
| sampled+atom |2.133|397.7|420.2 /375.3|7.880|

Round variation remains, so small training-speed differences are inconclusive.
Atom is8.8–17.6% slower than sampled-only in both orders, with no consistent
quality benefit. No new eight-GPU throughput claim follows from these probes.

One admitted long-chain capacity probe includes the complete trainable model,
Muon/AdamW, EMA and two accumulated microbatches, with atom recurrence enabled.
At crop1024, copies8 uses32.25/37.96/43.67GiB allocated at K1/2/3 (K3 reserved
44.68GiB). Copies4 uses19.78/22.65/25.53GiB (K3 reserved26.11GiB). Driver/NCCL
memory is outside these allocator numbers; this is a single-chain capacity
probe at0.01x peak LR, not a steady-throughput or DDP test.
The separate experimental recipe therefore uses copies4 × accumulation4,
preserving16 noise copies/rank and leaving room on48GB GPUs. The measured
copies4 capacity used accumulation2; four sequential backwards need no saved
four-microbatch graph, but the new eight-GPU configuration remains unmeasured.
Production108331 retains its original copies8 × accumulation2 and frozen code.

Verification:171 CPU tests passed,2 CUDA-dependent checks skipped,55.43s
(108594);34 tests passed in the GPU allocation, including21 real-kernel tests,
105.37s (108578). CPU coverage includes two-rank variable-depth accumulation,
atom history/feedback, initial identity, masking, CLI guards and prewarm graph
lifetime/capped lengths. GPU pilots and full-gradient profiles use real kernels.
Affected Ruff, syntax, config/effective-copy and whitespace checks pass.

Artifacts: `out/variable-depth-atom-20261018/comparison.md` and `comparison.json`,
three frozen code variants/manifests, admissions, migrations, training inputs,
400-step weights, all36 structures and standard scores, atom causal ablations,
reversed-order EMA profiles and capacity rows.108578 completed32m07s;
108586 scored all arms;108604 completed corrected capacity/rechecks in4m30s;
108605 aggregated them. Failed capacity attempt108587 used a1024-residue
example in a256-length warmup buffer; its log/source are preserved and the
warmup alone was corrected. It did not alter any learned/evaluated weights.
The model and sampler sources match the working tree. This remains a short,
correlated development comparison, not a generalization or convergence result.

## Status

- State: production108918 retired with70k preserved; historical warm-start screen complete; all12-model scratch WSD/max10k/no-distillation screen preparing; globalarchitecture unselected
- Root: repository root
- Last updated: 2026-10-23

## Project

- Name: MambaFold-clean
- Type: ml
- Goal: single-chain all-atom protein structure generation by direct all-atom
  flow matching, on a corpus that reproduces SimpleFold's data admission so the
  architecture claim and the benchmark claims rest on one contract.
- Core claim: a Mamba folding trunk without attention or a learned pair stack
  can produce useful sequence-conditioned, all-atom protein structures from
  frozen ESMC-6B embeddings. The upstream ESMC-6B encoder uses attention.
  The current external baseline is not a matched training/control arm.
- Why a separate root: the previous ESMC-6B track's corpus and this one make
  contradictory statements about the same benchmarks. Under the old contract
  CASP14 carries six exact training matches and is development-only; here it
  carries none by date and is reportable. A shared namespace would let a number
  be cited against the wrong contract. **No number produced under the previous
  contract may be reported here.**
- Scope: single-chain proteins, standard amino acids, monomer chains extracted
  from every admitted entry. MSA-free, conditioned on pinned sequence-only
  ESMC-6B embeddings.
- Non-goals: multimer/interface prediction, ligands, nucleic acids, metals,
  cofactors, water, non-standard residues/PTMs.

## Data

Full contract and evidence: `docs/data_contract.md`. Build: `pipeline/README.md`.
Reporting rules: `benchmarks/BENCHMARK_POLICY.md`.

- **Sources.** RCSB from the Boltz `rcsb_processed_targets` 2024-12-20 snapshot
  (216,870 records) and AFDB SwissProt **v4** from the frozen EBI archive,
  restricted to SimpleFold's published `swissprot_list.csv`.
- **Admission**, reproducing all three SimpleFold filters in their order:
  `SizeFilter(min_chains=1, max_chains=300)` from the top-level `filters:`
  block, then `ResolutionFilter(5.0)`, then
  `DateFilter(date="2020-05-01", ref="released")`.
- **Resolution is a no-op** on this archive: every Boltz record carries
  `resolution: 0.0`, so the reference filter admits everything.
  `RESOLUTION_SOURCE=boltz` reproduces that; `rcsb_api` is a stricter deviation.
- **AFDB v4, not v6.** The previous track resolved accessions through the live
  API and received v6 coordinates created 2025-08-01, from a generator whose
  cutoff is not established. v4 is AlphaFold 2 output with a 2018-04-30 cutoff,
  which precedes every benchmark, so the distillation channel is clean by
  construction rather than by screening.
- **AFESM is not used.** This corpus is therefore a strict subset of
  SimpleFold's, and any parity result is conservative.
- **PLM.** ESMC-6B, 2560-d, addressed by SHA-256 of the canonical sequence.
  268,639 of 268,977 AFDB accessions share a sequence between v4 and v6 and
  reuse the cached embedding; the 338 that changed were recomputed.
- Nothing is copied: structures are symlinks into the Boltz snapshot,
  embeddings are hard links. External inputs are declared only in
  `config/paths.env`.

### Current counts

| Bucket | Records |
| --- | ---: |
| **admitted (SimpleFold-exact) — the corpus figure** | **161,613** |
| admitted, protein-chain filter on (what the farm holds) | 158,210 |
| held out, released after cutoff (SimpleFold-exact) | 53,618 |
| held out, protein-chain filter on | 52,608 |
| excluded, no protein chain | 4,413 |
| excluded by the size rule | 1,639 |
| AFDB SwissProt v4 | 269,003 |

The two admitted figures differ only by entries with no protein chain, which
produce no training example either way, so the two corpora train identically.
SimpleFold's filters have no molecule-type rule, so **161,613 is the corpus
figure** — it is what its published "~160K" corresponds to. The protein
requirement is this project's addition; the 158,210 farm is what that rule
leaves on disk, and both numbers are always reported together.

Both are pipeline outputs, not arithmetic: stage 02 writes
`protein_filter_cross_report` into `data/splits/report.json`, carrying
`simplefold_exact_admitted: 161613` alongside `non_protein_pre_cutoff: 3403`
and `non_protein_post_cutoff: 1010`. Re-derived independently from the Boltz
manifest, both figures reproduce exactly.

### Chain-level totals

Every entry-level count above is an *entry*. The training unit is a *chain*,
and `extract_monomer_chains` indexes every protein chain of every admitted
entry as its own monomer example. `data/audit/chain_length_distribution.json`:

`11_verify_training_readiness.py` read every record of both sources and found
**nothing blocking**: no missing embedding, no row-count mismatch, no width
mismatch, no unreadable record. Every one of the 765,241 chains resolves a
2560-d embedding with the right number of rows.

| | RCSB | AFDB v4 | Combined |
| --- | ---: | ---: | ---: |
| records | 158,210 | 269,003 | 427,213 |
| trainable records | 156,949 | 268,841 | **425,790** |
| chains (≥20 aa) | 496,400 | 268,841 | **765,241** |
| residues | 139,310,232 | 85,078,216 | **224,388,448** |
| median / p90 / p99 | 238 / 509 / 1071 | 282 / 554 / 1027 | 248 / 521 / 1057 |
| max | 5,037 | 2,551 | 5,037 |

Records yielding nothing are all short, not broken: 1,250 RCSB and 162 AFDB
fall under the 20-residue floor, and 11 RCSB entries carry no protein chain.

RCSB's 496,400 is confirmed twice, by `04_build_fasta.py`'s per-chain TSV and
independently by the readiness stage. The length quantiles come from
`data/audit/chain_length_distribution.json`, whose AFDB side is derived from
`manifest.tsv` and so counts one chain (77 residues) that the readiness stage,
reading the file itself, canonicalises to under the floor. The readiness figures
are the authoritative ones.

A stricter population still — chains that also admit a valid observed-atom crop
— is what the loader actually indexes;
`pipeline/13_prebuild_loader_caches.py` measures it and writes
`data/audit/loader_caches.json`.

## Architecture

- Pure Mamba trunk. **There is no attention and no pair stack anywhere in the
  model, and neither is reachable by config.** `bimamba3.py`'s hybrid attention
  layer, `model/fold/pair_blocks.py` and `model/fold/multiplicative_update.py`
  were deleted, along with `use_pair_stack`, `trunk_attn_layers`,
  `trunk_attn_every`, `n_attn_heads`, `d_pair`, `n_pair_blocks`, `n_pair_heads`,
  `pair_mult_c` and `pair_use_cueq`.

  Deleted rather than switched off, for two reasons. A trunk that *could* be
  given attention by flipping a flag cannot support the claim that a pure SSM
  trunk suffices — the claim has to be a property of the code, not of a default.
  And a disabled O(L²) triangle path still has to be carried, sized, and
  reasoned about at every crop decision. That reasoning stands on its own — it
  is why attention is absent rather than disabled — and does not depend on the
  control arm, which was cancelled (see below).
- Atom → residue → atom: Run A and the latent recipe use four local atom-slot
  MLP-Mixer blocks on each side, plus a cross-residue BiMamba on five backbone
  streams in both encoder and decoder. The residue trunk is pair-free BiMamba.
  `atom_mixer: mamba` is an alternative local mixer, not the current recipe.
- There is no learned O(L²) pair tensor, including in the output path.
- Experimental post-trunk residual: `latent_compression: 4` or `8` compresses
  consecutive residue groups with offset-specific linear projections, mixes the ordered latents
  with time-conditioned BiMamba, then restores residue-specific updates using
  the original residue features and within-group positions. It runs after
  the complete residue trunk, directly before the atom decoder. The readout is
  zero-initialized; the full-resolution trunk and atom skip remain intact.
  `d_latent: 256` and `n_latent_layers: 2` set its width/depth; SSM settings
  are inherited from the trunk. `latent_compression: 0` (default) creates no
  branch parameters. Archived Run A uses its original source/runtime; the
  current working tree uses the pinned heavy-tail Mamba-3 implementation below.
  `latent_loops: 2` is the default; positive integers repeatedly
  apply the same latent stack with shared weights and full backpropagation.
  Compression runs once, and FM time and coordinates stay fixed.
  Effective latent depth is `n_latent_layers * latent_loops`; increasing loops
  adds no parameter keys. Train with the chosen loop count; extra inference
  loops on an existing checkpoint have unvalidated behavior. This adds
  computation; folding accuracy is unmeasured. Short GPU training-cost
  measurements for ratio 4 / K=2 are recorded below.
  Default `latent_loop_mode: conditioned` adapts LiFT's architectural boundary:
  `z_k = Mamba(RMSNorm(z_{k-1}) + z_0, temb + depth_embed(k/K))`.
  The boundary norm has no parameters; the depth embedder is shared across
  loops. Both final-only and LiFT training use per-example sorted random
  interior depths with endpoint 1; inference uses uniform depths k/K. The
  original FM time conditions the shared decoder at every readout.
  The active recipe defaults to `latent_supervision: lift`; the generic parser
  retains `final` for configs without a latent branch. `final` uses final-only FM.
  `lift` reads initial and all recurrent states with the same residue restoration
  and atom decoder. It averages masked velocity MSE against
  `(1-s_k)*stopgrad(u_0) + s_k*fm_target`, plus a fixed 0.01 prelude FM term.
  One detached alignment from the final clean prediction supplies all targets;
  full rollout gradients remain. lDDT and enabled geometry losses run only at
  the final output. Encoder/trunk features and the local restoration projection
  are cached. K=2 LiFT training decodes three times; inference and no-grad
  self-conditioning always decode once. No residue suffix repeats. No new
  parameter keys arise from changing supervision. GPU cost is measured below
  for two fixed crops on one device; accuracy remains unmeasured. No
  inference-depth extrapolation claim is made.
  Explicit `latent_position: midpoint` and `latent_loop_mode: plain` retain
  legacy final-only experiments. Old experimental checkpoint configs must pin
  their original position/mode/loops. Conditioned mode adds depth-embedding keys.
  Initial compression recommendation is 4, with 8 as a comparison, not a
  measured optimum. Re-reading both loader caches reproduces 498,546 chains
  and 146,311,147 cropped residues (median length 254): mean valid latent
  counts are 73.740 at ratio 4 and 37.120 at ratio 8 before padding. At fixed
  width/depth, ratio 8 halves latent sequence work but leaves trunk/readout
  work unchanged; more loops cannot reverse compression loss. See README for
  the length table, kernel-padding caveat and controlled comparisons.
  `latent_position: post` is now the default and supports a one-block trunk.
  The existing final-only continuation helper (`scripts/slurm_latent_post_compare.sh`:
  Run A step 300,000 continued with and without the branch, ratio 4,
  conditioned K=2, FM + lDDT only, PROVISIONAL budget) retains its original
  baseline-versus-latent comparison. LiFT is selected separately with the
  trainer's `--latent_supervision lift`; no jobs are submitted automatically.
- Baseline loss: rigid-aligned flow matching + exact all-atom lDDT. Pretraining uses
  α(t)=1; a separately declared fine-tune may use 1+8·ReLU(t−0.5).
  Confidence is trained later against a frozen folding model.

### Finalized experimental recipe and optimizer (2026-10-15)

`configs/latent_mamba.yaml` pins Run A data/dimensions/schedule plus post-trunk
ratio 4, latent width 256, two shared blocks, conditioned K=2, LiFT supervision and
`optimizer: muon`. Ratio 8 and final-only remain single-key alternatives. This is a
new experimental recipe; the completed Run A keeps its original AdamW config.

The current revision uses position-specific compression:
`z = sum(mask_j * W_j * h_j) / valid_count + bias`. The offset matrices start
equal with the original d_res fan-in scale, then learn independently. One
packed `Linear(r*d_res, d_latent)` remains eligible for Muon. Masking occurs
before projection, bias is added once, and empty groups are zeroed. At ratio 4
this adds 786,432 parameters over the earlier mean projection (376,182,874 total).
There is no additional compression-method switch. Ratio 8 also changes the
projection shape; switch ratios as separate runs, not as checkpoint resumes.

Mamba-3 is pinned to upstream `33e2849ff046f72d60db8b17dbf47b7c949ff8c1`
(2.3.2.post1) in both pyproject and lockfile. This includes the official
[heavy-tail A update](https://github.com/state-spaces/mamba/pull/962) and
[MIMO backward update](https://github.com/state-spaces/mamba/pull/967).
`is_outproj_norm` remains false, so no fused-norm speedup is claimed.
Runtime import rejects the older softplus-only module; new checkpoints and
probe reports record installed version, source revision, module hash and decay.
The three dense MIMO kernel factories reuse TIR symbols by name/dtype. The
pinned upstream creates fresh B/S/H/G objects per call, which miss TileLang
0.1.8's first JIT cache and rebuild TIR before finding the compiled kernel in
its second cache. The local adapter preserves static arguments and delegates
kernel compilation/launch to TileLang; it never caches activations or gradients.
Runtime metadata records `stable_kernel_symbols: true`.
The project-local environment replaces the previously borrowed interpreter.
Archived checkpoints retain their original implementation outside this work;
no migration of old mean-compression weights or optimizer states is attempted.
Atom geometry refinement and larger SSM/loop settings are not added to this
bounded change. Accuracy and convergence effects require training evaluation.

| Stage | Execution per network evaluation | Decision |
| --- | --- | --- |
| Coordinate/sequence/PLM and FM-time embeddings | once | retain split spatial scales and frozen cached PLM |
| Atom encoder | four local MLP blocks + one backbone-stream BiMamba stack | retain masked gated/mean pooling and atom skip |
| Residue trunk | 15 independent BiMamba blocks | retain full-resolution path and PLM reinjection |
| Latent | compress once, two shared blocks repeated twice | retain input injection, depth conditioning and full rollout gradients |
| Residue restoration and atom decoder | once for final; three readouts for K=2 LiFT training | preserve atom-specific context, one shared decoder and zero-initialized velocity head |
| Loss | final structural losses; optional recurrent velocity targets | same detached final alignment frame, no extra pair or confidence branch |

All trainable hidden `nn.Linear.weight` matrices use native Muon, including
packed Mamba input projections, output projections, SwiGLU, slot mixing,
conditioning and latent restoration. Embeddings (including aliases), the final
coordinate head, norms/biases and non-matrix SSM/MIMO parameters use AdamW.
The partition follows module roles, deduplicates shared parameters and excludes
frozen weights. No layer-wise optimizer settings are exposed.

Muon fixes momentum 0.95, Nesterov and NS iterations 5. The active recipe uses
`original` scaling with a 3e-3 Muon peak LR and a 1e-4 AdamW peak LR. The user
authorized reducing LR and replacing the unstable 1e-2 run. Both groups follow the existing schedule multiplier and
0.0 weight decay. Older configs retain RMS-matched scaling and a shared LR when
the new keys are omitted. LR remains provisional, and clipping gradient norm
does not directly bound the subsequently orthogonalized Muon update. There is
one optimizer update after accumulated/DDP-reduced gradients and one EMA update.
Full resume validates hybrid group names/shapes and restores both states;
switching algorithms requires an explicit optimizer reset. Native Muon is
required only when selected; the inspected environment is PyTorch 2.11.

Architecture inspection did not establish a need for additional repeated atom
or residue blocks. Both would increase full-resolution work beyond the measured
latent experiment. The code descriptions now distinguish local atom MLP mixing
from backbone SSM mixing, and distinguish tied parameters from executed scans.
The small comparison order is supervision at ratio 4/K=2/Muon, then compression
only if needed. Protein accuracy and Muon convergence remain
unmeasured; short execution checks cannot select an optimizer by fold quality.
No additional CI workflow or long training run is introduced.

Historical verification before position-specific compression and the dependency
update: the full CPU suite passed **181 tests**, with 16 CUDA-only tests
skipped, in 100.45 s (job 106990, COMPLETED/0). It includes exact continuation
of model updates, scheduler and EMA through a serialized hybrid checkpoint;
optimizer-switch/layout rejection; frozen/shared/embedding parameter partition;
and single-rank DDP graph checks for both objectives and optimizers. Syntax,
affected-file Ruff and diff-whitespace checks passed.

That earlier production-size recipe had 373,884,640 parameters (215 matrices) on Muon and
1,511,802 parameters (529 tensors) on AdamW, totaling 375,396,442. Replacing the
second FP32 moment on the Muon subset saves 1.393 GiB of optimizer state.
Separate real-CUDA execution verification (job 106989, COMPLETED/0, 2 min 8 s)
used the same admitted chain/cached PLM as the previous cost probe, crop 256,
eight copies, accumulation two, bf16, three warmup and five measured updates:

| Objective with Muon | Median ms/update | Peak allocated GiB |
| --- | ---: | ---: |
| final | 543.4 | 11.27 |
| LiFT | 613.0 | 13.00 |

Both runs retained finite parameters and identical source/input fingerprints.
Reports: `artifacts/muon_architecture_verify_20261015T052400Z/summary.json`
and the two full reports alongside it. Previous fused AdamW crop-256 medians
were 419.6/460.6 ms and peaks 12.66/14.39 GiB. This is a comparison across short
sessions with five versus ten measured updates, not a controlled convergence
result. Muon reduces state memory but adds per-update time; total time to a
target fold quality remains unknown. Loading, DDP and cold compilation are
outside the GPU timing. No long training checkpoint was produced.

### Training throughput audit (2026-10-15)

The active ratio-4/K=2/Muon recipe now batches EMA updates by device/dtype,
projects AdaLN conditioning once for both branches of each atom/residue/latent
block, and transfers all scalar loss metrics together. EMA retains the original
multiply-then-add rounding and parameter-only state; the loss API still returns
Python floats by default. Parameter keys, optimizer settings, FM/LiFT targets,
atom coverage, gradient accumulation, clipping and EMA frequency are unchanged.

Exact lDDT retains the same chunked cutoff discovery and all unordered pairs.
Its four thresholds are evaluated together. A small autograd function recomputes
bounded pair intermediates and applies the norm/abs/sigmoid derivative directly,
removing a checkpoint/autograd graph per pair chunk. It retains no full predicted
pair matrix. Dense-reference value/gradient tests, float32/float64 degenerate
cases, custom-threshold gradcheck and CUDA parity cover this change. Reference
coordinates/distances remain detached, as before. Floating-point reduction order
can differ; this is mathematical objective equivalence, not bitwise equivalence
to an older training trajectory.

`ProteinBatch.pin_memory()` implements PyTorch's
[custom batch protocol](https://docs.pytorch.org/docs/2.11/data.html#memory-pinning).
Previously, setting `DataLoader(pin_memory=True)` did not pin this dataclass.
Expanded views are materialized before pinning, and DDP length padding preserves
pinned storage with one destination allocation. Optional fields and the
single-row PLM representation are preserved. In a separate 24-batch warm-cache
input probe, pinned batches changed from 0/24 to 24/24 and median transfer wall
time from 4.20 to 0.65 ms. Loader-only wait medians were 114.7/124.8 ms with two
workers, so no disk-loading speedup is claimed. This repeated one admitted chain;
it does not measure cold/random corpus I/O or overlap with training.

Profiling the production update (two microbatches, eight copies each) found:

| Operation count per update | Crop 256 before → after | Crop 1024 before → after |
| --- | ---: | ---: |
| CUDA runtime kernel launches | 26,885 → 23,231 | 44,041 → 34,629 |
| Linear calls | 976 → 918 | 976 → 918 |
| Scalar `item` reads | 45 → 5 | 45 → 5 |
| Sigmoid calls | 512 → 128 | 2,048 → 512 |

These are profiler diagnostics, not profiled wall-time speed claims. Independent
warm-update measurements use synchronized wall time without the profiler.
Artifacts are under `artifacts/training_speed_20261015T061139Z/`, including an
immutable source snapshot for the before arm and input/source fingerprints.

The first three-warmup/ten-update comparison showed timing drift, including a
short-crop LiFT reversal. A reverse-order check used **10 warmup and 20 measured
updates per arm** on RTX PRO 6000 Blackwell Max-Q, bf16, ratio 4/K=2, eight copies
and accumulation two. Its synchronized median update times are:

| Supervision | Crop | Before ms | After ms | Time reduction | Peak allocated GiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| final | 256 | 450.1 | 397.6 | 11.7% | 11.291 → 11.294 |
| final | 1024 | 1157.8 | 991.1 | 14.4% | 26.123 → 26.122 |
| LiFT | 256 | 498.4 | 448.5 | 10.0% | 13.022 → 13.031 |
| LiFT | 1024 | 1261.7 | 1130.8 | 10.4% | 32.662 → 32.662 |

This includes forward, production loss, backward, clipping, Muon/AdamW and EMA.
It excludes loading/transfer, DDP, checkpoint I/O and cold compilation. Final
arms shared allocation 107021. The optimized LiFT arm completed in 107031; a
concurrent config-comment edit interrupted the ledger's source capture before
the baseline launched. Only that missing baseline was completed in 107041 on
the same GPU model. All eight cases retained finite parameters; paired arms
match config/input/runtime and parameter layout, and their source hashes were
stable. `summary.json` and `*_recheck.json` retain exact values and scope. The
earlier raw samples remain available; these short probes do not establish
statistical confidence or end-to-end multi-GPU throughput.

Final verification: the complete CPU suite passed **203 tests**, with 20 CUDA
tests skipped (job 107017, 30.53 s). A CUDA allocation passed **117 affected
tests** (job 107018, 31.10 s), including all 17 real-Mamba masking/backward/cache
checks, hybrid optimizer/scheduler/EMA checkpoint continuation, loss parity,
and actual DataLoader pinning/transfer. The first GPU pass exposed expanded-view
pinning; materializing non-contiguous fields fixed it, and the complete affected
suite then passed. Affected-file Ruff, syntax, diff-whitespace, dependency lock
and installed-package compatibility checks passed.

The pre-throughput pinned-Mamba/position-compression verification also completed:
187 CPU tests passed (17 skipped), 17 CUDA tests passed, and all four r4/r8 ×
final/LiFT production-size probes retained finite parameters. Those earlier
reports are in `artifacts/latent_upgrade_final_20261015T060320Z/`; the throughput
measurements supersede their timings for the current working tree.

The audit also checked existing PLM deduplication/mmap, length bucketing,
worker CPU limits, persistent prefetch, DDP bucket views/`no_sync`, fused AdamW,
and kernel prewarming. These paths already avoid their obvious redundant work.
Native Muon's five NS iterations remain about 76 ms/update on the probed GPU;
reducing iterations or changing precision would change the optimizer experiment.
Full model compilation, CUDA graph capture, spatially sparse reference-neighbor
construction and asynchronous checkpoint writing are not enabled without
separate validation of dynamic lengths, masks, resume state and memory costs.
No training options, dependencies, CI jobs or long training run were added by
this throughput change. Accuracy/convergence and larger world sizes remain
unmeasured; full two-rank pipeline measurements follow.

### Full loader and DDP verification (2026-10-15)

The production `scripts/train.py` entrypoint was measured with its complete
498,546-chain RCSB/AFDB dataset, real NPZ/PLM reads, augmentation, length-bucket
sampler, pinned transfer, gradient accumulation, NCCL DDP, clipping, Muon/AdamW,
scheduler, EMA and stdout logging. The requested `test` A5000 allocation was
unavailable; with the user's approval, the completed measurements used one/two
RTX PRO 6000 Blackwell Max-Q GPUs. The frozen before/after comparison on `test`
was queued as job **107060**. It was cancelled while pending and replaced by
the current-source LiFT-only comparison **107109** at the user's request (see
the longer audit below). Both use two A5000s and copies 4 × accumulation 4 to retain 16 noise
copies per rank within 24 GiB, with a 35-minute job limit.

The completed recipe uses ratio 4/K=2, copies 8 × accumulation 2, max crop 1024,
bf16 and native optimizer settings. DDP got three workers per rank after the
existing CPU-allocation cap; single rank got seven. Each run made 26 updates.
Six warmup updates, the profiled update/following update, and the final shutdown
update are excluded, leaving 17 measured updates. All binned model shapes were
prewarmed for the reported runs. The initial unprewarmed single-rank pilot is
preserved as a cold-start diagnostic and excluded from comparisons.

`benchmarks/probe_training_pipeline.py` instruments the actual entrypoint rather
than duplicating its loop. `--summarize` reports the maximum accumulated elapsed
time across ranks for throughput; it also retains each step's rank maximum.
The last update is excluded because its observation window includes final
rendezvous/process-group teardown. This aggregation has a separately selected
unit test; it prevents shutdown or sums of alternating rank tails from being
misreported as sustained training throughput. Checkpoint writes and W&B are
disabled in this bounded diagnostic. Startup, compilation and the profiling
sample are separate from the following warm measurements.

For a bounded rerun inside a two-GPU allocation, use the current checkout:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmarks/probe_training_pipeline.py --report-dir artifacts/pipeline-rerun -- \
  --config configs/latent_mamba.yaml --out_dir artifacts/pipeline-rerun/trainer \
  --total_steps 26 --warmup_steps 10 --lr_cooldown_steps 0 --log_interval 1
.venv/bin/python benchmarks/probe_training_pipeline.py --summarize artifacts/pipeline-rerun
```

| Run | GPUs | Median ms/update | Mean ms/update | Peak allocated GiB/rank | Median loader wait ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| final, before pipeline changes | 2 | 685.7 | 690.1 | 17.61 | 0.141 |
| final, current bin 64 | 2 | 605.5 | 704.2 | 17.61 | 0.102 |
| LiFT, current bin 64 | 2 | 676.3 | 691.3 | 20.92 | 0.107 |
| final, current bin 64 | 1 | 656.5 | 674.1 | 20.20 | 0.111 |
| final, experimental bin 32 | 2 | 624.8 | 630.0 | 17.12 | 0.109 |

These are real variable-length draws, not fixed-crop benchmarks. The two-rank
before/after and LiFT runs match per-rank sequence hashes/valid lengths and
sampler order; every rank completed all 26 optimizer/EMA updates with stable
source fingerprints. Changing bin width changes the padded hash payload, but
valid lengths and the sampler seed/order are retained. The measured subset
reached padded length 512; the peak above is not a worst-case crop-1024 bound.
Single/two-rank draws have different length distributions, so their numbers do
not establish controlled scaling efficiency.

The median final update fell 11.7%, but the mean did not improve in this short
comparison: one current-run forward/loss interval at step 17 took about 1.74 s
versus 0.25 s in the before arm. The cause was not isolated. The raw event stays
in the mean; no sustained-throughput gain is claimed from removing it. Likewise,
the short-run mean does not establish that LiFT is faster than final-only. The
typical median LiFT overhead here is about 11.7%; longer timing windows are
needed for a stable aggregate throughput comparison.

Implemented pipeline changes:

- Batch-missing and maximum-length agreement use one packed integer collective
  and one readback. The redundant post-loop skip collective is removed because
  missing/nonfinite decisions already agree before backward; distributed OOMs
  still fail the complete run. Profiler counts confirm control reductions
  **10 → 7** and total all-reduces **59 → 56** per update. Gradient buckets,
  full-precision reduction, per-microbatch nonfinite guards and preemption
  agreement remain in place. Packed timing metrics use one readback too.
- Chain-index fingerprints reuse resolved parent directories within each hash
  and avoid expensive Path-parent comparisons. Leaf symlinks are still followed;
  relative/absolute aliases, dangling links, parent targets, retargeting and
  loops are tested against pathlib's original fingerprint. Existing full-corpus
  indexes load unchanged, with no rebuild. In the profiled two-rank startup,
  loader construction fell **303.7 → 110.3 s** and cache-key work dominated both.
  These include cProfile overhead and shared-filesystem cache effects. A separate
  2,003-path ABBA check retained identical cache keys and reduced the warm hash
  interval **0.208 → 0.101 s**; it is not a cold-start speed guarantee.

The remaining measured loader wait is about 0.1 ms, so workers/prefetch were not
increased. Isolated two-GPU collectives moving approximately one full gradient
per repetition took about 60 ms with 25/64/128 MiB buckets and default/Simple
protocols; no useful protocol/bucket override was adopted. Earlier probes that
overlapped model prewarming are retained under `comm_pilot/` and excluded.
Bin 32 reduced measured padding **14.7% → 9.4%** and peak memory by 0.49 GiB,
but its median was slower and observed cold prewarm took **356.8 s** versus
**129.6 s** for bin 64. Its lower mean is confounded by the bin-64 outlier.
**Keep `length_bin: 64`**; no extra architecture/training option is introduced.

Verification: 205 full-suite CPU tests passed (20 CUDA-only skips), followed by
an explicitly selected timing-summary test passing. The new two-rank Gloo
protocol test and compatible-cache-key test are included. Full production
NCCL updates passed on two GPUs for before/after/final/LiFT/bin-32 cases. Jobs
107045, 107054, 107061, 107062 and 107064 completed with exit 0. Ruff, syntax and
diff-whitespace checks passed. Raw reports, CPU profiles, immutable source
snapshots and `comparison.json` live in
`artifacts/ddp_pipeline_20261015T063745Z/`. No long training or checkpoint-quality
evaluation was performed.

### Longer LiFT-only timing and GC stalls (2026-10-15)

The active latent recipe now defaults to **LiFT, ratio 4, K=2 and Muon**.
Final-only remains an explicit control; none of this follow-up's GPU timing
arms uses it. The generic parser keeps final-only for older configs without
a latent branch, avoiding a new requirement on the archived Run A recipe.

Job **107076** completed a frozen-source before/after/after/before comparison
on the same allocated two RTX PRO 6000 Blackwell Max-Q GPUs, using NCCL and
the full 498,546-chain loader. Each arm ran 221 real optimizer/EMA updates,
copies 8 × accumulation 2, crop limit 1024, bin 64, seed 0 and bf16. All 221
rank-local input hashes, padded lengths and valid-residue counts match across
all four arms. Every update applied, and source fingerprints stayed stable.
The first 20 updates and final shutdown update are excluded: **200 measured
updates per arm, 400 per variant**. All remaining stalls stay in the mean.
Loader cProfile and CUDA profiling are disabled; identical lightweight CPU,
CUDA-event, GC, allocator and slow-stack instrumentation runs in every arm.
Batch loading, transfer, DDP, loss, Muon/AdamW, scheduler, clipping, EMA and
per-update stdout are included. Startup/prewarm, checkpoint writes and W&B
are excluded.

| Order | Variant | Mean ms/update | Median ms/update | Peak allocated GiB/rank |
| --- | --- | ---: | ---: | ---: |
| 1 | before pipeline changes | 688.93 | 651.49 | 34.09 |
| 2 | after pipeline changes | 681.94 | 650.75 | 34.07 |
| 3 | after pipeline changes | 681.93 | 650.42 | 34.09 |
| 4 | before pipeline changes | 698.55 | 668.48 | 34.09 |

The combined mean is **693.74 → 681.94 ms/update**, a **1.70% time reduction**
or 1.73% more updates/s. This supports a small improvement in this experiment;
the earlier short-run 11.7% median difference is not a sustained-throughput
estimate. A paired circular-block bootstrap (blocks of 20 updates, 3,000
resamples) gives a conditional 95% reduction interval of **0.82–2.93%**. This
describes timing variation within these two matching repeats, not uncertainty
across independent seeds, nodes or hardware. The uninstrumented production
loop and checkpoint/W&B overhead are not separately benchmarked here.

The forward-stall investigation reproduced two natural steady-state pauses
in the first before arm:

- Rank 0, update 213: generation-2 GC **737.2 ms**, forward/loss **958.6 ms**,
  including **948.8 ms of host thread CPU**. Rank 1 spends **744.9 ms** in
  control collectives while its forward/loss takes 223.9 ms.
- Rank 1, update 216: generation-2 GC **736.7 ms**, forward/loss **948.3 ms**;
  rank 0 waits **745.6 ms** in controls. Both ranks have zero allocator retries,
  device allocations, allocator synchronizations and major faults at both
  updates. Python samples locate normal Triton dispatch and the GC callback.

These pauses are full Python heap scans delaying GPU launch dispatch and then
propagating to DDP peers. CUDA event intervals include the resulting idle gap.
This causally identifies the two new events. The old final-only update 17 has
no CPU/GC trace, so assigning that exact historical event the same cause remains
an inference. The separate profiler reproduction was cancelled while pending
once natural reproduction isolated this class of stall.

`freeze_training_setup()` collects startup garbage, then temporarily freezes
the existing loader/model/setup objects during the production loop. The first
batch is prepared before entering this context so the lazy sampler's persistent
epoch lists are included. `itertools.chain` replays that batch through the
ordinary microbatch counter, transfer and DDP protocol; no batch is dropped or
duplicated. Completed-step resumes do not prime an unused iterator. Normal
cyclic collection remains enabled for new batches and autograd graphs. The
permanent generation is unfrozen on normal or exceptional exit. This is owned
by the standalone training process, including its interpreter startup roots;
it is not a general embedding-context API. No training
switch, parameter, target, optimizer or reduction is added. The loop's AST is
identical inside the new lifetime context. Isolated CPU tests assert an actual
increase in frozen objects and exercise new autograd tensor-cycle cleanup,
preserved gradients, enabled cyclic collection and normal/exception thawing.

The first diagnostic, job **107083**, completed but did not apply the correction:
a guard skipped freezing because even `python -S` starts with 375 permanent
objects on this Python build. Its calibration recorded zero newly frozen objects,
so it is excluded as a fix validation. The guard was removed for the standalone
entrypoint and calibration now fails if the frozen set does not grow.
Job **107098** then exercised actual freezing, but did so before initializing
the lazy sampler: its 200-update mean was 691.56 ms and maximum per-update GC
82.8 ms. Source inspection identifies 249,273 persistent batch lists created
by the sampler's first iteration. This partial boundary is not the final
implementation. Job **107108** validates priming and then freezing on the
same 221-update LiFT workload; all 221 rank-local hashes, lengths, valid/padded
counts and update flags match the original reference. The measured source and
entrypoint exactly match the current checkout.

| Final correction | Measured updates | Mean ms/update | Median ms/update | Maximum GC ms/update | Peak GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| iterator primed, setup frozen | 200 | **672.61** | 641.16 | **3.13** | 34.09 |

The final mean is 3.05% below the original 400-update before mean and 1.37%
below the 400-update pipeline-after mean. This final correction has **one**
200-update timing window, not another ABBA repeat; its incremental speedup
is not established across independent runs/nodes. The largest final update is
1,153.3 ms, with real lengths reaching 1024. No allocator retry/OOM or major
fault is recorded in the measured window. Median loader wait is 0.130 ms;
more workers/prefetch were not added. The maximum GC above is total GC time
within one update, with normal collection for new objects still enabled.

The final startup-only counterfactual first collects twice, then measures a
full collection after freezing the same persistent roots: clean scans take
**688.88/695.15 ms**, versus **0.0259/0.0169 ms** with
**2,522,945/2,522,967** objects frozen. These are individual GC-call timings,
not per-update savings. Calibration is outside the measured training window
and asserts actual growth of the permanent generation. Epoch turnover,
checkpoint/W&B overhead and convergence remain outside this bounded check.

Verification: **144 affected CPU tests passed**, with 20 CUDA-only skips,
including isolated real-freeze/autograd-cycle/exception tests and the
profile-disabled summary regression. Jobs 107076 and 107108 completed with
exit 0; every final GPU rank applied all 221 updates. Ruff, syntax and diff
checks passed. The requested two-A5000 test comparison is now **107109**,
LiFT-only before/after, copies 4 × accumulation 4, 81 updates per arm and a
35-minute cap. It is pending for resources; the current Slurm estimate is
**2026-10-16 23:42 KST**, subject to queue changes. It has not run successfully
yet, and none of the timing claims above is an A5000 measurement.

Raw reports, frozen source/scripts, `comparison.json`, `final_validation.json`, GC/slow-stack
traces, telemetry, CPU test XML and `diagnosis.md` are stored in
`artifacts/lift_long_20261015T074242Z/`.

### Shared-memory limits (measured)

The Mamba-3 MIMO backward kernel's dynamic shared memory, at headdim=64,
d_model=1024, bf16, fits exactly:

```
smem = 27,008 + 736 × d_state + 250 × chunk_size   [bytes]
```

Four measurements, zero residual, at the stated residue width only. Do not
extrapolate this fit to the width-128 atom streams or width-256 latents.
`chunk_size = base // mimo_rank`, with a kernel minimum of 8. Base is 64 only
for capability >=9.0 with at least 160 KiB opt-in shared memory; otherwise 32.
The workstation sm_120 device has 101,376 B and therefore uses base 32. Rank 1 is
a separate case: it returns 64 on every device and takes the SISO kernel, whose
budget is not this formula (the one measurement is `d_state=256` → 114,944 B).

- RTX 6000 Ada, limit 101,376 B: `d_state=64` works at rank 1/2/4; `d_state=128`
  works **only at rank 1**, which takes the lighter SISO kernel; `rank=8` is
  impossible because chunk falls to 4.
- `mimo_rank` matters through the kernel path, not the budget: rank 2→4 halves
  chunk and saves 2,000 B, while `d_state=256` on SISO (114,944 B) is *lighter*
  than `d_state=128` on MIMO (123,216 B).

Applying the formula to the rest of the fleet:

| GPU | optin limit | `d_state=64` | `d_state=128` | `d_state=256` |
| --- | ---: | --- | --- | --- |
| RTX 6000 Ada, sm_89 | 101,376 | r2/r4 ok | **rank 1 only** | none |
| H100, sm_90 | 232,448 | r2/r4/r8 ok | r2/r4/r8 ok | r2 ok, 223,424 B |
| RTX PRO 6000, sm_120 | 101,376 † | r2 atom / r4 latent and residue verified; rank 8 unsupported | unmeasured | unmeasured |

† observed on-device in the 2026-10-15 current-width probe. The previous
generation-only chunk heuristic selected 32 for the width-128, rank-2 atom
block: forward ran, but backward requested 150,080 B and failed. With chunk 16,
the same block's forward/backward passed. The default now checks the current
device's shared-memory budget; model parameters and state/rank settings are
unchanged. Rank 8 would select chunk 4, below the kernel minimum.

Audit correction (2026-10-15): the stored H100 and RTX PRO 6000 probe reports
each contain 12 `temb`-argument failures before kernel execution. They do not
validate the hardware predictions above. The probe now supplies the required
time condition. Those old reports remain invalid; the current-width evidence
is in `artifacts/latent_training_probe_20261015T044301Z/chunk_diagnosis.json`.
The Ada report has
actual forward/backward results at d_model=1024, not at the new latent width.

### Short latent training-cost comparison (2026-10-15)

`benchmarks/probe_latent_training.py`, launched manually by
`scripts/slurm_probe_latent_training.sh`, measures the production FM + exact
all-atom lDDT loss, full backward, gradient clipping, **fused AdamW** and EMA.
On one RTX PRO 6000 Blackwell Max-Q, both objectives use the same admitted
chain, cached PLM, seeded initial weights, width-256/two-block post-trunk
conditioned latent, compression 4 and K=2 (375,396,442 parameters). An update
accumulates two microbatches of eight independent noise copies. Three warmup
updates precede ten measured updates per case; times are medians.

| Crop | final ms/update | LiFT ms/update | Time increase | Peak allocated GiB, final → LiFT |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 419.6 | 460.6 | 9.8% | 12.66 → 14.39 |
| 1024 | 978.1 | 1091.0 | 11.5% | 27.48 → 34.02 |

Raw timings, reserved-memory peaks, source/config checks and the summary are
in `artifacts/latent_training_probe_20261015T044301Z/fused_adamw/`; Slurm job
106986 completed with exit 0. Source/config/input-manifest and parameter-count
checks match between arms apart from supervision. Loader/collation/transfers,
DDP and cold compilation are excluded. These are short fixed-chain cost
measurements, not accuracy or end-to-end distributed-throughput results.
Individual timing spikes are retained in the raw files. Earlier failed-kernel
and unfused-optimizer attempts are preserved separately and are excluded from
this table. No CI job was added and no long training run was started.

### Eight-GPU LiFT production run (2026-10-15)

The user authorized full training from the current latent recipe on **eight
GPUs**. The launch uses one node with eight RTX 6000 Ada GPUs, seven-day wall
limit and the existing 300,000-update recipe. It starts from scratch, preserving
the archived Run A outputs. No final-only comparison is launched.

- LiFT, compression 4, conditioned K=2, post-trunk latent, native Muon/AdamW.
- Muon peak LR **1e-2** with `original` scaling; AdamW peak LR **1e-4**.
  Both groups use the same 5,000-step warmup and 50,000-step cooldown
  multiplier, retaining a 100:1 LR ratio (Muon endpoints 1e-4; AdamW 1e-6).
  Clip 2, EMA 0.999;
  crop limit 1024 and length bin 64 remain unchanged.
- One protein/rank, eight noise copies × two accumulated microbatches per
  protein: **8 distinct proteins and 128 noise samples/update globally**.
  LR is not multiplied by world size. Workers respect the CPU allocation cap.
- Checkpoint every 5,000 updates, retain the last three plus atomic
  `ckpt_latest.pt`; optimizer, scheduler, EMA, RNG and data cursor are saved.
- W&B is offline with local stdout/metrics. No external upload is scheduled.
- The job verifies frozen source/config and installed runtime versions before
  an eight-rank NCCL preflight; P2P is disabled as required by prior node probes.
  NCCL failure stops the job without a silent Gloo fallback. Shape prewarming
  covers all 16 binned lengths before training updates.
- A batch-shell USR1 trap forwards to the actual rank PIDs without signaling
  the torchrun manager or unrelated/stale PIDs. A completed preemption
  checkpoint requeues the same frozen run and resumes full state strictly.
  Failed training is not automatically retried; ordinary cancellation does
  not request a requeue.

The frozen run lives at
`out/latent-lift-r4-k2-muon-8gpu-20261015T084737Z/`: `code/`, `run_config.yaml`,
`runtime_expected.json`, `source_manifest.json`, `provenance.json`,
`launch.sh`, and checkpoint/log output under `trainer/`. Source snapshots are
authoritative for this dirty-worktree launch; later checkout edits do not change
the queued model. The existing environment must still match the recorded pinned
versions and Mamba revision/hash at startup.

CPU-only launch validation **107112** passed: frozen runtime/config hashes,
training failure status, USR1 delivery to an owned rank, ignored foreign PID,
and correct completion after an interrupted shell wait. Eight-GPU execution is
verified by the allocated job's preflight and first updates, not by this CPU
control test. Slurm accepted the production launch as **107113** at
**2026-10-15 17:50 KST**. The initial state was **PENDING / Resources** with an
estimated start of **2026-10-17 13:37 KST**, which later changed with the queue.
Eight GPUs were requested together on one eligible node. A read-only
12-hour/short-QoS scheduling probe was rejected by its per-job GPU cap; the
accepted eight-GPU verylong job was not changed.
Log: `logs/slurm/mf-lift-r4-k2-8gpu-107113.out` (created when allocated).
`submission.json` records the job and observation; `launch_verification.json`
records the completed CPU control checks. The pending A5000 diagnostic 107109
is separate and does not block this production job.

The user subsequently requested Muon LR around **1e-2** explicitly. Before any
training began, job 107113 was held while amending its frozen optimizer/config
and verifying independent group LRs, the schedule, and exact checkpoint resume.
The pre-change snapshot remains under `revisions/initial-rms-lr1e-4/` inside the
run directory; `lr_amendment.json` records the amendment. The active recipe uses
`muon_lr: 0.01`, `muon_adjust_lr_fn: original`, and `lr: 0.0001` for AdamW.
Unspecified Muon settings retain the old RMS-matched/shared-LR behavior for
older configs. Both group LRs are logged. This changes optimization, not model
dimensions, LiFT/r4/K2, the 300k budget or the eight-GPU resource request.
CPU-only verification job **107117** completed with exit **0:0**: **123 passed,
2 CUDA tests skipped** in 23.07 s, including the 100:1 group LR warmup/peak/floor,
bitwise checkpoint continuation for model and EMA, legacy optimizer behavior,
config validation, source/runtime checks and launch signal handling. Job 107113
was released after these checks with the same resource request.

Observed on **2026-10-18 11:54–11:56 KST**: job 107113 started at
**2026-10-17 19:39:21 KST**, passed the eight-rank NCCL preflight and is running.
A complete scan of the first 79,776 stdout lines (4,238,359 bytes) found
**79,104 counted steps, 71,571 gradient-guard skips and 7,533 inferred applied
updates**. No OOM/NaN-loss skip or traceback appeared in that scan. The first
gradient skip was step 4056 (warmup Muon LR about 8.13e-3); 10k–20k skipped
99.8%, and the last 1,000 counted steps all skipped. Step 79,000's 200-step
averages were loss 2.1015 and pre-clipping grad norm 2.38e11, with Muon LR 1e-2.
`clip_grad_norm_(..., 2.0)` is active: the returned/logged norm is measured
before clipping. A separate guard skips updates if that norm is >=1e4 or
nonfinite, while still advancing scheduler and step count. Muon's subsequent
normalization/orthogonalization means clipping raw gradients does not directly
cap its weight update. The failure's underlying cause has not been isolated by
a controlled replay; attributing it solely to LR would exceed this evidence.

The 200–4000 logging window averaged about **0.840 s/count**, with no gradient
guard skips. The 78k–79k window averaged **0.712 s/count**, mostly skipping
optimizer/EMA work; this is not successful-update throughput. Live nvidia-smi
showed **37.05–37.07 GiB/device** on all eight GPUs; rank-0 logged PyTorch
reservation was 36.33 GiB. These are observations, not activation-memory peaks.
The user explicitly selected **keep execution running** after being informed
of the skips. No cancellation, LR/guard change or restart was performed.

### Lower-LR production replacement (2026-10-18)

The user then authorized lowering LR, registering a replacement first, and
removing the unstable run. The new configuration uses **Muon 3e-3 (`original`)
and AdamW 1e-4**, with the same LiFT/r4/K2 architecture, 300k budget, crop 1024,
5k warmup, 50k cooldown, clip 2 and gradient-skip guard. The shared schedule
preserves a 30:1 group LR ratio (Muon start/floor 3e-5; AdamW start/floor 1e-6).
3e-3 is a provisional reduction, below the ~8.13e-3 LR at the first observed
skip; its folding convergence is not established by the earlier warmup.

**108331** was registered before stopping 107113, with dependencies on
successful CPU verification **108330** and termination of **107113**. It requests
one node/eight RTX 6000 Ada GPUs with the existing seven-day/requeue policy.
It starts **from scratch**: the retained 65k/70k/75k checkpoints are already
from the unstable period and are not used as initialization. The latest old
checkpoint was hard-linked under the old run's `archive/` for preservation;
old logs/config/source and the other checkpoints remain intact.

The new frozen run is
`out/latent-lift-r4-k2-muon-lr3e3-8gpu-20261018T025829Z/`. Its training/model/
optimizer code is byte-identical to the previous production snapshot. Only
recipe/launch provenance changed; existing `keep_checkpoint_steps` now retains
5k/10k/20k snapshots in addition to the latest three, so early states survive
later checkpoints. No new validation loader or final-only comparison was added.
The frozen affected tests include the 3e-3 optimizer/schedule/resume case.
`submission.json` records the replacement; the runtime verifies the fresh LR,
architecture, milestone settings, runtime versions and source hashes.

Verification **108330** completed **0:0**: **124 tests passed, two CUDA tests
skipped**, 133.22 s pytest / 2m23s job elapsed. This includes syntax checks,
the 30:1 schedule/optimizer/checkpoint-resume case, frozen config/runtime hashes
and launch signal handling. After that passed, **107113 was cancelled at
2026-10-18 12:02:19 KST** (batch termination signal 15 is expected). The new
**108331 started at 12:02:21 KST** on eight GPUs and passed its eight-rank NCCL
preflight and frozen-runtime/config verification. New stdout:
`logs/slurm/mf-lift-lr3m-8gpu-108331.out`. No CI job was added.

Initial post-warmup check on **2026-10-18 14:23 KST** read the complete available
stdout through **9,600 logged steps** (1,132 lines, 89,631 bytes): **zero gradient
guard skips, zero other skips, and zero matched errors**. Muon reached 3e-3 at
5k, with 4.6k further updates observed at that peak. Latest 200-step averages:
loss **0.3510**, FM **0.0544**, lDDT loss **0.3142**, pre-clipping grad norm
**1.31**. The recent 7.4k–9.6k window's logged grad-norm averages range
1.31–2.25; these are interval averages, not per-step maxima. Post-warmup speed
averages **0.831 s/applied update** (~154 noise samples/s globally), with ~1 ms
loader wait. Live driver usage across eight devices was **37.05–37.07 GiB**;
rank-0 PyTorch reserved memory was **36.35 GiB**. A 5k checkpoint exists and is
retained as a milestone. Evidence is in the run's `health_checks/` JSON receipt.
This is early numerical stability evidence, not a claim of final convergence
or structure accuracy. The configuration and running job were left unchanged.

## Training plan

Deliberately the same shape as SimpleFold, so the comparison means something.

- **No validation.** SimpleFold disables it outright (`limit_val_batches: 0.0`,
  `num_sanity_val_steps: 0`) and its checkpoint callback monitors
  `trainer/global_step` with `mode: max`, which makes "best k" mean "most recent
  k". It trains a fixed number of steps and takes the last EMA. This project
  does the same, and every admitted record goes to training.
- **One fixed crop of 1024. No schedule, no stages.** At 1024, 98.85% of chains
  fit whole and 98.4% of all residues fall inside the crop, so a curriculum
  would be buying the last 1.6%. Run A samples one distinct protein per GPU and
  repeats it for noise multiplicity, so all rows share a length; no length
  bucketing is needed. `length_bin: 64` matches the SSM kernel's own padding.
- **One arm.** The parameter-matched transformer control trunk was cancelled
  on 2026-08-21 as too expensive: at
  `copies 8 x grad_accum 2` a step is two micro-steps, so even a short 20k-step
  paired probe runs past a day of wall clock for the two arms, before queue time.
  What this costs is stated plainly rather than quietly dropped — no controlled
  comparison remains, so a difference against SimpleFold cannot be attributed to
  the trunk mixer as against any of this project's declared deviations from it.
  The released SimpleFold checkpoint is now the only comparator: a public
  baseline, not a controlled ablation, differing in PLM, corpus size and
  training infrastructure.

### Run A — `configs/run_a_mamba.yaml`

| | | why |
| --- | --- | --- |
| `d_res` / `n_trunk` | 1024 / 15 | 370.48M folding-model parameters in the final Run A config |
| `d_atom` / `n_atom_layers` | 128 / 4 | atom encoding and decoding remain outside the residue trunk |
| `d_state` / `mimo_rank` | 64 / 4 | final Run A uses the MIMO state-space kernel |
| `max_length` | 1024 | one fixed upper bound; no crop-schedule interface |
| loss | `w_fm` + `w_lddt_atom` only | no auxiliary loss knobs in the folding trainer |
| validation | none | no validation loader or CLI knobs; one `admitted_all` list |

The step count, optimizer, warmup, clipping, timestep distribution, and
16-way noise multiplicity match the declared SimpleFold reference recipe. The
global number of distinct proteins per step remains equal to the DDP world size.

Measured on one RTX 6000 Ada (47.4 GiB), worst case — crop 1024, every residue
and atom slot valid, real forward, objective, backward and AdamW step
(`benchmarks/probe_train_memory.py`, `data/audit/train_memory/`):

| model | crop | batch | peak | ms/step |
| --- | ---: | ---: | ---: | ---: |
| 1024/16 | 1024 | 4 | 18.80 GiB | 354 |
| 1024/18 | 1024 | 8 | 34.70 GiB | 753 |
| 1024/20, `d_atom` 160 | 1024 | 4 | 22.89 GiB | 444 |

The exact all-atom lDDT term was measured separately on the same GPU with
batch 8 and 512-row cutoff-neighbor chunks. At 2,048 / 4,096 / 8,192 resolved
atoms per protein, forward+backward took 40 / 78 / 154 ms and peaked at
115 / 253 / 528 MiB. The implementation discovers every ground-truth neighbor
within 15 Å but evaluates predicted distances only for those pairs, so it does
not materialize a dense differentiable pair matrix.

## Evaluation

Two gates, both run. Coverage after exclusions: **CASP14 62/70, CASP15 19/22,
CASP16 18/21**. CAMEO22 has 68/183 targets admitted by the same full-chain
gate, but lacks a frozen domain-local gate and is reported as a secondary
diagnostic rather than a clean-generalization set.

The exact gate alone called CASP14 completely clean and it is not: `T1045s1` is
a 154-residue prefix of a 157-residue training sequence, invisible to
whole-string comparison. The MMseqs2 screen at 30% identity over 80% of the
query excludes 14 targets in total, four of them matching RCSB entries admitted
long before the cutoff — the date rule removes exact matches, never homologs.

Sampler settings must be frozen against the post-cutoff pool, never against
CASP14.

### Latent inference loop budget, EMA10k (2026-10-18)

Production training **108331 stayed running unchanged**. A preserved step-10k
EMA from its frozen code was evaluated at compression 4 and K=1/2/3. The panel
contains 12 distinct chains from eight post-cutoff PDB entries, three chains per
64–128/129–256/257–512/513–1024 length bin, and two fixed seeds per chain.
Selection excluded exact matches against both training FASTAs and did not
depend on embedding-cache availability. All 12 have MMseqs homologs: this is a
development comparison, not evidence of clean generalization.

All **24/24 outputs per K** scored successfully with OpenStructure 2.9.1. The
same EMA, sequences, ESMC-6B conditioning and row-wise noise seeds were used on
one RTX PRO 6000 Blackwell Max-Q, bf16, batch 1, SDE 500 steps / 501 model calls,
tau .01, eps .01, cutoff .99, logarithmic time grid and no geometry guidance.
Forward hooks confirmed K actual latent-stack calls and one atom-decoder call
for every case. Only the inference loop budget changed.

| K | TM-score | all-atom lDDT | backbone lDDT | Mean / median seconds | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 0.82675 | 0.44887 | 0.80892 | 21.91 / 21.82 | 2.186 / 2.346 |
| 2 | 0.82671 | 0.45108 | 0.80896 | 24.70 / 23.77 | 2.186 / 2.346 |
| 3 | 0.82675 | 0.44908 | 0.80896 | 25.71 / 25.72 | 2.186 / 2.346 |

The median paired time ratios versus K2 are **0.9171 for K1** and **1.0811 for
K3**. Times include the sampler and final CPU copy, excluding ESM, loading,
model-forward warmup, PDB writing and scoring. Initial full-sampler overhead
remains in the raw mean: the first K2 sample took 32.37 s. A separate six-case
480-residue control with ten sampler warmup steps observed negligible GC time;
its later GC-disabled K1/K2/K3 times were 21.61/23.56/25.65 s. This does not
attribute the initial delay to GC; its precise cause remains unresolved.
Inference memory does not bound the activation memory of variable-K training.

Entry-cluster bootstrap intervals keep chains from the same PDB together.
K1 minus K2 all-atom lDDT is -0.00221 (95% interval [-0.00403, +0.00094]);
K3 minus K2 is -0.00200 ([-0.00466, +0.00131]). Backbone changes are negligible.
The current evidence supports retaining fixed-K2 training and deferring sampled
training depths; K1 is a candidate for cheaper inference. This early checkpoint
does not establish converged quality or the benefit of training with variable K.

Artifacts: `out/loop-depth-probe-20261018T055310Z/report.md`, `protocol.json`,
`panel.json` / `panel.fasta`, exact and homology audit artifacts, source and
checkpoint hashes, all predictions/references, `timing.json`, per-K scores,
`comparison.json`, `structure_cluster_deltas.json`, and `controlled_timing.json`.
GPU inference 108396, homology 108397, parallel CPU scoring 108402 and timing
control 108403 completed with exit 0. The affected benchmark tests passed **5/5**
(108400), and syntax/lint checks passed. The reusable comparison driver is
`benchmarks/compare_latent_loops.py`; no training configuration was changed.

### Latent contribution and recurrence diagnostics (2026-10-18)

Training 108331 remained untouched. Saved 5k/10k/15k weights and optimizer
states prove the branch is updating: zero-initialized latent `output_proj`
weight RMS grows 0.03246 → 0.09688 → 0.14935. All 73 latent parameter tensors
have optimizer state. Twelve diagnostic backwards on raw10k/15k (two chains ×
three FM times each) found connected, finite gradients for every latent tensor;
these are recomputed checkpoint gradients, not live DDP telemetry.

Matched EMA fixed-input probes use all 12 previous development chains, common
noise and t=.1/.5/.9. Velocity changes below are coordinate-RMS differences
relative to the K2 velocity RMS, averaged over 36 input states per checkpoint.

| Intervention | EMA10k | EMA15k |
|---|---:|---:|
| Remove entire latent residual | 0.7865% | 0.7560% |
| Keep local readout, zero recurrent context | 1.3810% | 1.3435% |
| Initial compressed readout, no recurrent refinement | 1.5351% | 1.4906% |
| Reset prior depth state on each pass / K1 | 0.2018% | 0.1971% |
| K4 | 0.1927% | 0.1900% |
| K8 | 0.1958% | 0.1958% |

The K1/K2 one-step clean-coordinate difference is about 0.0145 Å, not a final
sample RMSD. Thus a connected, trained branch does not establish useful
additional recurrence. Training-depth extrapolation cannot test the benefit
of training a deeper shared core or a broader recurrent state.

A scale trace on a fixed short-chain input at t=.5 localizes a concern inside
the first shared BiMamba at final depth1. EMA10k SSM input-projection RMS is
28.1, directional Mamba output RMS 3.81e7, and the final latent RMS 2.22e9.
The readout RMSNorm brings its output RMS back to 1.01. Zeroing depth embedding
reduces latent RMS to 2.41; clamping AdaLN modulation to [-1,1] still leaves
2.14e5. EMA15k/20k final RMS is 1.28e8/3.77e8 and falls to 2.98/5.36 without
depth embedding. Float64 measurements on cloned activations rule out a
float32-square overflow or hook alias as the explanation. The experiment
localizes depth-conditioned gain amplification, not its exact learned kernel
coefficient or a proven normalization fix. Small raw state gradients are scale
dependent; parameter gradients remain connected. Do not infer positive SSM
poles, a detached graph or a globally diverged training run from these values.

Full SDE500 ablations use four predeclared targets, one per original length
bin, seed0, EMA10k, with a newly sampled same-GPU K2 control; each route scores
4/4 with OpenStructure2.9.1.

| Route | TM-score | all-atom lDDT | backbone lDDT | Mean aligned CA RMSD vs K2 Å |
|---|---:|---:|---:|---:|
| Entire latent bypass | 0.85125 | 0.43600 | 0.81925 | 0.30269 |
| Local-only readout | 0.85975 | 0.46125 | 0.82050 | 0.61381 |
| Initial compressed readout | 0.86175 | 0.45475 | 0.82050 | 0.71683 |
| K2 control | 0.84250 | 0.43300 | 0.81525 | 0 |
| K8 | 0.84225 | 0.43125 | 0.81525 | 0.00668 |

The four-target result favors simpler readouts here, but is too small to claim
a general improvement. New K2 PDBs are not bitwise identical to the previous
independent run: raw CA differences are 0.027–0.074 Å and TM-score changes at
most .001. Their precise numerical-reproducibility cause was not isolated;
all ablation claims use the new matched control.

Next architectural priority is to validate internal Mamba output scaling and
depth/AdaLN conditioning before increasing K. A later broader refiner could
redistribute the 15 one-pass residue blocks into nine prelude blocks plus three
shared residue blocks executed twice, coupled to a larger compressed core.
This preserves 15 full-resolution block executions but changes parameter
sharing and adds latent work; it needs separate training and measurement.
Repeated decoder readouts alone do not feed atom state back into recurrence.
No numerical intervention or architectural change was applied to production.

Artifacts: `out/latent-path-probe-20261018T075000Z/report.md`,
`checkpoint_history.json`, `attribution.json`, `gradients.json`,
`scale_trace.json`, `summary.json`, audit/checkpoint/source hashes and per-route
predictions/scores. CPU preparation108463, GPU probe108464, scoring108465 and
scale trace108467 completed with exit0; **41 affected tests passed**. The
read-only driver is `benchmarks/probe_latent_contribution.py`.

## Commands

```bash
./pipeline/run_all.sh                        # build the corpus, stages 00–10
sbatch scripts/slurm_build_data.sh           # the same on a compute node
sbatch scripts/slurm_compute_missing_embeddings.sh   # GPU: fill embedding gaps
pytest -q tests/test_corpus_invariants.py    # check the corpus against the contract
PYTHONPATH=src python scripts/train.py --config configs/run_a_mamba.yaml
```

## Verification

- Latent residual checks: `tests/test_latent_mamba.py` exercises masked
  compression, readout, backward, shared-loop gradient flow and midpoint
  placement on CPU with a scan substitute; `tests/test_core_runtime.py` covers
  config/build/forward wiring and rejects invalid loop counts/modes. Conditioned
  tests check full rollout gradients, fp32 depth grids under bf16 autocast and
  the declared added checkpoint keys. LiFT checks additionally verify detached
  target anchors, masked MSE values, one alignment/final-only structure losses,
  shared-decoder call counts, paired final predictions and DDP iterations.
  The resource-probe API test uses CPU
  substitutions and is not hardware evidence. Research sources and the staged
  experiment are recorded in README's research section.
  Real bf16 CUDA padding and backward checks extend
  `tests/test_padding_equivalence.py`. They require bitwise output invariance
  when masked coordinate/PLM payloads change at a fixed shape, and bound both
  normalized RMS and peak differences across shapes by two bf16 epsilons.
  Old/new Mamba controls locate the original overly strict 0.002 comparison's
  first discrepancy in the CA projection, before any SSM. FP32 projections
  reduce but do not eliminate rounding through the remaining bf16 scans.
  Repeated real forward/backward also checks stable JIT cache counts and exact
  output/input-gradient/weight-gradient agreement. No CI jobs were added. CPU checks do not
  establish actual Mamba-kernel correctness, GPU runtime or memory.
- `tests/test_corpus_invariants.py` re-derives every admission rule from the
  Boltz manifest and the files on disk, and deliberately does not import the
  pipeline: a stage that mis-applies a filter still writes a self-consistent
  `report.json`, so the report is not evidence for itself.
- After any corpus change, rerun that suite and `pipeline/11_verify_training_readiness.py`.
- Before a long run: config diff reviewed, W&B name/tags set, no run writes into
  an existing output directory.

## Decisions taken

1. **Admitted set**: 161,613, the SimpleFold-exact figure, always reported with
   the 158,210 the farm holds.
2. **Final Run A model size**: `d_res=1024`, `n_trunk=15`, `d_atom=128`, `n_atom_layers=4`
   — 370,480,666 folding-model parameters (370.48M) in the config.
3. **Final Run A Mamba settings**: `d_state=64`, `mimo_rank=4` (MIMO), `headdim=64`,
   `expand=2`.
4. **No transformer control arm** — cancelled 2026-08-21 on cost.
5. **Crop**: one fixed 1024. No schedule.

## The geometric fine-tune is load-bearing

It was recorded as a phase that "may or may not happen". The measurements say it
has to. lDDT is the only term in the main run that sees bond-scale geometry — a
1.5 A bond is 0.19-0.39% of the FM target's variance — and its gradient scales
with `(1-t)`, so under `alpha_mode: const` the 31.5% of draws above t = 0.8
carry 9.8% of it. The main run therefore learns global structure and leaves
stereochemistry to the fine-tune, which is SimpleFold's split and is why `const`
stays. Skipping the fine-tune means shipping a model whose bonds were never
directly supervised.

### Its time schedule is `uniform`, paired with `ramp`

Decided 2026-08-28. The main run samples `t` from SimpleFold's convex form with
`t_uniform_weight: 0.10`, which — measured, not assumed — puts only 1.8% of
draws below t = 0.1 and 4.4% of lDDT's gradient mass there. Raising that
constant from SimpleFold's 0.02 was meant to widen the low-t floor and narrows
it instead, because averaging against `U(0,1)` shrinks both tails rather than
fattening the left one.

The fine-tune uses `t_schedule: uniform` rather than a repaired mixture: it is
already implemented, it removes the schedule as a confound, and it needs no new
sampler. Over 4M draws, against the main run's `logit_normal`/`const`:

| | t<0.1 draws | t<0.1 grad mass | t>0.8 draws | t>0.8 grad mass |
|---|---|---|---|---|
| main run, `logit_normal` w=0.10 / `const` | 1.8% | 4.4% | 31.5% | 9.8% |
| fine-tune, `uniform` / `ramp` | 10.0% | 14.3% | 20.0% | 11.8% |

Low-t exposure rises 5.5x and its gradient mass 3.3x. High t loses draws yet
*gains* gradient mass, because `ramp` offsets the `(1-t)` decay — which is why
the two go together. `uniform` with `const` would leave t > 0.8 holding 4.0% of
the gradient, and that band is where bond-scale error is the residual, so
`uniform` alone would trade one starved end for the other.

The first separately declared phase (`scripts/slurm_geo_finetune.sh`) completed
10k steps as job 61311 on 2026-09-02 from the final 300k EMA with a fresh
optimizer: uniform time, ramp weighting, and bond/angle/non-bonded-clash
weights of 1.0. Its final 2k-step means improved bond MAE from 0.0553 to 0.0476
A and angle MAE from 3.388 to 3.076 degrees, but the old clash diagnostic only
moved from 74.3 to 67.7/1k. That v1 clash was not benchmark-aligned: it used a
0.4-A overlap tolerance, excluded graph distance <=3 (therefore removing 1-3
and 1-4 pairs), and used the resolved-reference mask even though inference
emits every canonical atom.

The corrected v2 definition uses OpenStructure's heavy-atom floors: ordinary
pairs use summed element VDW radii minus 1.5 A, while S-S uses its separate
2.03-1.00 = 1.03-A floor. Only direct intra-residue bonds and the consecutive
peptide C(i)-N(i+1) bond are excluded; 1-3 and 1-4 pairs remain active. Clash
masking uses every canonical output atom, not only atoms resolved in the
training reference. A Huber barrier begins 0.1 A outside the hard threshold,
so atoms near the boundary still receive a gradient, and exact coincidences
receive a deterministic finite descent direction. Detached CA-centred bounding
spheres find a conservative residue neighbour list; gradient-checkpointed
256-residue-pair chunks bound the differentiable workspace without a learned
pair matrix. Hard clashes/1k atoms, smooth clashes/1k atoms, and mean overlap
are logged separately from the optimization surrogate. The continuation is
declared by `scripts/slurm_geo_clash_v2.sh` and starts from the v1 EMA.

Baseline defaults for all three geometry terms remain zero, so the 300k Run A
objective and its comparison contract are unchanged. The fine-tune writes to a
new output directory, initializes both train weights and EMA from the Run A EMA,
and refuses to overwrite an existing run.

## Validated on GPU (2026-08-22, jobs 57848 / 57856 / 57859)

Everything here is measured. Nothing in this section is an estimate.

- **Masking.** A padded batch matches each sequence run alone, and padded
  outputs are exactly zero. This rules out the depthwise-conv boundary, the
  `_flip_by_mask` contiguous-prefix assumption, the pooling softmax, the
  `BackboneStreamMixer` axis order and the AdaLN `repeat` broadcast — all of
  which fail silently rather than raising.
- **Memory.** `peak = 4.89 GiB + 2.538 MiB/residue`, eight points at or above
  1024 residues, max residual 0.223 GiB. Depends on the total residue count and
  not on how crop and batch divide it, checked directly. Add ~3 GiB for the EMA
  copy and DDP buckets: `copies 8 x 1024` projects to 28.2 GiB, 59% of a 47.4
  GiB card; 16 copies does not fit.
- **The model learns.** Four real chains, 400 steps: fm -76%, lDDT loss -21%.
- **Distributed.** One rank, two ranks, checkpoint and resume all pass. The
  per-interval packed metric all-reduce does not hang; the step counter
  continues correctly across a resume.
- **The loader is not the bottleneck.** `data_wait` settles at **0.6%** of step
  time after the first batch. This replaces an estimate of ~26x headroom with a
  reading from the training loop itself.
- **Step time, split (A5000, 768 residues, warm kernels, 422 ms/step).**
  backward 203.3 ms (48.2%), forward with its metric syncs 155.3 ms (36.8%),
  grad clip and AdamW 59.5 ms (14.1%), collate 2.9 ms, host-to-device 0.7 ms,
  unaccounted 0.0 ms. The clip-and-optimizer share is worth naming: 59.5 ms on
  370M parameters is about five hours of a 300k-step run spent on nothing but
  the optimizer.
- **A 3.7x step-time discrepancy is closed, and it was measurement.** The same
  workload read 1091 ms/step in the first overfit probe and 422 ms here. The
  difference is TileLang compilation landing inside the timed region: that probe
  has no prewarm stage and the kernel cache was cold. Neither `use_rigid_align`
  nor the metric `.item()` syncs — the two candidates named at the time —
  accounts for it. Time any step only with warm kernels.
- **The rollout path runs.** `scripts/rollout.py` sampled 6 CASP16 targets under
  both ODE and SDE from a smoke checkpoint: EMA weights loaded, sequence-only
  examples built, structures written. `t1208s1.pdb` carries 2,619 ATOM records,
  which is exactly the atom_mask count computed independently for that sequence.
  The geometry report correctly calls the output garbage — bond RMSD 37-56 A and
  0.0% of CA-CA steps within 0.5 A of ideal, from a model trained 40 steps whose
  zero-initialised head leaves the sampler integrating noise. A metric that says
  "bad" for a model that is bad is the property being checked here.
- **The chain index is built and findable.** 498,546 chains (RCSB 229,705 +
  AFDB 268,841) in 2,424 s. The loader resolves both caches to the files
  `pipeline/13` wrote, which is the point that was broken: the stage builds from
  absolute paths and the loader looks up from config-relative ones, and before
  the key was made spelling-independent the two never met. `slurm_train.sh`'s
  guard passes, so training starts without paying the rebuild.
- **CASP embeddings exist**: 113/113 across the three sets (lengths 35-881), in
  `data/casp_esmc6b`, kept apart from the training caches.
- **Prewarm is a first-run cost only.** Four shapes compiled in 900.0s cold and
  22-37s warm: TileLang caches kernels on disk. The 16 shapes of crop 1024 at
  bin 64 should cost about an hour once, and near nothing on every restart.
- **NCCL is broken on gpu2.** `benchmarks/probe_nccl.py`, which imports nothing
  from this project, segfaults for every NCCL variant tried (with `device_id`,
  without, and with P2P and SHM disabled) while gloo passes init, barrier and a
  value-checked all_reduce. Re-run that probe on any node before trusting it.
  Whether the 6000ada nodes are affected decides whether Run A can use NCCL.

## Open decisions

1. `total_steps`, `lr`, `warmup_steps`, and how many GPUs Run A takes.
2. Whether to embed the post-cutoff pool. Not needed under the SimpleFold-style
   plan, which selects nothing; needed if a selection set is ever wanted.

## Not yet done

- The rollout script. This is now the only path to any number at all: with the
  control arm cancelled, an external-benchmark rollout is the sole evidence the
  project produces. `src/mambafold/sampling/` and `src/mambafold/structure_io.py`
  exist and have zero callers because nothing invokes them yet.
- CASP domain-level homology queries: the policy requires them, stage 10 runs
  whole-chain queries only.
- The loader chain index. `pipeline/13_prebuild_loader_caches.py` exists but has
  not completed a run, so no training job has started from a warm cache and
  `data/audit/loader_caches.json` does not exist yet.
- H100 and RTX PRO 6000 shared-memory limits are predicted, not measured.

### Historical failed residue+latent transplant (2026-10-18; superseded)

The archived recipe at `out/residue-latent-refiner-20261018/code/configs/` selected
`residue_refiner: true`. Existing configs default false, preserving the legacy
module layout, strict checkpoint loading and original computations. Production
108331 continues from its frozen source/config without migration.

The superseded pilot executed atom encoding once, nine independent residue prelude
blocks, then a shared core of three residue blocks followed by compression,
four width 512 latent blocks and residue restoration, repeated K=2, then the
shared atom decoder. Every pass freshly compresses the updated residue state;
there is no nested K-by-K loop or detached recurrent state. The K=2 residue
execution count is 9+3*2=15, while independent residue blocks decrease to 12.
Latent work increases; equal block counts do not establish equal speed.
Compression 4/8, fixed K2 and the existing LiFT/final supervision remain. The
LiFT prelude is the nine-block residue output, with no latent readout; training
reads this and each core output through the shared decoder. Inference decodes
only the final state; atom encoder features and FM inputs stay fixed.

Only the new prelude/core use stable blocks: condition RMS normalization,
tanh-bounded AdaLN scale/shift/gate, official Mamba3 headwise pre-gate output
normalization, and non-affine RMS normalization of the combined directional
SSM and FFN residual updates. Pre-gate normalization alone cannot bound a
subsequent learned Z gate; the additional residual normalization addresses
that location. Each residual subbranch has token RMS at most one for finite
values, independent of its unnormalized gain. The latent restoration uses
tanh after its zero-initialized output projection; this bounds its update
without introducing a second zero gate. Internal nonfinite kernel values
remain failures, not values to hide with nan_to_num. Atom modules retain their
existing conditioning. Controlled GPU evidence below validates the observed scale failure; this
does not establish folding accuracy or long-run optimization improvements.

The separate pilot uses an explicit weight-only EMA20k migration, fresh
optimizer, no production writes: prelude copies old residue blocks 0..8, shared
residue core copies old 9..11, and width 512 latent/norm tensors initialize fresh.
Old 12..14 and width 256 latent weights are not silently reshaped. It trains
400 steps on 128 admitted pre-cutoff chains, crop 256, two corruption copies,
Muon 3e-3/AdamW 1e-4 with 50-step warmup and clip 2; this is a small pilot, not an
exact checkpoint resume. Fixed-noise one-step evaluations use all 12 existing
post-cutoff development chains at t=.1/.5/.9. Full SDE500 evaluation uses the
predeclared first target per length bin, seed0, K1/2/3 with common noise. All
12 have known training homologs; no generalization claim is allowed. Protocol,
weight mapping, source/input hashes, gradients, timing, internal scales,
coverage and scored failures must accompany any reported result.

Validation completed separately, with production 108331 still running:

- **Numerical control:** EMA20k's identical learned latent weights, with only
  the stabilization bundle and fresh unit native-norm scales, reduce the second
  pass's latent RMS from **3.77394e8 to 2.38172** on the original fixed t=.5
  probe. This is a controlled reproduction; it does not isolate which part of
  the bundle contributes most. The newly trained refiner's two latent RMS
  values are 5.70/5.36 on that input. Native kernel failures/nonfinite values are
  not silently converted or skipped in the regression checks.
- **Training:** 400 weight-only warm-start updates, **0 skips**; unclipped
  gradient norm median 1.364, p95≈2.73, max 45.19, followed by clip 2. With crop 256,
  two copies and one GPU, warm update mean 0.274s, median 0.265s, peak allocated
  **4.89GiB**. These timings exclude collation/transfers and cold compilation;
  this is not an end-to-end 8-GPU throughput measurement.
- **Fixed-input causal check:** all 12 chains × 3 times are covered. K1→K2
  reduces FM MSE 0.200189→0.180263 (9.95%) and soft-lDDT loss
  0.504190→0.474078. Both improve per-chain averages on 12/12. Velocity changes
  **10.29%** vs legacy 0.193%; resetting recurrent history exactly restores
  the K1 result. Thus the recurrent residue state materially affects output.
  Zeroing only latent restoration changes velocity **0.490%** and yields
  FM 0.180046, soft-lDDT loss 0.474157. Its additional quality benefit remains
  unproven; the broader residue recurrence accounts for most of the effect.
  Two actual backwards find all 99 residue-core and 132 latent-stack tensors,
  plus compression/readout/depth tensors, connected with finite nonzero group
  gradients. Current-source K2 fixed-input summaries exactly match the frozen
  training snapshot; the extra masking at the refiner entrance preserves these
  already masked production inputs.

Full SDE500, seed0, OpenStructure2.9.1 with its standard stereochemical checks:

| K | Coverage | TM-score | all-atom lDDT | backbone lDDT | Mean seconds/sample | Max allocated GiB | Aligned CA RMSD vs K2 Å |
|---|---|---:|---:|---:|---:|---:|---:|
|1|4/4|0.29250|0|0.41300|21.13|1.923|5.68|
|2|4/4|0.31275|0|0.45975|28.69|1.930|0|
|3|4/4|0.32225|0|0.45825|35.17|1.930|4.31|

All outputs are scored; no failed targets were removed. K2 improves mean
backbone quality over K1, but TM-score improves on only 2/4 targets. All-atom
lDDT is 0 on every route/target, with numerous clashes and bad bonds/angles in
raw scorer outputs. This is poor absolute folding quality. The short migrated
pilot has not recovered the legacy checkpoint's fixed-input FM 0.039301 and
must not replace production. Its scores are not comparable to the earlier
EMA10k ablations as a matched architecture trial: checkpoint, parameter
sharing, normalization, latent capacity and training exposure differ.

A same-initialization inference-only control preserves legacy prelude
normalization and reduces initial K2 FM 0.388708→0.239978, still well above
legacy FM 0.039301. This localizes part of the migration discontinuity to
prelude scaling; it does not demonstrate that dropping stabilization improves
trained folding quality. The archived pilot architecture stayed unchanged during these measurements;
the repair below supersedes it. All
12 development chains have known training homologs; the four sampled chains
come from three PDB entries. Neither the small pilot nor these correlated
inputs support a generalization claim. Random K training remains deferred.

Verification: **162 CPU tests passed**, 3 CUDA-dependent checks skipped on CPU;
**21 real-CUDA tests passed** (ratio 4/8 padding, recurrent LiFT backward and
native normalization gradients included). Consecutive DDP tests use single-rank
Gloo, not an 8-GPU run of the new architecture. Syntax, targeted Ruff checks and
`git diff --check` pass. No dependency updates or production mutations were
required. Full-resolution independent blocks 12, latent independent blocks 4;
parameter count **328,671,386**, unchanged across evaluation K.

Artifacts: `out/residue-latent-refiner-20261018/report.json`,
`protocol.json`, `migration.json`, `input_verification.json`,
`training.json`, `trained.pt`, `trained.json`, `causal.json`,
`trained_gradients.json`, `same_weight_stabilization.json`,
`current_source_equivalence.json`, `sampling_protocol.json`,
per-K predictions/scores and source/audit/checkpoint hashes. Jobs 108499
(training/sampling),108504 (causal/scale checks),108505 (scoring),108512
(coverage aggregation) completed exit0. CPU/GPU test receipts are retained.

Checkpoint runtime metadata now records actual native output-norm modes
(`mixed` when atom and stable residue/latent scans differ). Legacy all-disabled
models retain false; this metadata change does not alter computations.

### Repair of the zero all-atom lDDT regression (2026-10-18)

Production 108331 remains unchanged. Controlled EMA20k probes use all 12 previous
development chains and the same three FM times. The baseline below keeps all 15
learned trunk blocks and bypasses the old ill-conditioned latent branch.

| Intervention before training | FM MSE | Decoder input RMS | Backbone bond error Å |
|---|---:|---:|---:|
| Complete baseline |0.039257|141.17|0.04885|
| Legacy including old latent |0.039306|142.27|0.04825|
| Only replace trunk normalization by the stabilization bundle |0.380695|10.88|0.44212|
| Truncate trunk to 9, original normalization |0.264755|43.61|0.51267|
| Truncate to 9 and retain original PLM injection indices |0.261358|41.31|0.51093|
| Prior tied transplant, before training |0.388707|11.28|0.51780|

The normalization-only control retains all learned backbone weights and block
order. It alone destroys learned feature amplitudes; original copied AdaLN and
SSM projections no longer compute the functions their decoder was trained to
read. Truncation is another independent discontinuity; restoring PLM injection
indices alone does not repair it. The old migration copies valid tensor shapes
but does not preserve learned computation. Equal execution counts are not a
function-preserving conversion of independently learned suffix blocks.

On the predeclared first target, SDE500 with no recovery training gives standard
all-atom lDDT 0.755 for the intact legacy model,0.763 with old latent bypassed,
and 0 for either trunk-normalization-only or the initial failed transplant.
The normalization-only result has 91,770 clashes and 808 bad bonds, versus 4/6 for
the baseline. This reproduces the zero-score failure before optimizer updates;
it cannot be explained solely by a short 400-step training budget.

Paired rescoring of all four failed pilot K2 structures gives standard all-atom
lDDT 0 for 4/4, but0.370–0.459 without stereochemical checks. The zero is the
standard metric's response to invalid geometry, not a missing-score default.
The normalization-only structure is much worse even without those checks
(lDDT 0.054). Unfiltered scores remain diagnostics; standard OpenStructure
scoring and its geometry filters remain unchanged.

The corrected recipe preserves all 15 learned trunk blocks, original FM
conditioning, original normalization and PLM injection at 5/10. It replaces the
old latent residual with a fresh additive shared core: project 1024→512 channels,
three full-resolution residue blocks, ordered sequence compression 4/8, four
latent blocks, restore 512→1024 through one zero-initialized projection; repeat
K2. Both new stacks use stable conditioning and native output normalization.
This avoids putting new normalization onto trained trunk functions. Each pass
feeds its restored residue stream into the next. One final readout projection
opens the new correction without a second zero gate. The new core's channel
bottleneck limits added cost while preserving every residue position before
sequence compression. The full trunk is retained, so this deliberately adds
compute compared with the failed 9+3×2 redistribution.

`initialize_residue_refiner` admits only identical complete backbone keys,
shapes, hyperparameters, normalization and injection schedule, with a fresh zero
refiner readout. It refuses trunk truncation or prefix-based partial loading.
`--initialize_refiner_from` loads checkpoint EMA backbone weights before DDP,
EMA construction and optimizer setup, records source hash/step and exclusions,
and starts fresh training state. It cannot be combined with resume/reset/start
step overrides. The excluded old latent tensors are explicit in the receipt;
initial functional equality is to the baseline with that replaced branch
bypassed, not to the old unstable latent computation. Legacy/default models and
ordinary full-state resume keep their existing behavior.

The source checkpoint must also carry matching Mamba3 revision, module hash and
decay metadata. The file-initialization API rejects missing/incompatible runtime
receipts before changing model weights, including old softplus checkpoints.

Repair validation completed using the standard OpenStructure2.9.1 scorer:

| Model / stage | Coverage | TM-score | all-atom lDDT | backbone lDDT |
|---|---|---:|---:|---:|
| Complete baseline, old latent bypassed |4/4|0.91450|0.65800|0.84200|
| Corrected refiner before any training |4/4|0.91450|0.65850|0.84225|
| Corrected refiner 400-step K1 |4/4|0.91475|0.65600|0.84250|
| Corrected refiner 400-step K2 |4/4|0.91475|0.66450|0.84375|
| Corrected refiner 400-step K3 |4/4|0.91500|0.65700|0.84475|

All four repaired K2 structures have positive standard all-atom lDDT
(0.371–0.782), versus zero on all four failed pilot structures. For the first
target, clashes/bad bonds return from 519/558 in the failed trained pilot to 4/6
in the repaired K2 result (baseline 4/6). The long-chain target remains harder
and still has many clashes; this preserves baseline quality, not perfect
stereochemistry. No geometry-guidance postprocessing, scoring-filter changes
or target exclusions were introduced.

Initialization passes 108 exact velocity/latent comparisons (12 targets × 3 FM
times × K1/2/3) and all three LiFT training readouts. More decisively, every
velocity along four SDE500 trajectories is bitwise identical to the baseline
on the same sampled inputs: **2,004/2,004 calls, maximum difference 0**.
Independent-process PDBs still have small numerical drift. An initial attempt
(job 108538) incorrectly required byte equality against earlier-process PDBs
and failed at coordinate-RMS 0.011 Å, max 0.034 Å. Its artifacts are preserved.
The corrected validation checks identical functions on the same evolving
trajectory instead of treating cross-process bitwise reproducibility as a
model contract. Small score differences are not reliable evidence of broad
accuracy gains.

The separate learning pilot freezes the complete source backbone and trains
only 48,901,824 new parameters for 400 updates on the same 128 admitted training
chains, crop 256, two copies, Muon 3e-3/AdamW 1e-4, 50-step warmup, clip 2, existing
LiFT+soft-lDDT. Every non-refiner tensor is verified unchanged afterward;
**0 updates skipped**. Total model parameters: 419,382,490. Warm update mean
0.200s, peak allocated 2.400GiB; this excludes loading/collation/transfers and
only trains the new core. The normal trainer does not implicitly freeze the
backbone; freezing is specific to this diagnostic pilot.

Same-run fixed-input FM MSE improves 0.039244→0.039095 with the trained core.
K1/K2 velocity RMS differs 0.386%; resetting carried history restores K1.
Removing latent context changes velocity 0.311%, with only a small FM/lDDT
change. All 99 residue-core and 132 latent-stack tensors, compression, readout
and depth tensors receive connected finite gradients with nonzero group norms.
The latent stack RMS stays around 1.72/1.64 on the earlier diagnostic input,
without the old amplification. This is evidence of a working, stable learned
correction; it does not establish a long-run benefit from deeper recurrence.

Ordinary SDE500 wall times averaged 28.37/35.76/43.32s for K1/2/3 on the allocated
GPU. The combined pilot retained training tensors while loading inference
weights, inflating its recorded inference memory. The driver now exits training
scope and collects those objects before inference. A separate fresh-process
K2 profile on the longest panel chain (L799) measures 34.05s and 2.433GiB peak,
with no pilot optimizer or second model resident. Timing scopes differ from
end-to-end production throughput and from the doubled coupled admission calls.

**177 CPU tests passed**, 3 GPU-dependent checks skipped on CPU; **21 real-CUDA
tests passed**. Tests cover rejected partial/shape/norm/injection migrations,
readout identity, narrow-core padding/backpropagation, EMA-only checkpoint
initialization, fresh optimizer/EMA state, resume conflicts and runtime guards.
Ruff, syntax and diff-whitespace checks pass. New-architecture 8-GPU training
has not been performed; production 108331 continues from its unchanged frozen
source and config.

Artifacts: `out/refiner-geometry-repair-20261018/report.json`, frozen
`control_code` and `paired_code` snapshots/hashes, controlled probe/score JSON,
paired stereochemical diagnostics, audit/source/reference/ESM hashes, and
`fixed-paired/` admission, initialization samples, migration receipt, 400-step
checkpoint, geometry scores, gradient checks and standalone profile. Jobs 108528
(controls),108535 (control scoring),108546 (paired admission/training/sampling),
108548 (repair scoring),108550 (gradient/runtime/profile) and108551 (aggregation)
completed exit0. The four scored chains come from three PDB entries and all
have known training homologs. This demonstrates regression repair on the
predeclared development panel; generalization and production training efficiency
remain unmeasured for the new architecture.
