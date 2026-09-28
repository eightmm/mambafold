# Late SDE geometry guidance: development experiment

This is an exploratory inference change to the fixed Run A EMA, not a new
training run or an update to the manuscript benchmark table. It is inspired by
EFF-Dock's use of physical energies during docking/refinement, but MambaFold
currently applies a bounded Cartesian correction rather than EFF-Dock's rigid
fragment translation/rotation.

At selected SDE steps with `t >= 0.90`, the sampler constructs its clean
estimate `x_hat = x_t + (1-t) v`. A reference-free energy combines ideal bond
lengths, ideal bond angles, and the existing emitted-atom clash surrogate.
The ideal values come from OpenStructure 2.9.1's Engh-Huber parameter table,
SHA-256 `24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451`.
The sampler differentiates only the energy with respect to detached coordinates,
then adds the correction to both the next SDE state and self-conditioning. It
does not backpropagate through the folding network or use target coordinates.
Corrections are applied every 10 steps, ramp up from `t=0.90`, and move each
atom by at most 0.02 Å per event; backbone atom gradients are scaled by 0.1.
Setting the maximum move to zero preserves the existing sampler path.

## Four-target CASP14 pilot

These four admitted targets were chosen after inspecting baseline clash counts,
so their mean is **not** an estimate for CASP14 or an external benchmark. Both
arms use the same 500-step SDE, tau 0.01, seed 0, checkpoint, target sequence,
reference PDB, and default OpenStructure 2.9.1 scoring. The guided arm was
recomputed; the baseline arm is the frozen Run A result.

| Target | Standard lDDT baseline → guided | Reported clashes baseline → guided | TM-score baseline → guided |
| --- | ---: | ---: | ---: |
| t1034 | 0.719 → 0.739 | 46 → 34 | 0.927 → 0.927 |
| t1047s1 | 0.443 → 0.455 | 196 → 161 | 0.488 → 0.488 |
| t1049 | 0.060 → 0.075 | 294 → 237 | 0.350 → 0.352 |
| t1064 | 0.062 → 0.070 | 248 → 202 | 0.271 → 0.254 |

Mean standard lDDT changes from 0.321 to 0.335. Mean backbone lDDT remains
0.545, while mean TM-score changes from 0.509 to 0.505. One target loses 0.017
TM-score. The paired target rows and violation counts are in
[`data/late_geometry_pilot.json`](data/late_geometry_pilot.json). Slurm rollout
job 86354 and score job 86355 both completed with exit code 0. This pilot
demonstrates a small score gain on deliberately selected cases, with a structural
tradeoff to monitor.

## Frozen follow-up

The same guidance settings have been queued for all admitted CASP14, CASP15,
and CASP16 targets, using the same fixed 500-step sampler and the exact
reference PDBs from the original Run A evaluation. The CASP14, CASP15, and
CASP16 rollout jobs are 86373, 86374, and 86375; dependent scoring jobs are
86376, 86377, and 86378. Results are not reported until every job exits 0 and
target/reference counts match 62/62, 19/28 pairs, and 18/18, respectively.
CASP14 has been inspected during development; CASP15/16 are post-freeze checks
of this guidance setting, subject to the exposure limits in the manuscript.

The current energy does not enforce all stereochemical constraints, including
chirality and sidechain torsions. A lower clash count alone is not evidence of
an accurate sidechain; the primary acceptance criterion is higher **default**
OpenStructure all-atom lDDT without material loss of TM-score or backbone lDDT.
