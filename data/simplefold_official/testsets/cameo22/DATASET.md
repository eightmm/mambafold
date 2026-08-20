# SimpleFold cameo22 benchmark

- Targets: 183
- Length: 31-709
- Contract: one structure per target; OpenStructure 2.9.1 folding metrics.
- Input sequence: reconstructed from the official SimpleFold-3B prediction artifact and cross-checked against EigenFold metadata where available.
- References: the EigenFold checkout used by SimpleFold's released evaluator; 8QCW-A is the revised RCSB reference.
- Reference identity: state1 must have at least 95% sequence identity. Alternate states retain the exact official pairing; some CoDNaS state2 structures are homologous rather than sequence-identical.
- Model outputs are evaluated separately by model size; they are never pooled.
