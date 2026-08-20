# SimpleFold apo benchmark

- Targets: 90
- Length: 90-580
- Contract: five samples per target; maximum TM-score to each state and ensemble metrics.
- Input sequence: reconstructed from the official SimpleFold-3B prediction artifact and cross-checked against EigenFold metadata where available.
- References: the exact official EigenFold metadata pairs used by SimpleFold's released evaluator.
- Reference identity: state1 must have at least 95% sequence identity. Alternate states retain the exact official pairing; some CoDNaS state2 structures are homologous rather than sequence-identical.
- Model outputs are evaluated separately by model size; they are never pooled.
