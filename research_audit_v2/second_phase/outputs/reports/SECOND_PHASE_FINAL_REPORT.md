# Second-phase final report

## Executive result

The grouped cross-fitted design completed with zero duplicate-group overlap. Foldwise mean metrics were {'roc_auc': 0.929, 'pr_auc': 0.479, 'balanced_accuracy': 0.725, 'precision': 0.477, 'recall': 0.477, 'f1': 0.477, 'brier': 0.185}. They quantify recovery of a reconstructed synthetic target only. The dispersion across folds is material and should be reported rather than optimized away.

## Leakage

The all-record diagnostic has high leakage risk because target construction and centroid scoring share observations. The grouped cross-fitted design removes this direct reuse: clusters, target selection and centroid are fit in training folds only.

## Limitations

No historical clustering state, documented source attribution, reliable temporal field, supported historical extraction environment, GPU comparison or clean-environment comparison was available. No social, biometric, identity, criminal or causal inference is supported.
