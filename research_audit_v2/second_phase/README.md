# Second-phase implementation

This package contains the tested implementation used by the commands in the
parent README. Public outputs are written only below `research_audit_v2/outputs/`.

The pipeline enforces grouped train/test separation and records train-only fit
events for representation, clustering, target selection, centroid, calibration
and threshold selection. Stability is split into stochastic, record-order,
batch-size and representation analyses. PCA-64 is explicitly classified as a
new methodological reconstruction.

CI sets `RESEARCH_AUDIT_SYNTHETIC_ONLY=1` and executes only tests and generated
controls; it never requires repository images or real embeddings.
