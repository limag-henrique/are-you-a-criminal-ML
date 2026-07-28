# Embedding representation sensitivity

Deferred as a separate predeclared sensitivity. Learned representations (PCA, standardization, whitening) must be fitted only within each cross-fitting training fold. They must not replace the locked primary L2-normalized representation based on observed metrics.
