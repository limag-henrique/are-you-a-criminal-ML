# Research audit v2

This isolated package audits the derived embedding dataset without changing the
operational face-similarity package or historical artifacts. Public outputs use
pseudonymous record and group identifiers only; images, paths, URLs, names and
individual embeddings are never exported.

## Run

```powershell
python -m research_audit_v2.src.run_all --config research_audit_v2/configs/development.yaml
```

The development configuration is a reproducibility and integration check. The
final configuration records 100 seeds and the requested k grid, but can take a
substantial time on the complete embedding matrix. All configuration files are
JSON documents with a `.yaml` extension, so they require no YAML parser.

## Scientific scope

Labels created from clustering are synthetic. Metrics concerning their recovery
describe internal recovery of a synthetic label; they are neither identity
recognition nor biometric, social, legal or criminal validity. Resampling
intervals are conditional to the audited records and are not population
intervals.
