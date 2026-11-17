# Phase 4.2 — Model artifact management

## Registry design

`src/models/registry.py` centralizes Joblib saving and loading. Each versioned
model consists of:

- A complete fitted pipeline in `artifacts/models/<name>-<version>.joblib`,
  including feature engineering, imputation, scaling, one-hot encoding, and
  the fitted classifier.
- A JSON metadata sidecar with model name/version, UTC training date, dataset
  hash, seed, ordered feature schema and observed dtypes, target mapping,
  dependency versions, model/training configuration, and training-only CV
  metrics.
- `artifacts/models/registry.json`, which maps model/version identifiers to
  the artifact and metadata filenames and their SHA-256 digests.

Artifact and metadata writes use temporary files followed by atomic replacement.
The loader derives filenames from validated model/version identifiers, rejects
symlinked artifact/metadata/registry paths, requires registry membership, and
checks the artifact and metadata digests before `joblib.load`. Callers can
additionally require exact feature-schema and target-mapping matches. Missing,
modified, corrupt, or incompatible artifacts raise `ArtifactError`.

The default model version is configured as `training.model_version` in
`config.yaml`. `train_models` now writes artifacts through the registry rather
than saving bare Joblib files. Existing unregistered model files are not
loadable through this API.

## Trust boundary

Joblib uses Python pickle serialization; loading a malicious pickle can execute
code. SHA-256 detects changes relative to the local registry, and the loader
refuses files absent from that registry, but a registry and its hashes are not
a cryptographic signature if an attacker can modify both. Therefore the
registry directory and its parent must be writable only by trusted operators.
Never add externally supplied or otherwise untrusted model files to this
registry. Do not treat successful deserialization as proof of safety.

## Validation

Integration tests verify registered save/load prediction equivalence,
metadata, dataframe hashing, required schema and target mapping, rejection of
unregistered files, artifact tampering, metadata tampering, missing files,
missing registry, and invalid path identifiers. Model-training tests also
verify that each serialized training pipeline loads with the recorded
provenance and prediction schema.
