# Source provenance and preservation

This repository starts from an allowlisted copy of the reviewed candidate code package, with paper-specific pairing, screening, grouped-split, twin and prior-predictive summaries added from archived outputs. The full development history is excluded.

The primary models, loaders, training entry point, evaluation scripts, numerical CSVs, normalization metadata and fixed splits preserve their original scientific content. Package-maintenance changes add manifest verification, restrict generated outputs to ignored locations, and test noK input invariance and archived-summary consistency. They do not train or change a model.

Each record in `source_snapshot.json` gives the original SHA-256 and copied SHA-256. JSON transformations remove only listed restricted fields whose values expose machine-local paths or asset roots. Every other field is retained; original sources stay outside this repository. Scientific sample identifiers and hashes, including exceptions and the superseded grouped seed-0 run, remain visible.

The original BSD-3-Clause notice is preserved in `LICENSE-BSD-3-Clause-original.txt`. MIT is added at the user's request for the study code. Upstream dependencies are not vendored. A formal origin and copyright review remains a gate before public release; the repository does not establish rights to restricted assets.

Language-model assistance included repository preparation, code inspection, statistical checking and editorial work. Human authors remain responsible for scientific interpretation, source provenance and final release approval. No automated check establishes authorship or guarantees the absence of every possible sensitive item.
# Restricted nested fields

The private pairing audit omits 18 embedded LPF simulator-setup strings. All other diagnostic JSON values, including nominal identifiers, field hashes, pairing evidence and screening results, are preserved. The removed key paths and both source/copy hashes are recorded in `source_snapshot.json`. Fixed split files are copied byte for byte. Git preserves original line endings so evidence hashes remain portable across fresh clones.
