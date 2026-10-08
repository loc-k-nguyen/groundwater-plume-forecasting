# Metadata

`param_split_fixed.json` records the frozen parameter-level train, validation, and test split.
`train_stats.json` records the normalization statistics used by the released training and
evaluation code. These files contain identifiers and aggregate statistics only. They do not
contain simulation fields.

The split file is copied unchanged from the canonical experiment configuration so that released
results remain traceable to the exact protocol.
