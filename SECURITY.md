# Security and restricted inputs

This is research software. No formal security certification or dependency-vulnerability audit is claimed.

Only load checkpoints and serialized artifacts from trusted sources. The archived training and inference tools use PyTorch checkpoint loading, which can execute pickle payloads on the recorded stack. Do not download arbitrary checkpoints and load them to try the model. Checkpoints are not included in this release.

Keep simulator inputs, concentration/conductivity fields, private predictions and credentials outside version control. Do not attach those assets to public issues. Report suspected exposure privately to the repository owner through an available private contact channel, without posting secrets or restricted payloads.

The included content checks detect selected excluded file classes and credential/path patterns. They are a release safeguard, not proof that every possible secret or malicious input has been detected. Dependency upgrades need their own compatibility and numerical checks; do not change a paper-corresponding environment silently.
