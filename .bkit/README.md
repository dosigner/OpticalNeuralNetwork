This repository keeps only shareable `.bkit` scaffolding in git.

Tracked here:
- `workflows/`
- `decisions/`
- `checkpoints/`

Not tracked here:
- `runtime/`
- `state/`
- `audit/`
- `snapshots/`
- machine-local suggestion files

Execution state is intentionally local-only so collaborators can clone the repo
without inheriting another operator's session history or runtime memory.
