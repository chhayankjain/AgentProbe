# Release Process

This documents how AgentProbe releases are created.

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking API changes
- **MINOR** — new features, backward-compatible
- **PATCH** — bug fixes only

## Steps

### 1. Prepare the release branch

```bash
git checkout main && git pull
git checkout -b release/vX.Y.Z
```

### 2. Bump version numbers

Update in **three** places:
- `pyproject.toml` → `version = "X.Y.Z"`
- `agentprobe/__init__.py` → `__version__ = "X.Y.Z"`
- `CITATION.cff` → `version: "X.Y.Z"` and `date-released`

### 3. Update CHANGELOG.md

Add a new section at the top with the release date and changes:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Fixed
- ...

### Changed
- ...
```

### 4. Open a PR, merge, tag

```bash
git add -A && git commit -m "Bump version to X.Y.Z"
git push -u origin release/vX.Y.Z
gh pr create --title "Release vX.Y.Z" --base main
# Merge the PR, then:
git checkout main && git pull
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 5. Create GitHub release

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

Review the auto-generated notes and edit if needed.

### 6. Clean up

```bash
git push origin --delete release/vX.Y.Z  # delete the release branch
```

## Who can release

Currently only the repo owner (@chhayankjain) can create releases. Branch protection on `main` requires admin privileges to merge without review.
