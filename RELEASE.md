# Release Process

AgentProbe is published to PyPI as **`agentprobe-toolkit`** (import name `agentprobe`).
Releases are built and uploaded by `.github/workflows/release.yml` using PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/); no API tokens live in the repo.

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking changes to the public `agentprobe` API
- **MINOR** — new features, backward-compatible
- **PATCH** — bug fixes only

A published version is immutable. If a release is wrong, yank it on PyPI and ship a patch release;
never try to re-upload the same version number.

## One-time setup (done once per PyPI account)

1. **GitHub environments.** In the repo settings create two environments, `pypi` and `testpypi`.
   Optionally add yourself as a required reviewer on `pypi` so a tag push pauses for approval.
2. **PyPI pending publisher.** On <https://pypi.org/manage/account/publishing/> add a *pending*
   publisher for project `agentprobe-toolkit` with owner `chhayankjain`, repository `AgentProbe`,
   workflow `release.yml`, environment `pypi`. The first successful upload creates the project.
3. **TestPyPI pending publisher.** Same on <https://test.pypi.org/manage/account/publishing/>
   with environment `testpypi`.

## Dry run (TestPyPI)

Run the **Release** workflow manually from the Actions tab (`workflow_dispatch`). It builds the
package, checks the metadata and README rendering, smoke-tests the wheel, and uploads to TestPyPI.
Check <https://test.pypi.org/project/agentprobe-toolkit/> renders correctly, then:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agentprobe-toolkit
python -c "import agentprobe; print(agentprobe.__version__)"
```

## Steps

### 1. Prepare the release branch

```bash
git checkout main && git pull
git checkout -b release/vX.Y.Z
```

### 2. Bump version numbers

Update in **three** places (the test suite checks they agree):
- `pyproject.toml` → `version = "X.Y.Z"`
- `agentprobe/__init__.py` → `__version__ = "X.Y.Z"`
- `CITATION.cff` → `version: "X.Y.Z"` and `date-released`

Also update the supported-versions table in `SECURITY.md`.

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

Pushing the tag triggers the **Release** workflow, which:
1. builds the sdist and wheel and refuses to continue if the tag does not match `pyproject.toml`;
2. runs `twine check --strict` and installs the wheel in a clean venv;
3. publishes to PyPI via Trusted Publishing;
4. creates the GitHub release with auto-generated notes and the built files attached.

### 5. Verify

```bash
pip install agentprobe-toolkit==X.Y.Z
python -c "import agentprobe; print(agentprobe.__version__)"
```

Edit the generated GitHub release notes if needed.

### 6. Clean up

```bash
git push origin --delete release/vX.Y.Z
```

## Who can release

Only repository admins can push tags to the protected `main` history, and the `pypi` environment can
be restricted to specific reviewers. Both PyPI publishers are tied to this repository and the
`release.yml` workflow, so a fork cannot publish.
