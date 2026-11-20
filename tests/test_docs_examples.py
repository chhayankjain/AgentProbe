"""Run the code examples on the docs site, so they can't drift from the API.

Every ```python block on a page is executed in order, in one namespace, and then
every ``test_*`` function the page defined is called. Blocks fenced as ```py are
illustrative (they import your own agent, or show a fragment) and are skipped;
both render the same on the site.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted([DOCS / "getting-started.md", *(DOCS / "guides").glob("*.md")])
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


@pytest.mark.parametrize(
    "page",
    [p for p in PAGES if BLOCK.search(p.read_text())],
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_page_examples_run(page: Path):
    namespace: dict = {"__name__": f"docs_{page.stem.replace('-', '_')}"}
    for block in BLOCK.findall(page.read_text()):
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102 - our own docs
    tests = [fn for name, fn in namespace.items() if name.startswith("test_") and callable(fn)]
    assert tests, f"{page.name} has python blocks but defines no test_* function"
    for fn in tests:
        fn()


def test_every_guide_has_runnable_examples():
    """A guide whose examples are all ```py would silently go untested."""
    untested = [p.name for p in (DOCS / "guides").glob("*.md") if not BLOCK.search(p.read_text())]
    assert untested == ["ci.md"]  # YAML only
