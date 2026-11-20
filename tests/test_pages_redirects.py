"""The redirects that replace the old GitHub Pages docs site (.github/scripts)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "pages_redirects.py"
if not SCRIPT.exists():  # the source distribution ships without .github/
    pytest.skip("pages_redirects.py is not in this checkout", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("pages_redirects", SCRIPT)
pages_redirects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pages_redirects)

BASE = "https://agentprobe-toolkit.pages.dev/"


@pytest.fixture
def built(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    for page in ["index.html", "guides/retries/index.html", "reference/scenarios/index.html"]:
        (site / page).parent.mkdir(parents=True, exist_ok=True)
        (site / page).write_text("<html>real page</html>")
    (site / "404.html").write_text("<html>not found</html>")
    (site / "assets").mkdir()
    (site / "assets" / "logo.svg").write_text("<svg/>")
    (site / "sitemap.xml").write_text("<urlset/>")
    out = tmp_path / "redirects"
    pages_redirects.main(site, out, BASE.rstrip("/"))  # trailing slash is optional
    return out


@pytest.mark.parametrize(
    ("page", "target"),
    [
        ("index.html", BASE),
        ("guides/retries/index.html", BASE + "guides/retries/"),
        ("reference/scenarios/index.html", BASE + "reference/scenarios/"),
    ],
)
def test_every_page_redirects_to_the_same_path(built: Path, page: str, target: str):
    html = (built / page).read_text()
    assert f'<link rel="canonical" href="{target}">' in html
    assert f'content="0; url={target}"' in html
    assert f"location.replace('{target}' + location.search + location.hash)" in html
    assert "real page" not in html


def test_unknown_paths_keep_path_query_and_fragment(built: Path):
    html = (built / "404.html").read_text()
    assert 'var prefix = "/agentprobe-toolkit/";' in html
    assert f"location.replace('{BASE}' + path + location.search + location.hash)" in html


def test_only_redirects_and_a_pointer_are_published(built: Path):
    files = sorted(p.relative_to(built).as_posix() for p in built.rglob("*") if p.is_file())
    assert files == [
        "404.html",
        "guides/retries/index.html",
        "index.html",
        "llms.txt",
        "reference/scenarios/index.html",
    ]
    assert BASE + "llms.txt" in (built / "llms.txt").read_text()


def test_redirects_are_not_hidden_from_search_engines(built: Path):
    """A noindex on a redirect page would stop search engines following it."""
    for page in built.rglob("*.html"):
        assert not re.search(r"noindex", page.read_text())
