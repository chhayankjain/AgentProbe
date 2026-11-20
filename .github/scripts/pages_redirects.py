"""Turn a built MkDocs site into redirects to the docs' new home.

The docs moved from GitHub Pages to Cloudflare Pages. This rewrites every HTML
page in the built site as a small page that sends readers (and search engines,
via the canonical link) to the same path on the new host, so old links and
search results keep working. Other files are dropped; llms.txt becomes a pointer to the new host.

Usage: python .github/scripts/pages_redirects.py SITE_DIR OUT_DIR NEW_BASE_URL
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved: {title}</title>
<link rel="canonical" href="{url}">
<meta http-equiv="refresh" content="0; url={url}">
<script>location.replace({js_url} + location.hash);</script>
</head>
<body>
<p>The agentprobe-toolkit docs moved to <a href="{url}">{url}</a>.</p>
</body>
</html>
"""

NOT_FOUND = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved</title>
<script>
  var prefix = "/agentprobe-toolkit/";
  var path = location.pathname.indexOf(prefix) === 0 ? location.pathname.slice(prefix.length) : "";
  location.replace({js_base} + path + location.search + location.hash);
</script>
</head>
<body>
<p>The agentprobe-toolkit docs moved to <a href="{base}">{base}</a>.</p>
</body>
</html>
"""


def main(site: Path, out: Path, base: str) -> None:
    base = base.rstrip("/") + "/"
    out.mkdir(parents=True, exist_ok=True)
    for page in site.rglob("*.html"):
        rel = page.relative_to(site).as_posix()
        if rel == "404.html":
            continue
        path = rel.removesuffix("index.html")
        url = base + path
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            PAGE.format(title=html.escape(path or "home"), url=html.escape(url), js_url=repr(url))
        )
    (out / "404.html").write_text(NOT_FOUND.format(base=html.escape(base), js_base=repr(base)))
    (out / "llms.txt").write_text(f"# agentprobe-toolkit\n\nThe docs moved to {base}llms.txt\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
