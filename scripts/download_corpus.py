"""Download a fixed PostgreSQL version; preserve sources, hashes, and license."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "indexes-intro", "indexes-types", "indexes-multicolumn", "indexes-ordering",
    "indexes-bitmap-scans", "indexes-unique", "indexes-expressional", "indexes-partial",
    "indexes-index-only-scans", "indexes-examine", "transaction-iso", "explicit-locking",
    "mvcc", "ddl-constraints", "ddl-partitioning", "using-explain", "routine-vacuuming",
    "backup-dump", "wal-intro", "warm-standby", "textsearch-intro", "textsearch-indexes",
    "datatype-json", "sql-createindex", "sql-analyze",
]


def main() -> None:
    manifest = []
    for slug in PAGES:
        url = f"https://www.postgresql.org/docs/17/{slug}.html"
        html = urlopen(url, timeout=60).read()
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one("#docContent")
        if content is None:
            raise RuntimeError(f"Documentation content missing: {url}")
        for element in content.select(".navheader, .navfooter, .toc, .titlepage a"):
            element.decompose()
        md = markdownify(str(content), heading_style="ATX").strip() + "\n"
        path = ROOT / "corpus" / f"{slug}.md"
        path.write_text(md, encoding="utf-8")
        manifest.append({"file": path.name, "url": url, "words": len(md.split()),
                         "sha256": hashlib.sha256(md.encode()).hexdigest()})
        print(path.name, len(md.split()), flush=True)
    license_url = "https://www.postgresql.org/about/licence/"
    soup = BeautifulSoup(urlopen(license_url, timeout=60).read(), "html.parser")
    content = soup.select_one("#pgContentWrap") or soup
    (ROOT / "POSTGRESQL_LICENSE.txt").write_text(content.get_text("\n", strip=True))
    (ROOT / "data/corpus_manifest.json").write_text(json.dumps({
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "version": "PostgreSQL 17", "license_url": license_url, "pages": manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
