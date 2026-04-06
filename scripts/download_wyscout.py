"""
Download the Wyscout Open Dataset (Pappalardo et al. 2019) directly from Figshare.

Dataset: https://figshare.com/collections/Soccer_match_event_dataset/4415000/5

Usage:
    python scripts/download_wyscout.py
"""

import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

ROOT = Path(__file__).parent.parent
WYSCOUT_DIR = ROOT / "data" / "raw" / "wyscout"

FIGSHARE_API = "https://api.figshare.com/v2"

# Figshare article IDs and their target subdirectories
ARTICLES = [
    (7770599, "events",  "events.zip"),   # all league event files
    (7770422, "matches", "matches.zip"),  # all league match files
    (7765196, "",        "players.json"),
    (7765310, "",        "teams.json"),
    (7765316, "",        "competitions.json"),
]


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _download_file(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping")
        return
    print(f"  Downloading {label}...", end=" ", flush=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_000_000
        print(f"done ({size_mb:.1f} MB)")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {label}: {e}") from e


def download_all():
    WYSCOUT_DIR.mkdir(parents=True, exist_ok=True)
    (WYSCOUT_DIR / "events").mkdir(exist_ok=True)
    (WYSCOUT_DIR / "matches").mkdir(exist_ok=True)

    for article_id, subdir, filename in ARTICLES:
        dest_dir = WYSCOUT_DIR / subdir if subdir else WYSCOUT_DIR
        dest = dest_dir / filename

        # Get download URL from Figshare
        try:
            article = _fetch_json(f"{FIGSHARE_API}/articles/{article_id}")
        except URLError as e:
            print(f"Network error fetching article {article_id}: {e}")
            sys.exit(1)

        files = article.get("files", [])
        if not files:
            print(f"  No files for article {article_id}, skipping.")
            continue

        download_url = files[0]["download_url"]

        if filename.endswith(".zip"):
            zip_path = dest_dir / filename
            _download_file(download_url, zip_path, filename)
            # Extract into dest_dir
            print(f"  Extracting {filename}...", end=" ", flush=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            print("done")
            zip_path.unlink()  # remove zip after extraction
        else:
            _download_file(download_url, dest, filename)

    print(f"\nDownload complete → {WYSCOUT_DIR}")
    _verify()


def _verify():
    print("\nVerifying files:")
    required = [
        "events/events_Spain.json",
        "events/events_England.json",
        "events/events_France.json",
        "events/events_Germany.json",
        "events/events_Italy.json",
        "matches/matches_Spain.json",
        "players.json",
        "teams.json",
    ]
    ok = True
    for f in required:
        path = WYSCOUT_DIR / f
        if path.exists():
            mb = path.stat().st_size / 1_000_000
            print(f"  ✓ {f} ({mb:.1f} MB)")
        else:
            print(f"  ✗ {f} — MISSING")
            ok = False

    if ok:
        print("\nAll files present. Run feature engineering:")
        print("  python -m src.features.main_feature --leagues Spain")
    else:
        # Show what was extracted
        print("\nFiles in wyscout dir:")
        for p in sorted(WYSCOUT_DIR.rglob("*.json")):
            print(f"  {p.relative_to(WYSCOUT_DIR)}")


if __name__ == "__main__":
    download_all()
