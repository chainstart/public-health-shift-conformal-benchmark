"""Probe provider URLs and file signatures without downloading complete cohorts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def probe(item):
    result = {key: item[key] for key in ("dataset", "file_id", "url", "format")}
    request = Request(item["url"], headers={"Range": "bytes=0-511",
                                          "User-Agent": "OpenLabs-data-source-check/1.0"})
    try:
        with urlopen(request, timeout=35) as response:
            prefix = response.read(512)
            expected = b"PK\x03\x04" if item["format"] == "zip" else b"HEADER RECORD"
            result.update(http_status=response.status,
                          content_type=response.headers.get("Content-Type"),
                          content_range=response.headers.get("Content-Range"),
                          inspected_bytes=len(prefix),
                          signature_matches=prefix.startswith(expected))
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        result.update(error_type=type(error).__name__, http_status=getattr(error, "code", None),
                      signature_matches=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads((ROOT / "data_sources.json").read_text())["files"]
    with ThreadPoolExecutor(max_workers=2) as workers:
        rows = list(workers.map(probe, items))
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "files": rows,
              "all_signatures_match": all(row["signature_matches"] for row in rows),
              "complete_files_downloaded": False, "cohorts_regenerated": False,
              "scope": "At most 512 bytes per response; no full-file checksums or cohort validation."}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"files_checked": len(rows),
                      "matching_signatures": sum(r["signature_matches"] for r in rows)}))
    if not result["all_signatures_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
