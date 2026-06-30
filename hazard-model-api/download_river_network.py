#!/usr/bin/env python3
"""
Download the TDX-Hydro v2 river network for a region.

TDX-Hydro is a global vector river network derived from TanDEM-X 12 m DEM.
We fetch it through the ICPAC TIPG API (OGC API - Features) so no auth is
needed. Each feature has `stream_order`, `LENGTHKM`, and upstream-area
attributes — RIM2D v5/v6 use these to assign channel width/depth per
segment.

Size guide
----------
Even for country-scale bboxes this stays small (vector, not raster):

    bbox type       features    GeoJSON
    ---------       --------    -------
    city basin       ~50         ~0.2 MB
    country         ~3-5 k       ~5-15 MB
    East Africa     ~50 k        ~150 MB

Source
------
    TIPG (OGC API - Features). The endpoint is read from the environment, not
    hardcoded — set TIPG_API_BASE (and optionally TIPG_RIVER_COLLECTION) in
    .env or your shell. See .env.sample for the Cloud Run + Cloud SQL values.

Output
------
    <out>/tif/river_network_tdx.geojson

Usage
-----
    python download_river_network.py --bbox 36.6,-1.402,37.1,-1.098 --out ./runs/nbo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

from common import add_common_args, parse_region

# TIPG endpoint comes from the environment — the served URL is NOT baked into
# the code. Set TIPG_API_BASE (and optionally TIPG_RIVER_COLLECTION) in .env
# or your shell; see .env.sample for the Cloud Run + Cloud SQL tipg values.
API_BASE   = os.environ.get("TIPG_API_BASE")
COLLECTION = os.environ.get("TIPG_RIVER_COLLECTION", "gha.geoglows_rivers")
if not API_BASE:
    sys.exit("TIPG_API_BASE is not set. Copy .env.sample to .env and set it "
             "(or `export TIPG_API_BASE=...`) before running.")
ITEMS_URL  = f"{API_BASE}/collections/{COLLECTION}/items"
PAGE_LIMIT = 10000   # API max per page


def fetch_all(bbox_str: str) -> dict:
    """Paginate through the TIPG collection until empty page."""
    features = []
    offset = 0
    while True:
        params = {"bbox": bbox_str, "limit": PAGE_LIMIT, "offset": offset,
                  "f": "geojson"}
        r = requests.get(ITEMS_URL, params=params, timeout=60)
        r.raise_for_status()
        fc = r.json()
        feats = fc.get("features", [])
        if not feats:
            break
        features.extend(feats)
        print(f"[tdx] page offset={offset} → {len(feats)} features "
              f"(total {len(features)})")
        if len(feats) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return {"type": "FeatureCollection", "features": features}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    args = ap.parse_args()
    r = parse_region(args)

    if r.dry_run:
        print(f"[size] TDX-Hydro network: typically 0.2-15 MB for any "
              f"country-scale bbox.")
        return 0

    out = r.tif_dir / "river_network_tdx.geojson"
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"[cache] {out.name} already exists ({size_mb:.1f} MB) — skipping.")
        return 0

    bbox_str = f"{r.west},{r.south},{r.east},{r.north}"
    print(f"[tdx] fetching {COLLECTION} bbox={bbox_str}")
    fc = fetch_all(bbox_str)
    n = len(fc["features"])
    if n == 0:
        sys.exit(f"ERROR: TDX-Hydro returned 0 features for bbox {bbox_str}.")

    with open(out, "w") as f:
        json.dump(fc, f)
    size_mb = out.stat().st_size / 1e6
    print(f"[done] {out} ({n} segments, {size_mb:.1f} MB)")

    # Summary of stream orders
    orders = {}
    for feat in fc["features"]:
        so = feat["properties"].get("stream_order", 0)
        orders[so] = orders.get(so, 0) + 1
    print(f"[tdx] stream orders: {dict(sorted(orders.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
