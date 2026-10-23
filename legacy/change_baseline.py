"""Dot-product change baseline on AlphaEarth satellite embeddings.

Picks a random mangrove area with high change variance, computes
change = 1 - dot(embedding_year1, embedding_year2) per 10 m pixel, and writes
outputs you can compare against Google Earth's "Detect change" layer.

Usage:
    conda activate gee
    python change_baseline.py --project YOUR_GCP_PROJECT [--seed 7] [--year1 2017 --year2 2024]
"""

import argparse
import json
import os
import urllib.request

import ee

EMBEDDINGS = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
MANGROVES = "LANDSAT/MANGROVE_FORESTS"  # Giri et al. 2000 global mangrove extent, 30 m
S2 = "COPERNICUS/S2_HARMONIZED"  # L1C, available from 2015 (SR has gaps before ~2019)
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

# Major mangrove regions as [west, south, east, north].
MANGROVE_REGIONS = [
    [88.0, 21.4, 89.9, 22.6],      # Sundarbans, India/Bangladesh
    [94.2, 15.7, 95.6, 16.6],      # Ayeyarwady Delta, Myanmar
    [104.6, 8.5, 105.4, 9.4],      # Ca Mau, Vietnam
    [117.0, -1.2, 117.7, -0.3],    # Mahakam Delta, Indonesia
    [136.5, -5.0, 139.0, -4.0],    # Southern Papua, Indonesia
    [5.3, 4.2, 7.5, 5.0],          # Niger Delta, Nigeria
    [-47.5, -1.2, -44.5, -0.5],    # Pará coast, Brazil
    [35.8, -18.9, 37.3, -17.6],    # Zambezi Delta, Mozambique
    [-87.6, 12.9, -87.2, 13.3],    # Gulf of Fonseca, Honduras
    [-81.3, 25.1, -80.8, 25.6],    # Everglades, USA
]

BOX_HALF_M = 2500  # 5 km x 5 km area of interest
OUT_DIR = "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                   help="Google Cloud project registered for Earth Engine")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--candidates", type=int, default=40,
                   help="number of random mangrove locations to score")
    p.add_argument("--year1", type=int)
    p.add_argument("--year2", type=int)
    return p.parse_args()


def available_years():
    years = (ee.ImageCollection(EMBEDDINGS)
             .aggregate_array("system:time_start")
             .map(lambda t: ee.Date(t).get("year"))
             .distinct().sort())
    return years.getInfo()


def embedding(year):
    return (ee.ImageCollection(EMBEDDINGS)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic())


def change_image(year1, year2):
    # Embeddings are unit length, so the dot product is cosine similarity.
    dot = embedding(year1).multiply(embedding(year2)).reduce(ee.Reducer.sum())
    return ee.Image(1).subtract(dot).rename("change")


def pick_area(change, seed, n):
    """Sample random mangrove pixels in major mangrove regions, return the 5 km box with the highest change std dev."""
    mangrove = ee.Image(ee.ImageCollection(MANGROVES).first()).select(["1"], ["mangrove"]).selfMask().toInt()
    rings = [[[[w, s], [e, s], [e, n], [w, n], [w, s]]] for w, s, e, n in MANGROVE_REGIONS]
    regions = ee.Geometry.MultiPolygon(rings, None, False)
    points = mangrove.stratifiedSample(numPoints=n, classBand="mangrove", region=regions,
                                       scale=250, seed=seed, geometries=True)

    def score(pt):
        box = pt.geometry().buffer(BOX_HALF_M).bounds()
        stats = change.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=box, scale=30, maxPixels=1e8, bestEffort=True)
        return ee.Feature(box, stats)

    scored = points.map(score).filter(ee.Filter.notNull(["change_stdDev"]))
    best = ee.Feature(scored.sort("change_stdDev", False).first())
    return best.getInfo(), scored.size().getInfo()


def s2_rgb(year, region):
    cs = ee.ImageCollection(CLOUD_SCORE)
    col = (ee.ImageCollection(S2)
           .filterBounds(region)
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .linkCollection(cs, ["cs_cdf"])
           .map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.6))))
    return col.median().select(["B4", "B3", "B2"])


def save_thumb(image, region, vis, path):
    url = image.getThumbURL({**vis, "region": region, "dimensions": 768, "format": "png"})
    urllib.request.urlretrieve(url, path)


def write_kml(coords, path):
    ring = " ".join(f"{x},{y},0" for x, y in coords)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>
<name>change_baseline AOI</name>
<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark></Document></kml>
"""
    with open(path, "w") as f:
        f.write(kml)


def main():
    args = parse_args()
    if not args.project:
        raise SystemExit("Pass --project or set EE_PROJECT to your Earth Engine Cloud project.")
    ee.Initialize(project=args.project)
    os.makedirs(OUT_DIR, exist_ok=True)

    years = available_years()
    year1 = args.year1 or years[0]
    year2 = args.year2 or years[-1]
    print(f"Embedding years available: {years}; comparing {year1} -> {year2}")

    change = change_image(year1, year2)

    print(f"Scoring {args.candidates} random mangrove locations (seed {args.seed})...")
    best, n_scored = pick_area(change, args.seed, args.candidates)
    coords = best["geometry"]["coordinates"][0]
    region = ee.Geometry.Polygon(coords)
    lon, lat = region.centroid(1).coordinates().getInfo()
    props = best["properties"]
    print(f"Picked area centred at {lat:.4f}, {lon:.4f} out of {n_scored} scored "
          f"(mean change {props['change_mean']:.3f}, std {props['change_stdDev']:.3f})")

    # Percentiles at full 10 m resolution for the chosen area.
    pct = change.reduceRegion(ee.Reducer.percentile([50, 90, 99]), region,
                              scale=10, maxPixels=1e8).getInfo()

    change_vis = {"min": 0, "max": 0.6,
                  "palette": ["000004", "3b0f70", "8c2981", "de4968", "fe9f6d", "fcfdbf"]}
    rgb_vis = {"min": 0, "max": 3000}
    save_thumb(change.clip(region), region, change_vis, f"{OUT_DIR}/change_{year1}_{year2}.png")
    save_thumb(s2_rgb(year1, region), region, rgb_vis, f"{OUT_DIR}/rgb_{year1}.png")
    save_thumb(s2_rgb(year2, region), region, rgb_vis, f"{OUT_DIR}/rgb_{year2}.png")

    import geemap
    geemap.ee_export_image(change.clip(region), f"{OUT_DIR}/change_{year1}_{year2}.tif",
                           scale=10, region=region, file_per_band=False)

    write_kml(coords, f"{OUT_DIR}/aoi.kml")
    summary = {"year1": year1, "year2": year2, "centre_lat": lat, "centre_lon": lon,
               "seed": args.seed, "aoi_polygon": coords,
               "change_mean": props["change_mean"], "change_std": props["change_stdDev"],
               "change_percentiles": pct}
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote outputs to {OUT_DIR}/")
    print(f"Google Maps link: https://www.google.com/maps/@{lat},{lon},14z/data=!3m1!1e3")


if __name__ == "__main__":
    main()
