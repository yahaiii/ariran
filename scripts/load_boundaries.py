"""
Load Nigerian administrative boundaries into public.admin_boundaries.

Supports:
  - GeoJSON (.geojson, .json): OGR_GEOMETRY_AS_GEOJSON environment variable not needed.
  - Shapefiles (.shp): Requires geopandas; auto-detected by file extension.

Expected GeoJSON properties (at least one of the following must be present):
  - level/admin_level: 0–3 (0=country, 1=state, 2=LGA, 3=ward)
  - name: Name of the boundary entity
  - state_name: Name of containing state (for LGA level)
  - lga_name: Name of LGA (for ward level)
  - pcode: Unique place code (optional, for reconciliation)

Usage:
  python scripts/load_boundaries.py --source boundaries.geojson
  python scripts/load_boundaries.py --source boundaries.shp --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import text

from db.connection import get_session

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Nigerian administrative boundaries from GeoJSON or Shapefile"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to boundary dataset (GeoJSON or Shapefile)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if boundaries already loaded (check count)",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Filter: only load boundaries at this admin level (0–3)",
    )
    return parser.parse_args()


def _load_geojson(source_path: Path) -> list[dict]:
    """Load features from GeoJSON file."""
    logger.info(f"Loading GeoJSON: {source_path}")

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    else:
        features = [data] if data.get("type") == "Feature" else []

    logger.info(f"Parsed {len(features)} features from {source_path.name}")
    return features


def _load_shapefile(source_path: Path) -> list[dict]:
    """Load features from Shapefile using geopandas."""
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for Shapefile support. "
            "Install: pip install geopandas"
        )

    logger.info(f"Loading Shapefile: {source_path}")
    gdf = gpd.read_file(source_path)
    logger.info(f"Parsed {len(gdf)} features from {source_path.name}")

    features = []
    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": json.loads(row.geometry.to_geojson()),
            "properties": row.drop("geometry").to_dict(),
        }
        features.append(feature)

    return features


def _infer_admin_level(properties: dict) -> int | None:
    """Infer admin level from properties."""
    # Try explicit level fields
    for field in ("admin_level", "level", "admin_lv"):
        if field in properties:
            try:
                return int(properties[field])
            except (ValueError, TypeError):
                pass

    # Infer from presence of lga_name / state_name / etc.
    if "lga_name" in properties or "lga" in properties:
        return 2
    if "state_name" in properties or "state" in properties:
        return 1

    # Infer from name patterns (heuristic)
    name = str(properties.get("name", "")).lower()
    if "lga" in name or "local government" in name:
        return 2
    if any(state in name for state in ["state", "federal capital"]):
        return 1

    return None


def _build_boundary_row(feature: dict, filter_level: int | None = None) -> dict | None:
    """Extract and validate boundary record from GeoJSON feature."""
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})

    if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
        logger.warning("Skipping feature with invalid geometry type")
        return None

    # Infer/extract admin level
    admin_level = _infer_admin_level(props)
    if admin_level is None:
        logger.warning(f"Cannot infer admin level from {props.get('name')}")
        return None

    if filter_level is not None and admin_level != filter_level:
        return None

    # Extract core fields
    name = props.get("name") or props.get("NAME")
    state_name = props.get("state_name") or props.get("STATE")
    lga_name = props.get("lga_name") or props.get("LGA")
    pcode = props.get("pcode") or props.get("PCODE")

    if not name:
        logger.warning("Skipping boundary with no name")
        return None

    # Convert geometry to WKT
    geometry_wkt = _geojson_to_wkt(geom)
    if not geometry_wkt:
        logger.warning(f"Cannot convert geometry for {name}")
        return None

    return {
        "admin_level": admin_level,
        "name": str(name),
        "state_name": state_name,
        "lga_name": lga_name,
        "pcode": pcode,
        "geom_wkt": geometry_wkt,
    }


def _geojson_to_wkt(geojson_geom: dict) -> str | None:
    """Convert GeoJSON geometry to WKT."""
    geom_type = geojson_geom.get("type")
    coords = geojson_geom.get("coordinates", [])

    if geom_type == "Polygon":
        return _polygon_to_wkt(coords)
    elif geom_type == "MultiPolygon":
        return _multipolygon_to_wkt(coords)

    return None


def _polygon_to_wkt(coords: list) -> str | None:
    """Convert Polygon coordinates to WKT."""
    if not coords:
        return None
    rings = []
    for ring in coords:
        points = ", ".join([f"{lon} {lat}" for lon, lat in ring])
        rings.append(f"({points})")
    return f"POLYGON({', '.join(rings)})"


def _multipolygon_to_wkt(coords: list) -> str | None:
    """Convert MultiPolygon coordinates to WKT."""
    if not coords:
        return None
    polygons = []
    for poly_coords in coords:
        rings = []
        for ring in poly_coords:
            points = ", ".join([f"{lon} {lat}" for lon, lat in ring])
            rings.append(f"({points})")
        polygons.append(f"({', '.join(rings)})")
    return f"MULTIPOLYGON({', '.join(polygons)})"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        logger.error(f"Boundary source not found: {source_path}")
        return 1

    # Load features from file
    if source_path.suffix.lower() == ".shp":
        features = _load_shapefile(source_path)
    elif source_path.suffix.lower() in (".geojson", ".json"):
        features = _load_geojson(source_path)
    else:
        logger.error(
            f"Unsupported file type: {source_path.suffix}. "
            "Expected .geojson, .json, or .shp"
        )
        return 1

    if not features:
        logger.error("No features found in source file")
        return 1

    # Extract and filter boundaries
    boundaries = []
    for feature in features:
        row = _build_boundary_row(feature, args.level)
        if row:
            boundaries.append(row)

    logger.info(f"Extracted {len(boundaries)} valid boundaries")

    if not boundaries:
        logger.error("No valid boundaries after filtering")
        return 1

    # Load into database
    try:
        with get_session() as session:
            # Check if already loaded
            existing_count = session.execute(
                text("SELECT COUNT(*) FROM public.admin_boundaries")
            ).scalar()

            if existing_count > 0 and args.skip_existing:
                logger.info(
                    f"Database already has {existing_count} boundaries. "
                    "Skipping load (--skip-existing)"
                )
                return 0

            # Upsert boundaries (insert or update by pcode, or insert by name)
            inserted = 0
            updated = 0

            for row in boundaries:
                # Try to find existing record by pcode or (name, admin_level)
                existing = None
                if row["pcode"]:
                    existing = session.execute(
                        text(
                            "SELECT id FROM public.admin_boundaries WHERE pcode = :pcode"
                        ),
                        {"pcode": row["pcode"]},
                    ).scalar()

                if not existing:
                    existing = session.execute(
                        text(
                            "SELECT id FROM public.admin_boundaries "
                            "WHERE name = :name AND admin_level = :level"
                        ),
                        {"name": row["name"], "level": row["admin_level"]},
                    ).scalar()

                if existing:
                    # Update existing
                    session.execute(
                        text(
                            "UPDATE public.admin_boundaries "
                            "SET state_name = :state, lga_name = :lga, "
                            "    geom = ST_GeomFromText(:geom, 4326) "
                            "WHERE id = :id"
                        ),
                        {
                            "state": row["state_name"],
                            "lga": row["lga_name"],
                            "geom": row["geom_wkt"],
                            "id": existing,
                        },
                    )
                    updated += 1
                else:
                    # Insert new
                    session.execute(
                        text(
                            "INSERT INTO public.admin_boundaries "
                            "(admin_level, name, state_name, lga_name, pcode, geom) "
                            "VALUES (:level, :name, :state, :lga, :pcode, "
                            "        ST_GeomFromText(:geom, 4326))"
                        ),
                        {
                            "level": row["admin_level"],
                            "name": row["name"],
                            "state": row["state_name"],
                            "lga": row["lga_name"],
                            "pcode": row["pcode"],
                            "geom": row["geom_wkt"],
                        },
                    )
                    inserted += 1

            session.commit()
            logger.info(
                f"Load complete: {inserted} inserted, {updated} updated "
                f"(total {inserted + updated} records)"
            )
            return 0

    except Exception as e:
        logger.error(f"Database load failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

