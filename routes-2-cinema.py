import os
import re
from collections import defaultdict

import pandas as pd
import requests
from dotenv import load_dotenv

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
FIELD_MASK = (
    "routes.legs.steps.distanceMeters,routes.legs.steps.travelMode,"
    "routes.legs.steps.transitDetails.transitLine.vehicle.type"
)


def clean_postcode(value):
    return re.sub(r"\s+", "", str(value)).upper() + ", UK"


def admin_level_2(address, key):
    response = requests.get(
        GEOCODE_URL, params={"address": address, "key": key}, timeout=30
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return ""
    for component in results[0].get("address_components", []):
        if "administrative_area_level_2" in component.get("types", []):
            return component.get("long_name", "").upper()
    return ""


def main():
    load_dotenv()
    key = os.getenv("KEY")
    if not key:
        raise ValueError("Set KEY in .env")

    locations = pd.read_csv("locations.csv", dtype=str).fillna("")
    lookup = pd.read_csv("lookup.csv", dtype=str).fillna("")
    locations["Destination1"] = locations["Destination1"].map(clean_postcode)
    locations["Destination2"] = locations["Destination2"].map(clean_postcode)
    locations["Travel-method"] = locations["Travel-method"].str.strip().str.upper()
    lookup = lookup.apply(lambda column: column.str.strip().str.upper())
    lookup = lookup.sort_values("Region", key=lambda s: s.ne(""), kind="stable")
    by_method = lookup.drop_duplicates("Travel-method").set_index("Travel-method")

    routes = []
    for _, item in locations.iterrows():
        method = item["Travel-method"]
        allowed = by_method.loc[method, "Google-allowed"] if method in by_method.index else ""
        mode = allowed if allowed in {"DRIVE", "BICYCLE", "WALK", "TWO_WHEELER"} else "TRANSIT"
        body = {
            "origin": {"address": item["Destination1"]},
            "destination": {"address": item["Destination2"]},
            "travelMode": mode,
        }
        if mode == "TRANSIT" and allowed:
            body["transitPreferences"] = {"allowedTravelModes": [allowed]}
        response = requests.post(
            ROUTES_URL,
            headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELD_MASK,
                     "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json().get("routes", [{}])[0]
        row = {
            "ID": item.get("ID", ""),
            "Destination1": item["Destination1"],
            "Destination2": item["Destination2"],
            "Admin2_Destination1": admin_level_2(item["Destination1"], key),
            "Admin2_Destination2": admin_level_2(item["Destination2"], key),
            "Travel-method": method,
        }
        step_index = 0
        for leg in result.get("legs", []):
            for step in leg.get("steps", []):
                step_index += 1
                step_method = step.get("travelMode", "").upper()
                if step_method == "TRANSIT":
                    transit_vehicle = (
                        step.get("transitDetails", {})
                        .get("transitLine", {})
                        .get("vehicle", {})
                        .get("type", "")
                    )
                    row[f"Leg{step_index}_method"] = transit_vehicle.upper()
                else:
                    row[f"Leg{step_index}_method"] = step_method
                row[f"Leg{step_index}_distance_km"] = step.get("distanceMeters", 0) / 1000
        routes.append(row)

    route_df = pd.DataFrame(routes).astype(object).fillna("")
    route_df.to_csv("routes.csv", index=True)

    method_cols = [c for c in route_df if c.startswith("Leg") and c.endswith("_method")]
    aggregate = []
    for _, route in route_df.iterrows():
        distances = defaultdict(float)
        for col in method_cols:
            number = col[3:-7]
            method = str(route.get(col, "")).upper()
            if method:
                distances[method] += float(route.get(f"Leg{number}_distance_km", 0) or 0)
        london = (route["Admin2_Destination1"] == "GREATER LONDON"
                  and route["Admin2_Destination2"] == "GREATER LONDON")
        row = {c: route[c] for c in ["ID", "Destination1", "Destination2",
              "Admin2_Destination1", "Admin2_Destination2", "Travel-method"]}
        errors = []
        total = 0.0
        for method, distance in distances.items():
            row[f"{method}_distance_km"] = distance
            candidates = (lookup[lookup["Travel-method"].eq(route["Travel-method"].upper())]
                          if method == "DRIVE"
                          else lookup[lookup["Google-output"].eq(method)])
            if london:
                local = candidates[candidates["Region"].eq("GREATER LONDON")]
                if not local.empty:
                    candidates = local
            factor = pd.to_numeric(candidates.iloc[0]["Carbon-per-km"], errors="coerce") if not candidates.empty else 0
            if pd.isna(factor):
                factor = 0
                errors.append(method)
            elif candidates.empty:
                errors.append(method)
            impact = distance * factor
            row[f"{method}_carbon"] = impact
            total += impact
        row["Total_carbon"] = total
        row["Error"] = "; ".join(sorted(set(errors)))
        aggregate.append(row)
    aggregate_df = pd.DataFrame(aggregate).fillna(0)
    identifiers = [
        "ID", "Destination1", "Destination2", "Admin2_Destination1",
        "Admin2_Destination2", "Travel-method",
    ]
    summary = ["Total_carbon", "Error"]
    aggregate_fields = sorted(
        column for column in aggregate_df.columns
        if column not in identifiers + summary
    )
    aggregate_df = aggregate_df[
        [column for column in identifiers if column in aggregate_df]
        + [column for column in summary if column in aggregate_df]
        + aggregate_fields
    ]
    aggregate_df.to_csv("aggregate.csv", index=True)


if __name__ == "__main__":
    main()