#!/usr/bin/env python3
"""Build the metro-year economic panel from BEA, BLS, and Census BPS.

This script preserves the logic of `02_economic_data_downloader_v6.ipynb`
while removing the hard-coded 5-metro / 14-metro county mappings. County
membership is resolved dynamically from the official Census CBSA delineation
file, so expanding the metro list only requires updating `metro_config.py`.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metro_config import ALL_YEARS, CBSA_CODES, METROS


RAW_DIR = REPO_ROOT / "data" / "raw"
BLS_DIR = RAW_DIR / "bls"
BPS_DIR = RAW_DIR / "bps"
CENSUS_DIR = RAW_DIR / "census"
OUT_PATH = REPO_ROOT / "data" / "economic" / "panel.csv"

DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2023/delineation-files/list1_2023.xlsx"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}
STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
PERMIT_CBSA_CODES = {
    metro: [cbsa] for metro, cbsa in CBSA_CODES.items()
}
PERMIT_CBSA_CODES["cleveland"] = ["17410", "17460"]


def ensure_delineation_file() -> Path:
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CENSUS_DIR / "list1_2023.xlsx"
    if out_path.exists():
        return out_path

    resp = requests.get(DELINEATION_URL, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def normalize_delineation_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        if "CBSA Code" in clean:
            renamed[col] = "cbsa"
        elif "FIPS State Code" in clean:
            renamed[col] = "state_fips"
        elif "FIPS County Code" in clean:
            renamed[col] = "county_fips"
        elif "Metropolitan/Micropolitan Statistical Area" in clean:
            renamed[col] = "area_type"
        elif "CBSA Title" in clean:
            renamed[col] = "cbsa_title"
        elif "County/County Equivalent" in clean:
            renamed[col] = "county_name"
        elif "State Name" in clean:
            renamed[col] = "state_name"
    df = df.rename(columns=renamed)
    required = {"cbsa", "state_fips", "county_fips"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected delineation columns: {sorted(missing)}")
    return df


def load_cbsa_counties() -> dict[str, list[tuple[str, str]]]:
    path = ensure_delineation_file()
    df = pd.read_excel(path, dtype=str, header=2)
    df = normalize_delineation_columns(df)
    df["cbsa"] = df["cbsa"].astype(str).str.strip().str.zfill(5)
    df["state_fips"] = df["state_fips"].astype(str).str.strip().str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(3)
    metro_map = {}
    for metro, cbsa in CBSA_CODES.items():
        sub = df[df["cbsa"] == cbsa][["state_fips", "county_fips"]].drop_duplicates()
        if sub.empty:
            raise ValueError(f"No counties found for metro={metro} cbsa={cbsa}")
        metro_map[metro] = list(sub.itertuples(index=False, name=None))
    return metro_map


def fetch_bea_gdp_bulk(years: list[int], metro_counties: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    url = "https://apps.bea.gov/regional/zip/CAGDP9.zip"
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    states_needed = {
        STATE_ABBR[state_fips]
        for counties in metro_counties.values()
        for state_fips, _county_fips in counties
        if state_fips in STATE_ABBR
    }
    state_dfs: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        available = archive.namelist()
        for state in sorted(states_needed):
            names = [
                name for name in available
                if f"CAGDP9_{state}_" in name and name.endswith(".csv")
            ]
            if not names:
                continue
            with archive.open(names[0]) as fh:
                df = pd.read_csv(fh, dtype=str, encoding="latin-1")
            df["GeoFIPS"] = (
                df["GeoFIPS"]
                .astype(str)
                .str.strip()
                .str.replace('"', "", regex=False)
                .str.strip()
            )
            df = df[df["LineCode"].astype(str).str.strip() == "1"].copy()
            state_dfs[state] = df

    year_cols = [str(year) for year in years]
    rows = []
    for metro, counties in metro_counties.items():
        county_fips = {sf + cf for sf, cf in counties}
        metro_frames = []
        for state_fips, _county_fips in counties:
            state = STATE_ABBR.get(state_fips)
            if state and state in state_dfs:
                metro_frames.append(state_dfs[state][state_dfs[state]["GeoFIPS"].isin(county_fips)])
        if not metro_frames:
            for year in years:
                rows.append({"metro": metro, "year": year, "gdp_millions": None})
            continue

        merged = pd.concat(metro_frames, ignore_index=True).drop_duplicates(subset=["GeoFIPS"])
        for year in years:
            col = str(year)
            if col not in merged.columns:
                value = None
            else:
                value = pd.to_numeric(merged[col], errors="coerce").sum(min_count=1)
                value = round(float(value) / 1000.0, 1) if pd.notna(value) else None
            rows.append({"metro": metro, "year": year, "gdp_millions": value})
    return pd.DataFrame(rows).sort_values(["metro", "year"]).reset_index(drop=True)


def load_bls_county_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath, skiprows=4, dtype=str, header=None)
    df = df.dropna(how="all").reset_index(drop=True)
    df = df[pd.to_numeric(df[1], errors="coerce").notna()]
    df["state_fips"] = df[1].astype(str).str.strip().str.zfill(2)
    df["county_fips"] = df[2].astype(str).str.strip().str.zfill(3)
    df["employed"] = pd.to_numeric(df[6].astype(str).str.replace(",", ""), errors="coerce")
    df["unemployed"] = pd.to_numeric(df[7].astype(str).str.replace(",", ""), errors="coerce")
    return df[["state_fips", "county_fips", "employed", "unemployed"]]


def fetch_bls_from_files(years: list[int], metro_counties: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    county_sets = {metro: set(counties) for metro, counties in metro_counties.items()}
    for year in years:
        path = BLS_DIR / f"laucnty{str(year)[2:]}.xlsx"
        if not path.exists():
            for metro in METROS:
                rows.append({
                    "metro": metro,
                    "year": year,
                    "unemployment_rate": None,
                    "employment_thousands": None,
                })
            continue

        df = load_bls_county_file(path)
        pairs = list(zip(df["state_fips"], df["county_fips"]))
        for metro, counties in county_sets.items():
            mask = [pair in counties for pair in pairs]
            sub = df.loc[mask]
            if sub.empty:
                rows.append({
                    "metro": metro,
                    "year": year,
                    "unemployment_rate": None,
                    "employment_thousands": None,
                })
                continue
            employed = sub["employed"].sum()
            unemployed = sub["unemployed"].sum()
            labor_force = employed + unemployed
            rows.append({
                "metro": metro,
                "year": year,
                "unemployment_rate": round(unemployed / labor_force * 100, 1) if labor_force > 0 else None,
                "employment_thousands": round(employed / 1000.0, 1),
            })
    return pd.DataFrame(rows).sort_values(["metro", "year"]).reset_index(drop=True)


def safe_int(value) -> int:
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return 0


def detect_bps_file(year: int) -> Path | None:
    for ext in ("txt", "TXT", "xlsx", "xls"):
        candidate = BPS_DIR / f"bps_{year}.{ext}"
        if candidate.exists():
            return candidate
    return None


def file_type(path: Path) -> str:
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"\xd0\xcf":
        return "xls_binary"
    if magic[:4] == b"PK\x03\x04":
        return "xlsx"
    return "txt"


def load_bps_from_files(years: list[int]) -> pd.DataFrame:
    cbsa_to_metro = {
        cbsa: metro
        for metro, cbsa_list in PERMIT_CBSA_CODES.items()
        for cbsa in cbsa_list
    }
    rows = []
    for year in years:
        path = detect_bps_file(year)
        if path is None:
            for metro in METROS:
                rows.append({"metro": metro, "year": year, "total_permits": None})
            continue

        try:
            kind = file_type(path)
            if kind in {"xls_binary", "xlsx"}:
                engine = "xlrd" if kind == "xls_binary" else "openpyxl"
                df = pd.read_excel(path, dtype=str, header=None, skiprows=6, engine=engine)
                df[1] = df[1].astype(str).str.strip().str.zfill(5)
                matched = df[df[1].isin(cbsa_to_metro)]
                for _, row in matched.iterrows():
                    total = safe_int(row[4]) + safe_int(row[5]) + safe_int(row[6]) + safe_int(row[7])
                    rows.append({
                        "metro": cbsa_to_metro[row[1]],
                        "year": year,
                        "total_permits": total,
                    })
            else:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[11:]
                merged_lines: list[str] = []
                carry = None
                for raw in lines:
                    if not raw.strip():
                        continue
                    line = raw.rstrip("\n")
                    stripped = line.lstrip()
                    tokens = stripped.split()
                    is_new_record = (
                        line == stripped
                        and len(tokens) >= 2
                        and tokens[0].isdigit()
                        and tokens[1].isdigit()
                    )
                    if is_new_record:
                        if carry is not None:
                            merged_lines.append(carry)
                        carry = stripped
                    elif carry is not None:
                        carry += " " + stripped
                if carry is not None:
                    merged_lines.append(carry)

                for line in merged_lines:
                    tokens = line.split()
                    if len(tokens) < 8:
                        continue
                    cbsa = tokens[1].zfill(5)
                    if cbsa not in cbsa_to_metro:
                        continue
                    total = sum(safe_int(tok) for tok in tokens[-5:-1])
                    rows.append({
                        "metro": cbsa_to_metro[cbsa],
                        "year": year,
                        "total_permits": total,
                    })
        except Exception:
            for metro in METROS:
                rows.append({"metro": metro, "year": year, "total_permits": None})

    result = pd.DataFrame(rows).sort_values(["metro", "year"]).reset_index(drop=True)
    # Ensure one row per metro-year even if a parser returned nothing for a year.
    full_index = pd.MultiIndex.from_product([METROS, years], names=["metro", "year"])
    return (
        result.set_index(["metro", "year"])
        .reindex(full_index)
        .reset_index()
    )


def interpolate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    filled = (
        panel
        .groupby("metro", group_keys=False)
        .apply(lambda g: g.set_index("year").interpolate(method="index").reset_index())
        .reset_index(drop=True)
    )
    filled["interpolated"] = panel.isnull().any(axis=1).values
    return filled


def main() -> None:
    years = list(ALL_YEARS)
    metro_counties = load_cbsa_counties()
    gdp_df = fetch_bea_gdp_bulk(years, metro_counties)
    bls_df = fetch_bls_from_files(years, metro_counties)
    permits_df = load_bps_from_files(years)

    panel = (
        gdp_df
        .merge(bls_df, on=["metro", "year"], how="outer")
        .merge(permits_df, on=["metro", "year"], how="outer")
        .sort_values(["metro", "year"])
        .reset_index(drop=True)
    )
    panel = panel[
        [
            "year",
            "metro",
            "gdp_millions",
            "unemployment_rate",
            "employment_thousands",
            "total_permits",
        ]
    ]

    panel_filled = interpolate_panel(panel)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel_filled.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")
    print(f"Rows: {len(panel_filled)}")
    print(f"Metros: {len(panel_filled['metro'].unique())}")
    print(f"Years: {sorted(panel_filled['year'].unique().tolist())}")
    print(f"Interpolated rows: {int(panel_filled['interpolated'].sum())}")


if __name__ == "__main__":
    main()
