"""
Pure helpers for the X-Ray exposure page.

The app stores a mix of true sectors, ETF themes, regions, bond issuer types,
and commodity labels in the same `sector` field. These helpers normalize that
raw data into buckets that are more useful for portfolio exposure analysis.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


COUNTRY_ISO3 = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Brazil": "BRA",
    "China": "CHN",
    "France": "FRA",
    "Germany": "DEU",
    "Italy": "ITA",
    "Netherlands": "NLD",
    "Romania": "ROU",
    "United Kingdom": "GBR",
    "United States": "USA",
}

EUROPEAN_COUNTRIES = {
    "austria",
    "france",
    "germany",
    "italy",
    "netherlands",
    "romania",
    "united kingdom",
}

BROAD_EQUITY_LABELS = {
    "emergenti",
    "emerging markets",
    "europe broad",
    "europa",
    "european equity",
    "large cap usa",
    "small cap usa",
    "us equity",
    "us large cap",
    "us small cap",
}

EQUITY_SECTOR_NORMALIZATION = {
    "business services": "Industrials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "financials": "Financials",
    "gold miners": "Materials / Gold Miners",
    "healthcare": "Healthcare",
    "industrials": "Industrials",
    "materials": "Materials",
    "real estate": "Real Estate",
    "technology": "Technology",
    "us tech": "Technology",
    "utilities": "Utilities",
}


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def infer_xray_sector(row: Mapping) -> tuple[str, str]:
    """Return a portfolio exposure bucket and the attribution method used."""
    macro = _text(row.get("macro_class"))
    sector = _text(row.get("sector"))
    industry = _text(row.get("industry"))
    country = _text(row.get("country"))
    asset_type = _text(row.get("asset_type"))
    asset_sub_type = _text(row.get("asset_sub_type"))
    name = _text(row.get("name"))

    macro_l = macro.lower()
    sector_l = sector.lower()
    industry_l = industry.lower()
    country_l = country.lower()
    combined = " ".join([sector_l, industry_l, asset_type.lower(), asset_sub_type.lower(), name.lower()])

    if macro_l == "fixed income":
        if _contains_any(combined, ("corporate", "corp bond")):
            return "Fixed Income - Corporate", "Issuer type"

        if _contains_any(combined, ("government", "gov.", "btp", "bobl", "treasury", "tips", "bond")):
            if country_l == "italy":
                return "Fixed Income - Gov. Italia", "Issuer country"
            if country_l in EUROPEAN_COUNTRIES or country_l == "europe":
                return "Fixed Income - Gov. Europa", "Issuer country"
            return "Fixed Income - Gov. Internazionali", "Issuer country"

        return "Fixed Income - Altro", "Macro class fallback"

    if macro_l == "alternative":
        if _contains_any(combined, ("crypto", "bitcoin", "ethereum", "btc", "eth")):
            return "Alternative - Crypto", "Alternative subtype"
        if _contains_any(combined, ("uranio", "uranium", "nucleare", "nuclear")):
            return "Alternative - Uranio/Nucleare", "Commodity exposure"
        if _contains_any(combined, ("natural gas", "energia", "energy", "gas")):
            return "Alternative - Energia", "Commodity exposure"
        if _contains_any(combined, ("copper", "industrial metals")):
            return "Alternative - Metalli industriali", "Commodity exposure"
        if _contains_any(combined, ("gold", "silver", "oro", "argento", "precious")):
            return "Alternative - Metalli preziosi", "Commodity exposure"
        if "commod" in combined:
            return "Alternative - Commodities", "Commodity exposure"
        return "Alternative - Altro", "Macro class fallback"

    if macro_l == "equity":
        if sector_l in BROAD_EQUITY_LABELS or industry_l in BROAD_EQUITY_LABELS:
            return "Equity - Indici multi-settore", "Broad ETF"

        normalized = EQUITY_SECTOR_NORMALIZATION.get(sector_l)
        if normalized:
            return normalized, "Equity sector"

        if sector:
            return sector, "Raw sector"

        return "Equity - Non classificato", "Missing sector"

    if sector:
        return sector, "Raw sector"

    return "Non classificato", "Missing sector"


def add_xray_sector(positions: pd.DataFrame) -> pd.DataFrame:
    """Add normalized X-Ray sector columns to a positions DataFrame."""
    df = positions.copy()
    if df.empty:
        df["xray_sector"] = pd.Series(dtype="object")
        df["xray_sector_method"] = pd.Series(dtype="object")
        return df

    inferred = df.apply(infer_xray_sector, axis=1)
    df["xray_sector"] = inferred.apply(lambda item: item[0])
    df["xray_sector_method"] = inferred.apply(lambda item: item[1])
    return df


def build_exposure_table(
    positions: pd.DataFrame,
    group_col: str,
    total_value: float | None = None,
    names_limit: int = 5,
) -> pd.DataFrame:
    """Aggregate exposure by a column with value, count, weight, and sample names."""
    if positions.empty or group_col not in positions.columns:
        return pd.DataFrame(columns=[group_col, "value", "count", "weight", "names"])

    df = positions.copy()
    df[group_col] = df[group_col].apply(lambda value: _text(value) or "Non classificato")
    df["current_value"] = pd.to_numeric(df["current_value"], errors="coerce").fillna(0.0)

    grouped = df.groupby(group_col, dropna=False).agg(
        value=("current_value", "sum"),
        count=("isin", "count"),
        names=("name", lambda values: ", ".join(values.astype(str).head(names_limit))),
    ).reset_index()

    denominator = float(total_value) if total_value is not None and total_value > 0 else float(grouped["value"].sum())
    grouped["weight"] = (grouped["value"] / denominator * 100).round(2) if denominator > 0 else 0.0
    return grouped.sort_values("weight", ascending=False).reset_index(drop=True)


def build_country_exposure(
    positions: pd.DataFrame,
    total_value: float | None = None,
) -> pd.DataFrame:
    """Aggregate country exposure and mark which rows can be drawn on a country map."""
    country_data = build_exposure_table(positions, "country", total_value=total_value)
    if country_data.empty:
        country_data["iso3"] = pd.Series(dtype="object")
        country_data["is_mappable"] = pd.Series(dtype="bool")
        return country_data

    country_data["iso3"] = country_data["country"].map(COUNTRY_ISO3)
    country_data["is_mappable"] = country_data["iso3"].notna()
    return country_data
