"""
Level 1 benchmark look-through for the Vanguard LifeStrategy benchmarks.

The fund's official benchmark VNGA50/50 is 50% VNGA60 + 50% VNGA40, so its
underlying ETFs are the two funds' building blocks blended 50/50.

The live path fetches each LifeStrategy fund's underlying ETF list from
StockAnalysis. If the public page is unavailable, the app falls back to the
latest known underlying weights so the dashboard remains usable.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
import requests


LIFESTRATEGY_HOLDINGS_URL = "https://stockanalysis.com/quote/bit/{ticker}/holdings/"
VNGA60_HOLDINGS_URL = LIFESTRATEGY_HOLDINGS_URL.format(ticker="VNGA60")
VNGA40_HOLDINGS_URL = LIFESTRATEGY_HOLDINGS_URL.format(ticker="VNGA40")


VNGA60_FALLBACK_HOLDINGS = [
    {"symbol": "LON: VHVG", "name": "Vanguard FTSE Developed World UCITS ETF", "weight_pct": 19.41},
    {"symbol": "LON: VWRA", "name": "Vanguard FTSE All-World UCITS ETF", "weight_pct": 19.29},
    {"symbol": "ETR: VAGF", "name": "Vanguard Global Aggregate Bond UCITS ETF", "weight_pct": 19.11},
    {"symbol": "LON: VNRG", "name": "Vanguard FTSE North America UCITS ETF", "weight_pct": 12.65},
    {"symbol": "ETR: VDTE", "name": "Vanguard USD Treasury Bond UCITS ETF", "weight_pct": 7.57},
    {"symbol": "ETR: VGEA", "name": "Vanguard EUR Eurozone Government Bond UCITS ETF", "weight_pct": 5.08},
    {"symbol": "ETR: VDCE", "name": "Vanguard USD Corporate Bond UCITS ETF", "weight_pct": 4.92},
    {"symbol": "ETR: VFEA", "name": "Vanguard FTSE Emerging Markets UCITS ETF", "weight_pct": 4.16},
    {"symbol": "LON: VEUA", "name": "Vanguard FTSE Developed Europe UCITS ETF", "weight_pct": 2.79},
    {"symbol": "ETR: VECA", "name": "Vanguard EUR Corporate Bond UCITS ETF", "weight_pct": 1.83},
    {"symbol": "LON: VJPA", "name": "Vanguard FTSE Japan UCITS ETF", "weight_pct": 1.19},
    {"symbol": "ETR: VGEK", "name": "Vanguard FTSE Developed Asia Pacific ex Japan UCITS ETF", "weight_pct": 1.07},
    {"symbol": "ETR: VGUE", "name": "Vanguard U.K. Gilt UCITS ETF", "weight_pct": 0.92},
]


VNGA40_FALLBACK_HOLDINGS = [
    {"symbol": "LON: VWRA", "name": "Vanguard FTSE All-World UCITS ETF", "weight_pct": 19.41},
    {"symbol": "ETR: VAGF", "name": "Vanguard Global Aggregate Bond UCITS ETF", "weight_pct": 19.18},
    {"symbol": "LON: VHVG", "name": "Vanguard FTSE Developed World UCITS ETF", "weight_pct": 18.83},
    {"symbol": "ETR: VDTE", "name": "Vanguard USD Treasury Bond UCITS ETF", "weight_pct": 15.07},
    {"symbol": "ETR: VGEA", "name": "Vanguard EUR Eurozone Government Bond UCITS ETF", "weight_pct": 10.15},
    {"symbol": "ETR: VDCE", "name": "Vanguard USD Corporate Bond UCITS ETF", "weight_pct": 9.71},
    {"symbol": "ETR: VECA", "name": "Vanguard EUR Corporate Bond UCITS ETF", "weight_pct": 3.59},
    {"symbol": "ETR: VFEA", "name": "Vanguard FTSE Emerging Markets UCITS ETF", "weight_pct": 2.16},
    {"symbol": "ETR: VGUE", "name": "Vanguard U.K. Gilt UCITS ETF", "weight_pct": 1.90},
]


LIFESTRATEGY_FALLBACK_HOLDINGS = {
    "VNGA60": VNGA60_FALLBACK_HOLDINGS,
    "VNGA40": VNGA40_FALLBACK_HOLDINGS,
}

# Benchmark composti da più fondi LifeStrategy: fondo -> peso.
BENCHMARK_BLENDS = {
    "VNGA50/50": {"VNGA60": 0.5, "VNGA40": 0.5},
}


VNGA60_METADATA = {
    "VHVG": {"macro_class": "Equity", "region": "Developed World", "currency": "USD", "segment": "Developed markets equity"},
    "VWRA": {"macro_class": "Equity", "region": "Global", "currency": "USD", "segment": "All-world equity"},
    "VAGF": {"macro_class": "Fixed Income", "region": "Global", "currency": "EUR Hedged", "segment": "Global aggregate bond"},
    "VNRG": {"macro_class": "Equity", "region": "North America", "currency": "USD", "segment": "North America equity"},
    "VDTE": {"macro_class": "Fixed Income", "region": "North America", "currency": "EUR Hedged", "segment": "US Treasury"},
    "VGEA": {"macro_class": "Fixed Income", "region": "Europe", "currency": "EUR", "segment": "Eurozone government bond"},
    "VDCE": {"macro_class": "Fixed Income", "region": "North America", "currency": "EUR Hedged", "segment": "USD corporate bond"},
    "VFEA": {"macro_class": "Equity", "region": "Emerging Markets", "currency": "USD", "segment": "Emerging markets equity"},
    "VEUA": {"macro_class": "Equity", "region": "Europe", "currency": "EUR", "segment": "Developed Europe equity"},
    "VECA": {"macro_class": "Fixed Income", "region": "Europe", "currency": "EUR", "segment": "EUR corporate bond"},
    "VJPA": {"macro_class": "Equity", "region": "Japan", "currency": "USD", "segment": "Japan equity"},
    "VGUE": {"macro_class": "Fixed Income", "region": "United Kingdom", "currency": "EUR Hedged", "segment": "UK gilt"},
    "VGEK": {"macro_class": "Equity", "region": "Asia Pacific ex Japan", "currency": "USD", "segment": "Developed Asia Pacific equity"},
}


def _clean_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.strip().upper()


def _parse_weight(value) -> float:
    text = str(value or "").replace("%", "").replace(",", "").strip()
    return float(text) if text else 0.0


def parse_stockanalysis_holdings(html: str) -> pd.DataFrame:
    """Parse the public StockAnalysis holdings table."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table"):
        headers = [cell.get_text(" ", strip=True).lower() for cell in table.find_all("th")]
        # StockAnalysis labels the column "Weight" or "% Weight" depending on the page version.
        headers = ["weight" if "weight" in h else h for h in headers]
        if not {"symbol", "name", "weight"}.issubset(set(headers)):
            continue
        symbol_idx = headers.index("symbol")
        name_idx = headers.index("name")
        weight_idx = headers.index("weight")
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
            if len(cells) <= max(symbol_idx, name_idx, weight_idx):
                continue
            rows.append({
                "symbol": cells[symbol_idx],
                "name": cells[name_idx],
                "weight": cells[weight_idx],
            })
        break

    if not rows:
        raise ValueError("No holdings table found")

    result = pd.DataFrame(rows)
    result["weight_pct"] = result["weight"].apply(_parse_weight)
    result = result[result["weight_pct"] > 0]
    return result[["symbol", "name", "weight_pct"]].reset_index(drop=True)


def _infer_benchmark_metadata(symbol: str, name: str) -> dict:
    clean = _clean_symbol(symbol)
    if clean in VNGA60_METADATA:
        return VNGA60_METADATA[clean]

    text = str(name or "").lower()
    if any(token in text for token in ("bond", "treasury", "gilt", "corporate")):
        macro = "Fixed Income"
    else:
        macro = "Equity"

    if "north america" in text or "usd" in text or "treasury" in text:
        region = "North America"
    elif "emerging" in text:
        region = "Emerging Markets"
    elif "europe" in text or "eurozone" in text:
        region = "Europe"
    elif "japan" in text:
        region = "Japan"
    elif "asia pacific" in text:
        region = "Asia Pacific ex Japan"
    else:
        region = "Global"

    return {"macro_class": macro, "region": region, "currency": "", "segment": name}


def enrich_benchmark_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Add Level 1 metadata to benchmark underlying ETFs."""
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["symbol", "ticker", "name", "weight_pct", "macro_class", "region", "currency", "segment"])

    df = holdings.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["ticker"] = df["symbol"].apply(_clean_symbol)
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.0)

    meta_rows = df.apply(lambda row: _infer_benchmark_metadata(row["symbol"], row["name"]), axis=1)
    for col in ["macro_class", "region", "currency", "segment"]:
        df[col] = meta_rows.apply(lambda item: item.get(col, ""))

    return df[["symbol", "ticker", "name", "weight_pct", "macro_class", "region", "currency", "segment"]]


def fetch_vnga60_holdings(url: str = VNGA60_HOLDINGS_URL, timeout: int = 12) -> pd.DataFrame:
    """Fetch VNGA60 holdings from the public source."""
    return fetch_lifestrategy_holdings("VNGA60", url=url, timeout=timeout)


def fetch_lifestrategy_holdings(ticker: str, url: str | None = None, timeout: int = 12) -> pd.DataFrame:
    """Fetch a LifeStrategy fund's underlying ETFs (e.g. VNGA60, VNGA40) from the public source."""
    url = url or LIFESTRATEGY_HOLDINGS_URL.format(ticker=ticker)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SFCPortfolioTracker/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    holdings = parse_stockanalysis_holdings(response.text)
    if holdings["weight_pct"].sum() < 90:
        raise ValueError("Fetched benchmark holdings look incomplete")
    enriched = enrich_benchmark_holdings(holdings)
    enriched["source"] = url
    return enriched


def load_vnga60_holdings(cache_path: str | Path | None = None, force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Load VNGA60 holdings automatically, with cache and fallback."""
    return load_lifestrategy_holdings("VNGA60", cache_path=cache_path, force_refresh=force_refresh)


def load_lifestrategy_holdings(ticker: str, cache_path: str | Path | None = None,
                               force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Load a LifeStrategy fund's holdings automatically, with cache and fallback."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists() and not force_refresh:
        try:
            cached = pd.read_csv(cache)
            if not cached.empty and cached["weight_pct"].sum() > 90:
                return enrich_benchmark_holdings(cached), "cache"
        except Exception:
            pass

    try:
        live = fetch_lifestrategy_holdings(ticker)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            live.to_csv(cache, index=False)
        return live, "live"
    except Exception:
        fallback = enrich_benchmark_holdings(pd.DataFrame(LIFESTRATEGY_FALLBACK_HOLDINGS.get(ticker, VNGA60_FALLBACK_HOLDINGS)))
        fallback["source"] = "fallback"
        return fallback, "fallback"


def blend_holdings(parts: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """Weighted sum of several funds' underlying ETFs (same ETF -> one row)."""
    frames = []
    for holdings, weight in parts:
        if holdings is None or holdings.empty:
            continue
        df = enrich_benchmark_holdings(holdings[["symbol", "name", "weight_pct"]])
        df["weight_pct"] = df["weight_pct"] * float(weight)
        frames.append(df)
    if not frames:
        return enrich_benchmark_holdings(pd.DataFrame())

    combined = pd.concat(frames, ignore_index=True)
    blended = combined.groupby("ticker", as_index=False, sort=False).agg(
        symbol=("symbol", "first"), name=("name", "first"), weight_pct=("weight_pct", "sum"),
    )
    result = enrich_benchmark_holdings(blended[["symbol", "name", "weight_pct"]])
    result["weight_pct"] = result["weight_pct"].round(4)
    return result.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def rescale_sleeves(holdings: pd.DataFrame, equity_pct: float, bond_pct: float) -> pd.DataFrame:
    """Rescale a LifeStrategy look-through to another equity/bond split
    (equity ETFs sum to `equity_pct`, bond ETFs to `bond_pct`)."""
    if holdings is None or holdings.empty:
        return holdings
    df = holdings.copy()
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.0)
    for macro, target in (("Equity", equity_pct), ("Fixed Income", bond_pct)):
        mask = df["macro_class"] == macro
        total = float(df.loc[mask, "weight_pct"].sum())
        if total > 0:
            df.loc[mask, "weight_pct"] = df.loc[mask, "weight_pct"] / total * float(target)
    df["weight_pct"] = df["weight_pct"].round(4)
    return df.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def _combined_source(sources: list[str]) -> str:
    unique = set(sources)
    return sources[0] if len(unique) == 1 else "mixed"


def load_benchmark_holdings(key: str, equity_pct: float | None = None,
                            bond_pct: float | None = None) -> tuple[pd.DataFrame, str]:
    """Underlying ETFs of a benchmark, with the source label (live/cache/fallback/mixed).

    - VNGA50/50: 50% VNGA60 + 50% VNGA40 holdings;
    - VNGA60 / VNGA40: the fund's own holdings;
    - other LifeStrategy splits (VNGA20, VNGA80): VNGA60 sleeves rescaled to
      `equity_pct` / `bond_pct` (indicative).
    """
    if key in BENCHMARK_BLENDS:
        parts, sources = [], []
        for fund, weight in BENCHMARK_BLENDS[key].items():
            holdings, source = load_lifestrategy_holdings(fund)
            parts.append((holdings, weight))
            sources.append(source)
        return blend_holdings(parts), _combined_source(sources)

    if key in LIFESTRATEGY_FALLBACK_HOLDINGS:
        return load_lifestrategy_holdings(key)

    holdings, source = load_lifestrategy_holdings("VNGA60")
    if equity_pct is None or bond_pct is None:
        return holdings, source
    return rescale_sleeves(holdings, equity_pct, bond_pct), source


def _infer_fund_region(row: pd.Series) -> str:
    name = str(row.get("name", "") or "").lower()
    sector = str(row.get("sector", "") or "").lower()
    currency = str(row.get("currency", "") or "").upper()
    macro = str(row.get("macro_class", "") or "").lower()
    combined = f"{name} {sector}"

    if macro == "alternative":
        return "Alternative / Global"
    if any(token in combined for token in ("china", "cina")):
        return "China"
    if any(token in combined for token in ("emerging", "mercadolibre", "brasile", "brazil")):
        return "Emerging Markets"
    if any(token in combined for token in ("us equity", "nasdaq", "s&p", "s&p500", "russell", "treasury", "tips", "utilities us", "biotech us")):
        return "North America"
    if any(token in combined for token in ("europe", "europa", "euro", "btp", "bobl", "austria", "romania", "iliad", "lvmh", "moncler", "asml", "prysmian", "fineco")):
        return "Europe"
    if any(token in combined for token in ("uk", "rentokil")) or currency == "GBP":
        return "United Kingdom"
    if any(token in combined for token in ("australia", "emr")) or currency == "AUD":
        return "Asia Pacific ex Japan"
    if currency == "USD":
        return "North America"
    return "Global"


def fund_level1_holdings(positions: pd.DataFrame, nav_total: float, cash: float = 0.0) -> pd.DataFrame:
    """Convert current fund positions into Level 1 comparable holdings."""
    columns = ["symbol", "name", "weight_pct", "macro_class", "region", "currency", "segment"]
    if positions is None:
        positions = pd.DataFrame()

    nav_total = float(nav_total or 0)
    rows = []
    if not positions.empty:
        df = positions.copy()
        df["current_value"] = pd.to_numeric(df.get("current_value"), errors="coerce").fillna(0.0)
        for _, row in df.iterrows():
            value = float(row.get("current_value", 0) or 0)
            rows.append({
                "symbol": row.get("isin", ""),
                "name": row.get("name", ""),
                "weight_pct": value / nav_total * 100 if nav_total > 0 else 0.0,
                "macro_class": row.get("macro_class", "N/A") or "N/A",
                "region": _infer_fund_region(row),
                "currency": row.get("currency", "EUR") or "EUR",
                "segment": row.get("sector", "") or "",
            })

    cash = float(cash or 0)
    if cash > 0 and nav_total > 0:
        rows.append({
            "symbol": "CASH",
            "name": "Liquidità",
            "weight_pct": cash / nav_total * 100,
            "macro_class": "Cash",
            "region": "Cash",
            "currency": "EUR",
            "segment": "Cash",
        })

    return pd.DataFrame(rows, columns=columns).sort_values("weight_pct", ascending=False).reset_index(drop=True)


def compare_group_exposures(fund: pd.DataFrame, benchmark: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compare fund and benchmark weights by a Level 1 grouping column."""
    if fund is None or fund.empty:
        fund_group = pd.DataFrame(columns=[group_col, "fund_weight_pct"])
    else:
        fund_group = fund.copy()
        fund_group[group_col] = fund_group[group_col].fillna("N/A").replace("", "N/A")
        fund_group = fund_group.groupby(group_col, dropna=False)["weight_pct"].sum().reset_index(name="fund_weight_pct")

    if benchmark is None or benchmark.empty:
        bench_group = pd.DataFrame(columns=[group_col, "benchmark_weight_pct"])
    else:
        bench_group = benchmark.copy()
        bench_group[group_col] = bench_group[group_col].fillna("N/A").replace("", "N/A")
        bench_group = bench_group.groupby(group_col, dropna=False)["weight_pct"].sum().reset_index(name="benchmark_weight_pct")

    result = pd.merge(fund_group, bench_group, on=group_col, how="outer").fillna(0.0)
    result["active_weight_pct"] = result["fund_weight_pct"] - result["benchmark_weight_pct"]
    for col in ["fund_weight_pct", "benchmark_weight_pct", "active_weight_pct"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).round(2)
    return result.sort_values("active_weight_pct", ascending=False).reset_index(drop=True)
