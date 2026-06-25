"""
Approximate ("a spanne") sector look-through for the fund and for the VNGA60
benchmark.

Broad equity ETFs don't carry a single GICS sector, so we approximate each one
with a *typical* sector profile for the index it tracks (e.g. an S&P 500 ETF ~31%
Technology, ~13% Financials, ...). Single stocks use their real sector. Bonds,
commodities, crypto and cash fall into their own non-equity buckets.

The result is an indicative full-portfolio sector breakdown that can be compared
fund-vs-benchmark. It is NOT a precise holdings look-through — the index profiles
are rounded, broadly-known weights and will drift over time.
"""

from __future__ import annotations

import pandas as pd


# Canonical sector keys (English GICS) + non-equity buckets, in display order.
SECTOR_ORDER = [
    "Technology", "Financials", "Health Care", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Communication Services", "Energy",
    "Utilities", "Materials", "Real Estate",
    "Bonds", "Commodities", "Crypto", "Cash",
]

SECTOR_COLORS = {
    "Technology": "#6366f1",
    "Financials": "#22c55e",
    "Health Care": "#ef4444",
    "Consumer Discretionary": "#f59e0b",
    "Consumer Staples": "#eab308",
    "Industrials": "#38bdf8",
    "Communication Services": "#a855f7",
    "Energy": "#fb7185",
    "Utilities": "#14b8a6",
    "Materials": "#f97316",
    "Real Estate": "#84cc16",
    "Bonds": "#64748b",
    "Commodities": "#d4a373",
    "Crypto": "#fbbf24",
    "Cash": "#94a3b8",
}


# ── Index archetype sector profiles (approx %, normalised to 100 at import) ──
_RAW_PROFILES: dict[str, dict[str, float]] = {
    "US_BROAD": {  # S&P 500 / total US, cap-weighted
        "Technology": 31, "Financials": 13, "Health Care": 11, "Consumer Discretionary": 10,
        "Communication Services": 9, "Industrials": 8, "Consumer Staples": 6, "Energy": 4,
        "Utilities": 3, "Materials": 2.5, "Real Estate": 2.5,
    },
    "US_EQUALWEIGHT": {  # S&P 500 Equal Weight
        "Industrials": 15, "Financials": 14, "Technology": 13, "Health Care": 12,
        "Consumer Discretionary": 11, "Consumer Staples": 7, "Utilities": 6, "Materials": 6,
        "Real Estate": 6, "Energy": 5, "Communication Services": 5,
    },
    "NASDAQ100": {  # tech/growth heavy
        "Technology": 50, "Communication Services": 16, "Consumer Discretionary": 13,
        "Consumer Staples": 6, "Health Care": 6, "Industrials": 5, "Utilities": 1,
        "Materials": 1, "Financials": 1, "Energy": 0.5, "Real Estate": 0.5,
    },
    "US_SMALLCAP": {  # Russell 2000
        "Industrials": 19, "Financials": 16, "Health Care": 16, "Technology": 13,
        "Consumer Discretionary": 11, "Real Estate": 6, "Energy": 5, "Materials": 5,
        "Consumer Staples": 3, "Utilities": 3, "Communication Services": 3,
    },
    "WORLD": {  # FTSE All-World / Developed World
        "Technology": 26, "Financials": 16, "Industrials": 11, "Health Care": 11,
        "Consumer Discretionary": 10, "Communication Services": 8, "Consumer Staples": 6,
        "Energy": 4, "Materials": 4, "Utilities": 2.5, "Real Estate": 1.5,
    },
    "EUROPE": {  # STOXX Europe 600 / FTSE Developed Europe
        "Financials": 18, "Industrials": 17, "Health Care": 14, "Consumer Staples": 11,
        "Consumer Discretionary": 10, "Technology": 8, "Materials": 6, "Energy": 5,
        "Utilities": 5, "Communication Services": 4, "Real Estate": 2,
    },
    "EM": {  # Emerging Markets
        "Financials": 22, "Technology": 22, "Consumer Discretionary": 13,
        "Communication Services": 9, "Materials": 8, "Industrials": 7, "Energy": 5,
        "Consumer Staples": 5, "Health Care": 4, "Utilities": 3, "Real Estate": 2,
    },
    "JAPAN": {
        "Industrials": 23, "Consumer Discretionary": 19, "Technology": 14, "Financials": 12,
        "Health Care": 8, "Consumer Staples": 7, "Materials": 7, "Communication Services": 5,
        "Real Estate": 3, "Utilities": 1, "Energy": 1,
    },
    "ASIA_PAC_EXJP": {  # Australia-heavy
        "Financials": 30, "Materials": 15, "Industrials": 12, "Technology": 10,
        "Real Estate": 8, "Consumer Discretionary": 7, "Energy": 6, "Communication Services": 5,
        "Health Care": 3, "Consumer Staples": 2, "Utilities": 2,
    },
    "CHINA_TECH": {  # MSCI China tech tilt
        "Technology": 30, "Communication Services": 30, "Consumer Discretionary": 28,
        "Financials": 5, "Industrials": 4, "Health Care": 3,
    },
}


def _normalise(profile: dict[str, float]) -> dict[str, float]:
    total = float(sum(profile.values())) or 1.0
    return {k: v / total * 100.0 for k, v in profile.items()}


SECTOR_PROFILES = {name: _normalise(p) for name, p in _RAW_PROFILES.items()}


def _sector_profile(name: str) -> dict[str, float]:
    """Return a sector profile (dict summing to 100). `name` may be an archetype
    key or "SECTOR:<sector>" for a pure single-sector ETF."""
    if name.startswith("SECTOR:"):
        return {name.split(":", 1)[1]: 100.0}
    return SECTOR_PROFILES.get(name, {})


# Fund equity ETF (broad/sector) ISIN -> archetype or SECTOR:<gics>.
# Anything NOT listed and in macro Equity is treated as a single stock.
FUND_ETF_ARCHETYPE: dict[str, str] = {
    "IE00BNGJJT35": "US_EQUALWEIGHT",          # S&P Equal Weight
    "IE00BLNMYC90": "US_EQUALWEIGHT",          # SP Equal Weight
    "IE00B53SZB19": "NASDAQ100",               # Nasdaq
    "IE00BFMXXD54": "US_BROAD",                # Vanguard S&P500
    "IE00B4YBJ215": "US_BROAD",                # SPY4 (S&P500)
    "IE00BJ38QD84": "US_SMALLCAP",             # Russell 2000
    "IE00B4KBBD01": "SECTOR:Utilities",        # Utilities US
    "LU0489337690": "SECTOR:Real Estate",      # Real Estate EU
    "IE000NFR7C63": "CHINA_TECH",              # MSCI China Tech
    "LU0908500753": "EUROPE",                  # Stoxx Europe 600
    "IE000P16KP52": "SECTOR:Technology",       # CyberSecurity
    "IE00BM67HK77": "SECTOR:Health Care",      # Healthcare
    "IE00BYXG2H39": "SECTOR:Health Care",      # Biotech US
    "IE00BM67HN09": "SECTOR:Consumer Staples", # Consumer Staples
    "LU1437017350": "EM",                      # Emerging Markets
    "US4642875151": "SECTOR:Technology",       # IGV (software)
    "HK0000516697": "SECTOR:Health Care",      # Biotech Cina
}


# Normalise the fund's single-stock sector strings to canonical GICS keys.
_STOCK_SECTOR_MAP = {
    "technology": "Technology",
    "financials": "Financials",
    "financial": "Financials",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "industrials": "Industrials",
    "business services": "Industrials",
    "communication services": "Communication Services",
    "communications": "Communication Services",
    "energy": "Energy",
    "utilities": "Utilities",
    "materials": "Materials",
    "real estate": "Real Estate",
}


def _normalise_stock_sector(sector: str) -> str:
    return _STOCK_SECTOR_MAP.get(str(sector or "").strip().lower(), str(sector or "").strip() or "Industrials")


# Benchmark underlying-ETF region -> equity archetype.
_BENCH_REGION_ARCHETYPE = {
    "North America": "US_BROAD",
    "Developed World": "WORLD",
    "Global": "WORLD",
    "Europe": "EUROPE",
    "Emerging Markets": "EM",
    "Japan": "JAPAN",
    "Asia Pacific ex Japan": "ASIA_PAC_EXJP",
}


def _add(acc: dict[str, float], sector: str, weight: float) -> None:
    acc[sector] = acc.get(sector, 0.0) + float(weight)


def fund_sector_breakdown(positions: pd.DataFrame, nav_total: float, cash: float = 0.0) -> pd.Series:
    """Indicative full-portfolio sector weights (% of NAV) for the fund."""
    acc: dict[str, float] = {}
    nav_total = float(nav_total or 0)
    if positions is not None and not positions.empty and nav_total > 0:
        df = positions.copy()
        df["current_value"] = pd.to_numeric(df.get("current_value"), errors="coerce").fillna(0.0)
        for _, row in df.iterrows():
            w = float(row.get("current_value", 0) or 0) / nav_total * 100.0
            if w == 0:
                continue
            isin = str(row.get("isin", "") or "")
            macro = str(row.get("macro_class", "") or "")
            sector = str(row.get("sector", "") or "")

            if macro == "Fixed Income":
                _add(acc, "Bonds", w)
            elif macro == "Alternative":
                _add(acc, "Crypto" if "crypto" in sector.lower() else "Commodities", w)
            elif isin in FUND_ETF_ARCHETYPE:
                for sec, pct in _sector_profile(FUND_ETF_ARCHETYPE[isin]).items():
                    _add(acc, sec, w * pct / 100.0)
            else:  # single stock
                _add(acc, _normalise_stock_sector(sector), w)

    if cash and nav_total > 0:
        _add(acc, "Cash", float(cash) / nav_total * 100.0)

    return _ordered_series(acc)


def benchmark_sector_breakdown(holdings: pd.DataFrame, equity_pct: float = 60.0,
                               bond_pct: float = 40.0) -> pd.Series:
    """Indicative sector weights for a LifeStrategy benchmark.

    `holdings` is the VNGA60 underlying-ETF look-through. Equity ETFs are
    rescaled so they sum to `equity_pct` and bond ETFs to `bond_pct`, so the same
    function serves VNGA20/40/60/80.
    """
    acc: dict[str, float] = {}
    if holdings is None or holdings.empty:
        return _ordered_series(acc)

    df = holdings.copy()
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.0)
    df["macro_class"] = df.get("macro_class", "").astype(str)

    eq = df[df["macro_class"] == "Equity"]
    bd = df[df["macro_class"] == "Fixed Income"]
    eq_sum = float(eq["weight_pct"].sum()) or 1.0
    bd_sum = float(bd["weight_pct"].sum()) or 1.0

    for _, row in eq.iterrows():
        w = float(row["weight_pct"]) / eq_sum * float(equity_pct)
        archetype = _BENCH_REGION_ARCHETYPE.get(str(row.get("region", "")), "WORLD")
        for sec, pct in _sector_profile(archetype).items():
            _add(acc, sec, w * pct / 100.0)

    if not bd.empty:
        _add(acc, "Bonds", float(bond_pct))
    else:
        # No bond ETFs in look-through: approximate bond sleeve as a single bucket
        _add(acc, "Bonds", float(bond_pct))

    return _ordered_series(acc)


def _ordered_series(acc: dict[str, float]) -> pd.Series:
    if not acc:
        return pd.Series(dtype=float)
    ordered = [(s, acc[s]) for s in SECTOR_ORDER if s in acc and acc[s] > 1e-9]
    # any unexpected sectors not in SECTOR_ORDER, appended at the end
    extras = [(s, v) for s, v in acc.items() if s not in SECTOR_ORDER and v > 1e-9]
    items = ordered + sorted(extras, key=lambda kv: kv[1], reverse=True)
    return pd.Series({s: round(v, 4) for s, v in items})


def colors_for(labels) -> list[str]:
    return [SECTOR_COLORS.get(lbl, "#64748b") for lbl in labels]
