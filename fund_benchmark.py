"""
Benchmark ufficiale del fondo: VNGA50/50.

Blend 50% VNGA60 (Vanguard LifeStrategy 60% Equity) + 50% VNGA40 (Vanguard
LifeStrategy 40% Equity), livello base 100 al 16/10/2023.

- `data/benchmark_vnga5050.csv` contiene la serie storica del benchmark.
- Dopo l'ultima data della serie il livello prosegue giorno per giorno pesando
  al 50% i rendimenti giornalieri dei due ETF (V60A.DE, V40A.DE):
      livello_t = livello_{t-1} x (1 + 0,5 x r_VNGA60 + 0,5 x r_VNGA40)
- La colonna `benchmark` di `fund_nav_history.csv` è il livello VNGA50/50 alla
  data di ogni NAV: Dashboard, Performance, Fondo vs Benchmark e Contribuzione
  la leggono tutte da lì.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
SERIES_FILE = DATA_DIR / "benchmark_vnga5050.csv"
NAV_HISTORY_FILE = DATA_DIR / "fund_nav_history.csv"

BENCHMARK_NAME = "VNGA50/50"
# Ticker Yahoo (Xetra) -> peso nel blend.
COMPONENTS: dict[str, float] = {"V60A.DE": 0.5, "V40A.DE": 0.5}

# Un 50/50 bilanciato non si muove del 25% tra due NAV consecutivi: oltre questa
# soglia il valore salvato è un dato sporco (es. un prezzo ETF al posto del livello).
_MAX_STEP_MOVE = 0.25


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float, name=BENCHMARK_NAME)


def _normalize_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def load_official_series(path: Path | str | None = None) -> pd.Series:
    """Serie storica VNGA50/50 (indice = data, valore = livello base 100)."""
    path = Path(path) if path else SERIES_FILE
    if not path.exists():
        return _empty_series()
    df = pd.read_csv(path)
    series = pd.Series(
        pd.to_numeric(df["benchmark"], errors="coerce").values,
        index=_normalize_index(df["date"]),
        name=BENCHMARK_NAME,
    ).dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def download_component_prices(start, end=None) -> pd.DataFrame:
    """Chiusure giornaliere dei due ETF del blend (colonne = ticker).

    Usa l'endpoint bulk di Yahoo (lo stesso dello storico NAV, affidabile anche
    da Streamlit Cloud). DataFrame vuoto se il download fallisce o se manca uno
    dei due ETF: senza entrambi il blend non si può calcolare.
    """
    import yfinance as yf

    tickers = list(COMPONENTS)
    start_ts = pd.Timestamp(start).normalize() - pd.Timedelta(days=7)
    end_ts = (pd.Timestamp(end) if end is not None else pd.Timestamp.today()).normalize() + pd.Timedelta(days=1)
    try:
        raw = yf.download(
            tickers, start=start_ts.strftime("%Y-%m-%d"), end=end_ts.strftime("%Y-%m-%d"),
            auto_adjust=False, progress=False, threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()
    if "Close" not in raw.columns.get_level_values(0):
        return pd.DataFrame()
    close = raw["Close"].copy()
    if any(t not in close.columns or close[t].dropna().empty for t in tickers):
        return pd.DataFrame()
    close.index = _normalize_index(close.index)
    return close[tickers].sort_index()


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Scarta prezzi non positivi e spike isolati (tipici glitch di Yahoo)."""
    p = prices.apply(pd.to_numeric, errors="coerce")
    p = p.where(p > 0)
    median = p.rolling(9, min_periods=3, center=True).median()
    spike = (p > median * 1.5) | (p < median / 1.5)
    return p.mask(spike)


def extend_series(official: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Livelli VNGA50/50 per i giorni di borsa successivi all'ultima data della
    serie storica, concatenati giorno per giorno dal suo ultimo livello."""
    if official is None or official.empty or prices is None or prices.empty:
        return _empty_series()
    if any(t not in prices.columns for t in COMPONENTS):
        return _empty_series()

    anchor_date = official.index.max()
    anchor_level = float(official.iloc[-1])

    p = prices[list(COMPONENTS)].copy()
    p.index = _normalize_index(p.index)
    p = _clean_prices(p[~p.index.duplicated(keep="last")].sort_index())

    base = p[p.index <= anchor_date].dropna()
    after = p[p.index > anchor_date].dropna(how="all")
    if base.empty or after.empty:
        return _empty_series()

    # Un giorno in cui manca uno dei due prezzi conta come rendimento nullo per
    # quell'ETF: il movimento viene recuperato al primo prezzo disponibile.
    chain = pd.concat([base.iloc[[-1]], after]).ffill()
    returns = chain.pct_change().iloc[1:]
    weights = pd.Series(COMPONENTS)
    blended = (returns[weights.index] * weights).sum(axis=1)
    levels = anchor_level * (1.0 + blended).cumprod()
    levels.name = BENCHMARK_NAME
    return levels


def benchmark_series(prices: pd.DataFrame | None = None, download: bool = True,
                     official: pd.Series | None = None) -> pd.Series:
    """Serie VNGA50/50 completa: storica + estensione giornaliera.

    Con `download=True` e senza `prices` scarica i prezzi degli ETF dall'ultima
    data storica; se il download fallisce restituisce la sola serie storica.
    """
    official = load_official_series() if official is None else official
    if official.empty:
        return official
    if prices is None and download:
        prices = download_component_prices(official.index.max())
    extension = extend_series(official, prices) if prices is not None else _empty_series()
    if extension.empty:
        return official
    return pd.concat([official, extension]).sort_index()


def align_to_dates(series: pd.Series, dates) -> pd.Series:
    """Livello del benchmark per ogni data richiesta (indice = date richieste).

    Valore esatto se la data è nella serie, altrimenti l'ultimo livello
    precedente (weekend, festivi). NaN dopo l'ultima data della serie.
    """
    idx = _normalize_index(dates)
    if series is None or series.empty:
        return pd.Series(np.nan, index=idx, name=BENCHMARK_NAME)
    s = series.copy()
    s.index = _normalize_index(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    aligned = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    aligned = aligned.where(np.asarray(idx <= s.index.max()))
    aligned.name = BENCHMARK_NAME
    return aligned


def apply_to_nav_history(nav_df: pd.DataFrame, series: pd.Series | None = None) -> pd.DataFrame:
    """Riscrive la colonna `benchmark` dello storico NAV con il livello VNGA50/50.

    Fino all'ultima data di `series` vale la serie; per le date successive
    (serie non ancora estesa, es. download non riuscito) si tengono i valori già
    salvati se coerenti, altrimenti si riporta avanti l'ultimo livello valido.
    Senza `series` usa la serie storica (nessun accesso di rete).
    """
    if nav_df is None or nav_df.empty or "date" not in nav_df.columns:
        return nav_df
    series = load_official_series() if series is None else series
    if series.empty:
        return nav_df

    out = nav_df.copy()
    dates = _normalize_index(out["date"])
    order = np.argsort(dates.values, kind="stable")
    new_vals = align_to_dates(series, dates).to_numpy(dtype=float)
    if "benchmark" in out.columns:
        old_vals = pd.to_numeric(out["benchmark"], errors="coerce").to_numpy(dtype=float)
    else:
        old_vals = np.full(len(out), np.nan)
    beyond = np.asarray(dates > series.index.max())

    result = np.full(len(out), np.nan)
    prev = np.nan
    for pos in order:
        if not beyond[pos]:
            val = new_vals[pos]
        else:
            val = old_vals[pos]
            if not np.isfinite(val) or (np.isfinite(prev) and abs(val / prev - 1.0) > _MAX_STEP_MOVE):
                val = prev
        result[pos] = val
        if np.isfinite(val):
            prev = val

    out["benchmark"] = np.round(result, 4)
    return out


def refresh_nav_history_file(path: Path | str | None = None, prices: pd.DataFrame | None = None,
                             download: bool = True) -> float | None:
    """Ricalcola e salva la colonna `benchmark` di fund_nav_history.csv.

    Restituisce il livello VNGA50/50 dell'ultima data NAV (None se non disponibile).
    """
    path = Path(path) if path else NAV_HISTORY_FILE
    if not path.exists():
        return None
    nav = pd.read_csv(path)
    if nav.empty:
        return None
    nav = apply_to_nav_history(nav, benchmark_series(prices=prices, download=download))
    nav.to_csv(path, index=False)
    bench = pd.to_numeric(nav["benchmark"], errors="coerce").dropna()
    return float(bench.iloc[-1]) if not bench.empty else None
