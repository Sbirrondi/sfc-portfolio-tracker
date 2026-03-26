"""
Analytics Plus - Modulo avanzato di analisi supplementare.
Integra QuantStats e PyPortfolioOpt per:
- Metriche di rischio avanzate (VaR, CVaR, Calmar, Information Ratio, ...)
- Rolling statistics (Sharpe, Sortino, Volatilità, Beta)
- Drawdown dettagliati
- Monte Carlo simulation
- Ottimizzazione portafoglio (Efficient Frontier, Max Sharpe, Min Vol, HRP)
- Risk contribution analysis
- Report HTML generabile
NON modifica i dati del fondo. Solo analisi e visualizzazione.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ── Lazy imports (non bloccare se manca qualcosa) ────────────────────────────

def _import_qs():
    import quantstats as qs
    return qs

def _import_pypfopt():
    from pypfopt import (
        expected_returns, risk_models,
        EfficientFrontier, HRPOpt,
    )
    return expected_returns, risk_models, EfficientFrontier, HRPOpt


# ══════════════════════════════════════════════════════════════════════════════
# DAILY NAV RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_daily_nav_series(
    positions: pd.DataFrame,
    isin_map: dict,
    inception_date: str = "2023-10-01",
    initial_nav: float = 10_000_000,
    cash: float = 0,
) -> pd.DataFrame:
    """
    Ricostruisce la serie giornaliera del NAV del portafoglio.
    Scarica i prezzi storici di tutti gli strumenti mappati e calcola
    il valore giornaliero del portafoglio.

    Returns: DataFrame con colonne [date, nav, benchmark]
    """
    import yfinance as yf

    if positions.empty:
        return pd.DataFrame(columns=["date", "nav", "benchmark"])

    # Get mapped tickers and their quantities/costs
    holdings = []
    for _, row in positions.iterrows():
        isin = row.get("isin", "")
        ticker = isin_map.get(isin)
        if ticker and row.get("quantity", 0) > 0:
            holdings.append({
                "isin": isin,
                "ticker": ticker,
                "quantity": float(row["quantity"]),
                "invested_capital": float(row.get("invested_capital", 0)),
                "currency": row.get("currency", "EUR"),
            })

    if not holdings:
        return pd.DataFrame(columns=["date", "nav", "benchmark"])

    tickers = list(set(h["ticker"] for h in holdings))
    tickers.append("VNGA60.MI")  # Benchmark: Vanguard LifeStrategy 60% Equity UCITS ETF

    # Download all historical prices in one batch
    start = inception_date
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    except Exception:
        return pd.DataFrame(columns=["date", "nav", "benchmark"])

    if data.empty:
        return pd.DataFrame(columns=["date", "nav", "benchmark"])

    # Extract close prices
    if len(tickers) == 1:
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]
    else:
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        else:
            return pd.DataFrame(columns=["date", "nav", "benchmark"])

    # Forward fill missing prices (holidays, different exchanges)
    close = close.ffill()

    # FX rates for non-EUR positions
    fx_tickers = set()
    for h in holdings:
        if h["currency"] != "EUR":
            fx_tickers.add(f"{h['currency']}EUR=X")

    fx_data = {}
    if fx_tickers:
        try:
            fx_raw = yf.download(list(fx_tickers), start=start, auto_adjust=True, progress=False)
            if not fx_raw.empty:
                if len(fx_tickers) == 1:
                    pair = list(fx_tickers)[0]
                    fx_data[pair] = fx_raw["Close"].ffill()
                else:
                    for pair in fx_tickers:
                        if pair in fx_raw["Close"].columns:
                            fx_data[pair] = fx_raw["Close"][pair].ffill()
        except Exception:
            pass

    # Calculate daily portfolio value
    dates = close.index
    nav_values = []

    for date in dates:
        port_value = 0
        for h in holdings:
            ticker = h["ticker"]
            if ticker in close.columns and pd.notna(close.loc[date, ticker]):
                price = close.loc[date, ticker]
                qty = h["quantity"]
                value = qty * price

                # Convert to EUR if needed
                if h["currency"] != "EUR":
                    fx_pair = f"{h['currency']}EUR=X"
                    if fx_pair in fx_data and date in fx_data[fx_pair].index:
                        fx_rate = fx_data[fx_pair].loc[date]
                        if pd.notna(fx_rate) and fx_rate > 0:
                            value = value * fx_rate
                port_value += value

        nav_values.append(port_value + cash)

    # Build result DataFrame
    result = pd.DataFrame({
        "date": dates,
        "nav": nav_values,
    })

    # Add benchmark
    if "VNGA60.MI" in close.columns:
        bench = close["VNGA60.MI"].dropna()
        if not bench.empty:
            # Rebase benchmark to initial_nav
            bench_rebased = bench / bench.iloc[0] * initial_nav
            result = result.merge(
                pd.DataFrame({"date": bench.index, "benchmark": bench_rebased.values}),
                on="date", how="left"
            )
    if "benchmark" not in result.columns:
        result["benchmark"] = np.nan

    result = result.dropna(subset=["nav"])
    result = result[result["nav"] > 0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# QUANTSTATS METRICS
# ══════════════════════════════════════════════════════════════════════════════

def advanced_risk_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    """
    Calcola metriche di rischio avanzate con QuantStats.
    Returns: dict con tutte le metriche.
    """
    qs = _import_qs()
    returns = returns.dropna()
    if len(returns) < 5:
        return {}

    metrics = {}

    # Core performance
    metrics["CAGR"] = qs.stats.cagr(returns)
    metrics["Sharpe"] = qs.stats.sharpe(returns)
    metrics["Sortino"] = qs.stats.sortino(returns)
    metrics["Calmar"] = qs.stats.calmar(returns)
    metrics["Omega"] = qs.stats.omega(returns)

    # Drawdown
    metrics["Max Drawdown"] = qs.stats.max_drawdown(returns)
    metrics["Recovery Factor"] = qs.stats.recovery_factor(returns)
    metrics["Ulcer Index"] = qs.stats.ulcer_index(returns)

    # Risk
    metrics["Volatilità Ann."] = qs.stats.volatility(returns)
    metrics["VaR 95%"] = qs.stats.var(returns)
    metrics["CVaR 95%"] = qs.stats.cvar(returns)
    metrics["Skew"] = qs.stats.skew(returns)
    metrics["Kurtosis"] = qs.stats.kurtosis(returns)
    metrics["Tail Ratio"] = qs.stats.tail_ratio(returns)

    # Win/Loss
    metrics["Win Rate"] = qs.stats.win_rate(returns)
    metrics["Best Day"] = qs.stats.best(returns)
    metrics["Worst Day"] = qs.stats.worst(returns)
    metrics["Profit Factor"] = qs.stats.profit_factor(returns)
    metrics["Payoff Ratio"] = qs.stats.payoff_ratio(returns)
    metrics["Kelly Criterion"] = qs.stats.kelly_criterion(returns)

    # vs Benchmark
    if benchmark_returns is not None and len(benchmark_returns) > 5:
        bench = benchmark_returns.dropna()
        # Align dates
        common = returns.index.intersection(bench.index)
        if len(common) > 5:
            r = returns.loc[common]
            b = bench.loc[common]
            metrics["Information Ratio"] = qs.stats.information_ratio(r, b)
            metrics["Treynor Ratio"] = qs.stats.treynor_ratio(r, b)
            try:
                greeks = qs.stats.greeks(r, b)
                metrics["Alpha"] = greeks.get("alpha", None)
                metrics["Beta"] = greeks.get("beta", None)
            except Exception:
                pass

    return metrics


def rolling_statistics(returns: pd.Series, window: int = 63) -> pd.DataFrame:
    """
    Calcola statistiche rolling: Sharpe, Sortino, Volatilità.
    Default window = 63 (3 mesi di trading).
    """
    qs = _import_qs()
    returns = returns.dropna()
    if len(returns) < window:
        return pd.DataFrame()

    result = pd.DataFrame(index=returns.index)
    result["Rolling Sharpe"] = qs.stats.rolling_sharpe(returns, rolling_period=window)
    result["Rolling Sortino"] = qs.stats.rolling_sortino(returns, rolling_period=window)
    result["Rolling Volatility"] = qs.stats.rolling_volatility(returns, rolling_period=window)

    return result.dropna()


def drawdown_details(returns: pd.Series) -> pd.DataFrame:
    """
    Restituisce i dettagli dei drawdown: inizio, fine, durata, profondità.
    """
    qs = _import_qs()
    try:
        dd = qs.stats.drawdown_details(returns)
        if dd is not None and not dd.empty:
            return dd.head(10)
    except Exception:
        pass
    return pd.DataFrame()


def montecarlo_simulation(returns: pd.Series, n_sims: int = 500, n_days: int = 252) -> dict:
    """
    Esegue simulazione Monte Carlo sui rendimenti.
    Returns: dict con percentili del CAGR e drawdown simulati.
    """
    qs = _import_qs()
    returns = returns.dropna()
    if len(returns) < 20:
        return {}

    try:
        mc_cagr = qs.stats.montecarlo(returns, sims=n_sims, compound=True)
        mc_dd = qs.stats.montecarlo(returns, sims=n_sims, compound=False)

        return {
            "cagr_simulations": mc_cagr,
            "drawdown_simulations": mc_dd,
        }
    except Exception:
        return {}


def generate_html_report(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    title: str = "SFC Fund Performance Report",
    output_path: str = None,
) -> str:
    """
    Genera un report HTML completo con QuantStats.
    Returns: path al file HTML generato.
    """
    qs = _import_qs()
    if output_path is None:
        output_path = str(Path(__file__).parent / "data" / "performance_report.html")

    try:
        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            title=title,
            output=output_path,
        )
        return output_path
    except Exception as e:
        return f"Errore: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# PYPORTFOLIOOPT - OTTIMIZZAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_position_prices(positions: pd.DataFrame, isin_map: dict, period: str = "2y") -> pd.DataFrame:
    """
    Scarica i prezzi storici per tutte le posizioni mappate.
    Returns: DataFrame con colonne = nomi strumenti, indice = date.
    """
    import yfinance as yf

    ticker_to_name = {}
    tickers = []
    for _, row in positions.iterrows():
        isin = row.get("isin", "")
        ticker = isin_map.get(isin)
        if ticker:
            tickers.append(ticker)
            ticker_to_name[ticker] = row.get("name", ticker)

    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        close = data[["Close"]].copy()
        close.columns = [ticker_to_name.get(tickers[0], tickers[0])]
    else:
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
            close.columns = [ticker_to_name.get(t, t) for t in close.columns]
        else:
            return pd.DataFrame()

    return close.dropna(axis=1, how="all").ffill()


def optimize_max_sharpe(prices: pd.DataFrame, current_weights: dict = None) -> dict:
    """
    Trova il portafoglio ottimale Max Sharpe.
    Returns: dict con pesi ottimali, expected return, volatility, sharpe.
    """
    exp_ret, risk_mod, EF, _ = _import_pypfopt()

    if prices.empty or len(prices) < 60:
        return {"error": "Dati insufficienti (servono almeno 60 giorni)"}

    try:
        mu = exp_ret.mean_historical_return(prices)
        S = risk_mod.CovarianceShrinkage(prices).ledoit_wolf()

        ef = EF(mu, S)
        ef.max_sharpe(risk_free_rate=0.03)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.03)

        return {
            "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
            "expected_return": perf[0],
            "volatility": perf[1],
            "sharpe": perf[2],
        }
    except Exception as e:
        return {"error": str(e)}


def optimize_min_volatility(prices: pd.DataFrame) -> dict:
    """
    Trova il portafoglio a Minima Volatilità.
    """
    exp_ret, risk_mod, EF, _ = _import_pypfopt()

    if prices.empty or len(prices) < 60:
        return {"error": "Dati insufficienti"}

    try:
        mu = exp_ret.mean_historical_return(prices)
        S = risk_mod.CovarianceShrinkage(prices).ledoit_wolf()

        ef = EF(mu, S)
        ef.min_volatility()
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.03)

        return {
            "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
            "expected_return": perf[0],
            "volatility": perf[1],
            "sharpe": perf[2],
        }
    except Exception as e:
        return {"error": str(e)}


def optimize_hrp(prices: pd.DataFrame) -> dict:
    """
    Hierarchical Risk Parity - non richiede expected returns.
    Più robusto dell'ottimizzazione classica.
    """
    _, _, _, HRPOpt = _import_pypfopt()

    if prices.empty or len(prices) < 60:
        return {"error": "Dati insufficienti"}

    try:
        returns = prices.pct_change().dropna()
        hrp = HRPOpt(returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        perf = hrp.portfolio_performance(verbose=False, risk_free_rate=0.03)

        return {
            "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
            "expected_return": perf[0],
            "volatility": perf[1],
            "sharpe": perf[2],
        }
    except Exception as e:
        return {"error": str(e)}


def efficient_frontier_curve(prices: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """
    Genera i punti della frontiera efficiente per visualizzazione.
    Returns: DataFrame con colonne [volatility, return].
    """
    exp_ret, risk_mod, EF, _ = _import_pypfopt()

    if prices.empty or len(prices) < 60:
        return pd.DataFrame()

    try:
        mu = exp_ret.mean_historical_return(prices)
        S = risk_mod.CovarianceShrinkage(prices).ledoit_wolf()

        # Generate points along the frontier
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier_points = []

        for target in target_returns:
            try:
                ef = EF(mu, S)
                ef.efficient_return(target)
                perf = ef.portfolio_performance(verbose=False)
                frontier_points.append({
                    "volatility": perf[1],
                    "return": perf[0],
                })
            except Exception:
                continue

        return pd.DataFrame(frontier_points)
    except Exception:
        return pd.DataFrame()


def risk_contribution(positions: pd.DataFrame, isin_map: dict, prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calcola il contributo al rischio di ogni posizione.
    Returns: DataFrame con [name, weight, risk_contribution, pct_contribution].
    """
    _, risk_mod, _, _ = _import_pypfopt()

    if prices is None or prices.empty or len(prices) < 60:
        return pd.DataFrame()

    try:
        returns = prices.pct_change().dropna()
        cov = risk_mod.CovarianceShrinkage(prices).ledoit_wolf()

        # Current weights from positions
        total = positions["current_value"].sum()
        if total <= 0:
            return pd.DataFrame()

        # Map position names to price columns
        name_to_weight = {}
        for _, row in positions.iterrows():
            name = row.get("name", "")
            if name in prices.columns:
                name_to_weight[name] = row["current_value"] / total

        if not name_to_weight:
            return pd.DataFrame()

        w = np.array([name_to_weight.get(col, 0) for col in prices.columns])
        w = w / w.sum()  # Normalize

        # Marginal risk contribution
        sigma = cov.values
        port_vol = np.sqrt(w @ sigma @ w)
        mrc = sigma @ w
        rc = w * mrc / port_vol
        pct_rc = rc / rc.sum() * 100

        result = pd.DataFrame({
            "name": prices.columns,
            "weight": [name_to_weight.get(col, 0) * 100 for col in prices.columns],
            "risk_contribution": rc,
            "pct_contribution": pct_rc,
        })
        result = result[result["weight"] > 0.1]
        return result.sort_values("pct_contribution", ascending=False).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Matrice di correlazione dei rendimenti."""
    if prices.empty or len(prices) < 30:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    return returns.corr()
