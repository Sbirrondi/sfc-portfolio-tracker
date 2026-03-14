"""
Analytics Module - Performance, risk metrics, benchmark comparison.
Calcola: rendimenti, Sharpe, Sortino, max drawdown, alpha, beta, ecc.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── Frequency Detection ──────────────────────────────────────────────────────

def detect_frequency(prices: pd.Series) -> tuple:
    """
    Detect the frequency of price data based on median time gaps.

    Returns:
        tuple: (periods_per_year, frequency_label)
            - daily: 252, "day"
            - weekly: 52, "week"
            - monthly: 12, "month"
            - quarterly: 4, "quarter"
    """
    if len(prices) < 2:
        # Default to daily if insufficient data
        return (252, "day")

    # Calculate time gaps in days
    date_diffs = prices.index.to_series().diff().dt.days
    median_gap = date_diffs.median()

    if median_gap < 5:
        return (252, "day")
    elif median_gap <= 15:
        return (52, "week")
    elif median_gap <= 45:
        return (12, "month")
    else:
        return (4, "quarter")


# ── Return Calculations ──────────────────────────────────────────────────────

def calculate_returns(prices: pd.Series, frequency: str = "daily") -> pd.Series:
    """Calcola i rendimenti percentuali da una serie di prezzi."""
    if prices.empty:
        return pd.Series(dtype=float)
    returns = prices.pct_change().dropna()
    return returns


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calcola i rendimenti cumulativi."""
    return (1 + returns).cumprod() - 1


def total_return(prices: pd.Series) -> float:
    """Rendimento totale."""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Rendimento annualizzato."""
    if returns.empty:
        return 0.0
    total_periods = len(returns)
    cum_return = (1 + returns).prod()
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return cum_return ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Volatilità annualizzata."""
    if returns.empty:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


# ── Risk Metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """Sharpe Ratio annualizzato."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """Sortino Ratio (usa solo downside deviation)."""
    ann_ret = annualized_return(returns, periods_per_year)
    downside = returns[returns < 0]
    if downside.empty:
        return float('inf')
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return float('inf')
    return (ann_ret - risk_free_rate) / downside_vol


def max_drawdown(prices: pd.Series) -> float:
    """Maximum Drawdown."""
    if prices.empty:
        return 0.0
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Serie dei drawdown."""
    if prices.empty:
        return pd.Series(dtype=float)
    peak = prices.expanding(min_periods=1).max()
    return (prices - peak) / peak


def calmar_ratio(returns: pd.Series, prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar Ratio = Ann. Return / |Max Drawdown|."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return float('inf')
    return ann_ret / mdd


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk (Historical)."""
    if returns.empty:
        return 0.0
    return returns.quantile(1 - confidence)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (Expected Shortfall)."""
    var = var_historical(returns, confidence)
    return returns[returns <= var].mean()


# ── Benchmark Comparison ─────────────────────────────────────────────────────

def calculate_alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> dict:
    """Calcola Alpha e Beta vs benchmark (CAPM)."""
    # Align dates
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "tracking_error": 0.0, "info_ratio": 0.0, "correlation": 0.0}

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    # Beta = Cov(rp, rb) / Var(rb)
    cov = np.cov(port_ret, bench_ret)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

    # Alpha (Jensen's Alpha, annualized)
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    alpha = (port_ret.mean() - rf_period) - beta * (bench_ret.mean() - rf_period)
    alpha_ann = alpha * periods_per_year

    # R-squared
    correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
    r_squared = correlation ** 2

    # Tracking Error
    excess = port_ret - bench_ret
    tracking_error = excess.std() * np.sqrt(periods_per_year)

    # Information Ratio
    info_ratio = excess.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0

    return {
        "alpha": round(alpha_ann, 4),
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "tracking_error": round(tracking_error, 4),
        "info_ratio": round(info_ratio, 4),
        "correlation": round(correlation, 4)
    }


def rolling_metrics(
    returns: pd.Series,
    window: int = 63  # ~3 months
) -> pd.DataFrame:
    """Calcola metriche rolling."""
    rolling_ret = returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
        raw=False
    )

    return pd.DataFrame({
        "rolling_return": rolling_ret,
        "rolling_volatility": rolling_vol,
        "rolling_sharpe": rolling_sharpe
    })


# ── Monthly / Yearly Returns ────────────────────────────────────────────────

def monthly_returns_table(prices: pd.Series) -> pd.DataFrame:
    """Crea la matrice dei rendimenti mensili (righe=anno, colonne=mese)."""
    if prices.empty:
        return pd.DataFrame()

    monthly = prices.resample("ME").last().pct_change(fill_method=None).dropna()
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month],
        names=["Year", "Month"]
    )
    table = monthly.unstack(level="Month")

    # Flatten column names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table.columns = [month_names[m - 1] for m in table.columns.get_level_values("Month")]

    # Add YTD column
    yearly = prices.resample("YE").last().pct_change(fill_method=None).dropna()
    yearly.index = yearly.index.year
    table["YTD"] = yearly

    return table


# ── Full Performance Report ──────────────────────────────────────────────────

def performance_report(
    portfolio_prices: pd.Series,
    benchmark_prices: pd.Series = None,
    risk_free_rate: float = 0.04
) -> dict:
    """Genera un report completo delle performance con frequenza auto-rilevata."""
    # Auto-detect frequency from portfolio data
    periods_per_year, freq_label = detect_frequency(portfolio_prices)

    returns = calculate_returns(portfolio_prices)

    # Determine period labels based on detected frequency
    if freq_label == "day":
        best_label = "Best Day"
        worst_label = "Worst Day"
        positive_label = "Positive Days %"
    elif freq_label == "week":
        best_label = "Best Week"
        worst_label = "Worst Week"
        positive_label = "Positive Weeks %"
    elif freq_label == "month":
        best_label = "Best Month"
        worst_label = "Worst Month"
        positive_label = "Positive Months %"
    else:  # quarter
        best_label = "Best Quarter"
        worst_label = "Worst Quarter"
        positive_label = "Positive Quarters %"

    report = {
        "Total Return": f"{total_return(portfolio_prices):.2%}",
        "Annualized Return": f"{annualized_return(returns, periods_per_year):.2%}",
        "Annualized Volatility": f"{annualized_volatility(returns, periods_per_year):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free_rate, periods_per_year):.2f}",
        "Sortino Ratio": f"{sortino_ratio(returns, risk_free_rate, periods_per_year):.2f}",
        "Max Drawdown": f"{max_drawdown(portfolio_prices):.2%}",
        "Calmar Ratio": f"{calmar_ratio(returns, portfolio_prices, periods_per_year):.2f}",
        "VaR (95%)": f"{var_historical(returns, 0.95):.2%}",
        "CVaR (95%)": f"{cvar(returns, 0.95):.2%}",
        best_label: f"{returns.max():.2%}" if not returns.empty else "N/A",
        worst_label: f"{returns.min():.2%}" if not returns.empty else "N/A",
        positive_label: f"{(returns > 0).mean():.1%}" if not returns.empty else "N/A",
        "Observations": len(returns),
        "Data Frequency": freq_label,
    }

    if benchmark_prices is not None and not benchmark_prices.empty:
        # Detect benchmark frequency separately (in case it differs)
        bench_periods_per_year, bench_freq_label = detect_frequency(benchmark_prices)
        # Use portfolio frequency for consistency in alpha/beta calculation
        bench_returns = calculate_returns(benchmark_prices)
        ab = calculate_alpha_beta(returns, bench_returns, risk_free_rate, periods_per_year)
        report.update({
            "Alpha (ann.)": f"{ab['alpha']:.2%}",
            "Beta": f"{ab['beta']:.2f}",
            "R²": f"{ab['r_squared']:.2f}",
            "Tracking Error": f"{ab['tracking_error']:.2%}",
            "Information Ratio": f"{ab['info_ratio']:.2f}",
            "Correlation": f"{ab['correlation']:.2f}",
            "Benchmark Return": f"{total_return(benchmark_prices):.2%}",
            "Benchmark Ann. Return": f"{annualized_return(bench_returns, periods_per_year):.2%}",
            "Benchmark Volatility": f"{annualized_volatility(bench_returns, periods_per_year):.2%}",
        })

    return report
