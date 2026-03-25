"""
Data Fetcher - Recupera dati di mercato da yfinance.
Gestisce: prezzi storici, info fondamentali, benchmark, FX rates.
Cache in memoria per evitare chiamate ripetute.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import time

# In-memory cache
_info_cache = {}
_price_cache = {}


# ── Ticker Info ──────────────────────────────────────────────────────────────

def get_ticker_info(ticker: str) -> dict:
    """
    Recupera le info fondamentali di un ticker da yfinance.
    Restituisce un dict con campi normalizzati.
    """
    if ticker in _info_cache:
        return _info_cache[ticker]

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        normalized = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "asset_type": _classify_asset(info, ticker),

            # Valuation multiples
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),

            # Profitability
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),

            # Dividends & yield
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),

            # Risk
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "50d_avg": info.get("fiftyDayAverage"),
            "200d_avg": info.get("twoHundredDayAverage"),

            # Price
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),

            # Additional
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
        }

        _info_cache[ticker] = normalized
        return normalized

    except Exception as e:
        fallback = {
            "ticker": ticker, "name": ticker, "sector": "N/A",
            "industry": "N/A", "country": "N/A", "currency": "USD",
            "exchange": "N/A", "market_cap": 0, "asset_type": "Unknown",
            "trailing_pe": None, "forward_pe": None, "price_to_book": None,
            "ev_to_ebitda": None, "ev_to_revenue": None, "peg_ratio": None,
            "price_to_sales": None, "profit_margin": None, "operating_margin": None,
            "roe": None, "roa": None, "revenue_growth": None, "earnings_growth": None,
            "dividend_yield": None, "dividend_rate": None, "payout_ratio": None,
            "beta": None, "52w_high": None, "52w_low": None,
            "50d_avg": None, "200d_avg": None, "current_price": 0,
            "description": "", "employees": None,
        }
        _info_cache[ticker] = fallback
        return fallback


def _classify_asset(info: dict, ticker: str) -> str:
    """Classifica il tipo di asset in base alle info yfinance."""
    qtype = info.get("quoteType", "").upper()
    if qtype == "ETF":
        return "ETF"
    elif qtype == "MUTUALFUND":
        return "Fund"
    elif qtype == "CRYPTOCURRENCY":
        return "Crypto"
    elif qtype == "FUTURE" or qtype == "COMMODITY":
        return "Commodity"
    elif qtype == "EQUITY":
        return "Stock"
    # Fallback heuristics
    ticker_upper = ticker.upper()
    if ticker_upper.endswith("-USD") or ticker_upper.startswith("BTC") or ticker_upper.startswith("ETH"):
        return "Crypto"
    return "Stock"


# ── Historical Prices ────────────────────────────────────────────────────────

def get_historical_prices(
    tickers: list,
    start: str = None,
    end: str = None,
    period: str = "5y"
) -> dict:
    """
    Scarica prezzi storici (Close adjusted) per una lista di ticker.
    Restituisce: {ticker: pd.Series}
    """
    cache_key = tuple(sorted(tickers)) + (start, end, period)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    result = {}

    try:
        if start:
            data = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True
            )
        else:
            data = yf.download(
                tickers, period=period,
                auto_adjust=True, progress=False, threads=True
            )

        if data.empty:
            return result

        if len(tickers) == 1:
            # Single ticker: data has simple columns
            if "Close" in data.columns:
                series = data["Close"].dropna()
                result[tickers[0]] = series
        else:
            # Multiple tickers: multi-level columns
            if "Close" in data.columns.get_level_values(0):
                close_data = data["Close"]
                for ticker in tickers:
                    if ticker in close_data.columns:
                        series = close_data[ticker].dropna()
                        if not series.empty:
                            result[ticker] = series

    except Exception as e:
        # Fallback: fetch one by one
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                if start:
                    hist = t.history(start=start, end=end, auto_adjust=True)
                else:
                    hist = t.history(period=period, auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    result[ticker] = hist["Close"].dropna()
            except:
                pass

    _price_cache[cache_key] = result
    return result


def get_current_prices(tickers: list) -> dict:
    """Recupera i prezzi correnti per una lista di ticker."""
    prices = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        price = info.get("current_price", 0)
        if price and price > 0:
            prices[ticker] = price
        else:
            # Fallback: last close from history
            try:
                hist = get_historical_prices([ticker], period="5d")
                if ticker in hist and not hist[ticker].empty:
                    prices[ticker] = hist[ticker].iloc[-1]
            except:
                prices[ticker] = 0
    return prices


# ── Benchmark Data ───────────────────────────────────────────────────────────

BENCHMARKS = {
    "FTSE All-World": "VWCE.DE",     # Vanguard FTSE All-World UCITS ETF (EUR)
    "S&P 500 Equal Weight": "RSP",   # Invesco S&P 500 Equal Weight ETF
    "S&P 500": "^GSPC",
    "MSCI World": "URTH",
    "MSCI EM": "EEM",
    "NASDAQ 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "DAX": "^GDAXI",
    "FTSE MIB": "FTSEMIB.MI",
    "BTC": "BTC-USD",
    "Gold": "GC=F",
    "US 10Y Bond": "^TNX",
}


def get_benchmark_prices(benchmark_key: str, start: str = None, period: str = "5y") -> pd.Series:
    """Scarica i prezzi storici di un benchmark."""
    ticker = BENCHMARKS.get(benchmark_key, benchmark_key)
    prices = get_historical_prices([ticker], start=start, period=period)
    return prices.get(ticker, pd.Series(dtype=float))


# ── FX Rates ─────────────────────────────────────────────────────────────────

def get_fx_rate(from_currency: str, to_currency: str = "EUR") -> float:
    """Recupera il tasso di cambio corrente."""
    if from_currency == to_currency:
        return 1.0
    try:
        pair = f"{from_currency}{to_currency}=X"
        t = yf.Ticker(pair)
        info = t.info
        return info.get("regularMarketPrice", 1.0)
    except:
        return 1.0


def get_fx_history(from_currency: str, to_currency: str = "EUR", period: str = "5y") -> pd.Series:
    """Scarica la serie storica di un tasso di cambio."""
    if from_currency == to_currency:
        return pd.Series(dtype=float)
    pair = f"{from_currency}{to_currency}=X"
    prices = get_historical_prices([pair], period=period)
    return prices.get(pair, pd.Series(dtype=float))


# ── Batch Info Fetcher ───────────────────────────────────────────────────────

def get_all_tickers_info(tickers: list) -> pd.DataFrame:
    """Recupera le info per tutti i ticker del portafoglio. Restituisce un DataFrame."""
    infos = []
    for ticker in tickers:
        info = get_ticker_info(ticker)
        infos.append(info)
        time.sleep(0.2)  # Rate limiting
    return pd.DataFrame(infos)


def clear_cache():
    """Svuota la cache."""
    global _info_cache, _price_cache
    _info_cache = {}
    _price_cache = {}
