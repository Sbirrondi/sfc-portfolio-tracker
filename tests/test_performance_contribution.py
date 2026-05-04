import unittest

import pandas as pd

from performance_contribution import (
    benchmark_period_comparison,
    compute_period_contributions,
    contribution_waterfall_items,
    period_bounds,
    summarize_contributions,
)


class PerformanceContributionTests(unittest.TestCase):
    def test_computes_eur_contribution_from_values_and_period_flows(self):
        transactions = pd.DataFrame([
            {
                "date": "2026-01-10",
                "transaction_type": "BUY",
                "isin": "US0001",
                "name": "US Stock",
                "macro_class": "Equity",
                "sector": "Technology",
                "currency": "USD",
                "quantity": 10,
                "price": 50,
                "fx_rate": 1.25,
                "fees": 2,
            },
            {
                "date": "2026-02-10",
                "transaction_type": "DIVIDEND",
                "isin": "US0001",
                "name": "US Stock",
                "macro_class": "Equity",
                "sector": "Technology",
                "currency": "USD",
                "quantity": 10,
                "price": 1,
                "fx_rate": 1.20,
                "fees": 0,
            },
            {
                "date": "2026-02-20",
                "transaction_type": "SELL",
                "isin": "US0001",
                "name": "US Stock",
                "macro_class": "Equity",
                "sector": "Technology",
                "currency": "USD",
                "quantity": 4,
                "price": 70,
                "fx_rate": 1.10,
                "fees": 1,
            },
        ])
        positions = pd.DataFrame([
            {
                "isin": "US0001",
                "name": "US Stock",
                "macro_class": "Equity",
                "sector": "Technology",
                "currency": "USD",
                "current_value": 600,
            }
        ])

        result = compute_period_contributions(
            transactions=transactions,
            positions=positions,
            start_date="2026-01-01",
            end_date="2026-03-01",
            nav_start=10_000,
            nav_end=10_462,
            start_values={"US0001": 300},
            end_values={"US0001": 600},
        )

        row = result[result["isin"] == "US0001"].iloc[0]
        expected_buys = 10 * 50 / 1.25 + 2
        expected_sells = 4 * 70 / 1.10 - 1
        expected_dividends = 10 * 1 / 1.20
        expected_contribution = 600 - 300 - expected_buys + expected_sells + expected_dividends

        self.assertAlmostEqual(row["buys_eur"], expected_buys)
        self.assertAlmostEqual(row["sells_eur"], expected_sells)
        self.assertAlmostEqual(row["dividends_eur"], expected_dividends)
        self.assertAlmostEqual(row["contribution_eur"], expected_contribution)
        self.assertAlmostEqual(row["contribution_pp"], expected_contribution / 10_000 * 100)

    def test_keeps_closed_positions_when_trade_happens_inside_period(self):
        transactions = pd.DataFrame([
            {
                "date": "2026-01-15",
                "transaction_type": "BUY",
                "isin": "IT0001",
                "name": "Italian Stock",
                "macro_class": "Equity",
                "sector": "Industrials",
                "currency": "EUR",
                "quantity": 100,
                "price": 10,
                "fx_rate": 1,
                "fees": 0,
            },
            {
                "date": "2026-02-15",
                "transaction_type": "SELL",
                "isin": "IT0001",
                "name": "Italian Stock",
                "macro_class": "Equity",
                "sector": "Industrials",
                "currency": "EUR",
                "quantity": 100,
                "price": 11,
                "fx_rate": 1,
                "fees": 0,
            },
        ])

        result = compute_period_contributions(
            transactions=transactions,
            positions=pd.DataFrame(),
            start_date="2026-01-01",
            end_date="2026-03-01",
            nav_start=10_000,
            nav_end=10_100,
            start_values={},
            end_values={},
        )

        row = result[result["isin"] == "IT0001"].iloc[0]
        self.assertEqual(row["end_value"], 0)
        self.assertAlmostEqual(row["contribution_eur"], 100)
        self.assertAlmostEqual(row["contribution_pp"], 1.0)

    def test_summarizes_contribution_by_group(self):
        rows = pd.DataFrame([
            {"macro_class": "Equity", "contribution_eur": 100, "contribution_pp": 1.0, "end_value": 500},
            {"macro_class": "Equity", "contribution_eur": -25, "contribution_pp": -0.25, "end_value": 250},
            {"macro_class": "Fixed Income", "contribution_eur": 10, "contribution_pp": 0.1, "end_value": 100},
        ])

        result = summarize_contributions(rows, "macro_class", nav_end=1_000)
        equity = result[result["macro_class"] == "Equity"].iloc[0]

        self.assertAlmostEqual(equity["contribution_eur"], 75)
        self.assertAlmostEqual(equity["contribution_pp"], 0.75)
        self.assertAlmostEqual(equity["end_weight_pct"], 75.0)

    def test_period_bounds_uses_latest_available_nav_before_target(self):
        nav = pd.DataFrame([
            {"date": "2026-01-01", "nav": 100.0},
            {"date": "2026-03-31", "nav": 108.0},
            {"date": "2026-04-30", "nav": 110.0},
            {"date": "2026-05-04", "nav": 112.0},
        ])

        result = period_bounds(nav, "1M", end_date="2026-05-04")

        self.assertEqual(result["start_date"], pd.Timestamp("2026-03-31"))
        self.assertEqual(result["end_date"], pd.Timestamp("2026-05-04"))
        self.assertEqual(result["nav_start"], 108.0)
        self.assertEqual(result["nav_end"], 112.0)

    def test_contribution_waterfall_excludes_nav_start_and_end(self):
        contrib = pd.DataFrame([
            {"name": "Big Winner", "contribution_eur": 100.0},
            {"name": "Loser", "contribution_eur": -40.0},
            {"name": "Small", "contribution_eur": 5.0},
        ])

        result = contribution_waterfall_items(contrib, residual=10.0, limit=2)

        self.assertNotIn("NAV iniziale", result["label"].tolist())
        self.assertNotIn("NAV finale", result["label"].tolist())
        self.assertEqual(result["measure"].tolist(), ["relative", "relative", "relative", "relative"])
        self.assertIn("Altri strumenti", result["label"].tolist())
        self.assertIn("Residuo/Cash", result["label"].tolist())

    def test_benchmark_period_comparison_computes_active_return(self):
        nav = pd.DataFrame([
            {"date": "2026-01-01", "nav": 10_000.0, "benchmark": 100.0},
            {"date": "2026-03-31", "nav": 10_500.0, "benchmark": 108.0},
        ])

        result = benchmark_period_comparison(
            nav,
            start_date="2026-01-01",
            end_date="2026-03-31",
            fund_return_pct=5.0,
            nav_start=10_000,
        )

        self.assertAlmostEqual(result["benchmark_return_pct"], 8.0)
        self.assertAlmostEqual(result["active_return_pp"], -3.0)
        self.assertAlmostEqual(result["active_return_eur"], -300.0)


if __name__ == "__main__":
    unittest.main()
