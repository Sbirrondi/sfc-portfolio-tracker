import unittest

import pandas as pd

from benchmark_contribution import (
    benchmark_symbol_to_yahoo,
    compute_benchmark_underlying_contributions,
)


class BenchmarkContributionTests(unittest.TestCase):
    def test_maps_vnga60_symbols_to_yahoo_tickers(self):
        self.assertEqual(benchmark_symbol_to_yahoo("LON: VHVG"), "VHVG.L")
        self.assertEqual(benchmark_symbol_to_yahoo("ETR: VAGF"), "VAGF.DE")
        self.assertEqual(benchmark_symbol_to_yahoo("BIT: VNGA60"), "VNGA60.MI")

    def test_computes_weighted_underlying_contribution(self):
        holdings = pd.DataFrame([
            {
                "symbol": "LON: TEST",
                "ticker": "TEST",
                "name": "Test Equity ETF",
                "weight_pct": 20.0,
                "macro_class": "Equity",
                "region": "Global",
            }
        ])
        price_data = {
            "TEST.L": pd.Series(
                [100.0, 105.0],
                index=pd.to_datetime(["2026-01-01", "2026-03-31"]),
            )
        }

        result = compute_benchmark_underlying_contributions(
            holdings,
            price_data,
            start_date="2026-01-01",
            end_date="2026-03-31",
            benchmark_return_pct=6.0,
        )

        detail = result["detail"]
        self.assertAlmostEqual(detail.loc[0, "period_return_pct"], 5.0)
        self.assertAlmostEqual(detail.loc[0, "contribution_pp"], 1.0)
        self.assertAlmostEqual(result["reconstructed_return_pct"], 1.0)
        self.assertAlmostEqual(result["residual_pp"], 5.0)

    def test_keeps_missing_underlying_in_detail_and_excludes_from_reconstruction(self):
        holdings = pd.DataFrame([
            {
                "symbol": "LON: MISS",
                "ticker": "MISS",
                "name": "Missing ETF",
                "weight_pct": 20.0,
                "macro_class": "Equity",
                "region": "Global",
            }
        ])

        result = compute_benchmark_underlying_contributions(
            holdings,
            price_data={},
            start_date="2026-01-01",
            end_date="2026-03-31",
            benchmark_return_pct=6.0,
        )

        detail = result["detail"]
        self.assertEqual(detail.loc[0, "data_status"], "N/A")
        self.assertTrue(pd.isna(detail.loc[0, "period_return_pct"]))
        self.assertAlmostEqual(detail.loc[0, "contribution_pp"], 0.0)
        self.assertAlmostEqual(result["reconstructed_return_pct"], 0.0)
        self.assertAlmostEqual(result["residual_pp"], 6.0)

    def test_summarizes_active_contribution_by_group(self):
        holdings = pd.DataFrame([
            {
                "symbol": "LON: EQ",
                "ticker": "EQ",
                "name": "Equity ETF",
                "weight_pct": 60.0,
                "macro_class": "Equity",
                "region": "North America",
            },
            {
                "symbol": "ETR: BD",
                "ticker": "BD",
                "name": "Bond ETF",
                "weight_pct": 40.0,
                "macro_class": "Fixed Income",
                "region": "Europe",
            },
        ])
        price_data = {
            "EQ.L": pd.Series([100.0, 110.0], index=pd.to_datetime(["2026-01-01", "2026-03-31"])),
            "BD.DE": pd.Series([100.0, 95.0], index=pd.to_datetime(["2026-01-01", "2026-03-31"])),
        }
        fund = pd.DataFrame([
            {"macro_class": "Equity", "region": "North America", "contribution_pp": 4.0},
            {"macro_class": "Fixed Income", "region": "Europe", "contribution_pp": -1.0},
        ])

        result = compute_benchmark_underlying_contributions(
            holdings,
            price_data,
            start_date="2026-01-01",
            end_date="2026-03-31",
            benchmark_return_pct=4.0,
            fund_contributions=fund,
        )

        macro = result["macro_summary"]
        equity = macro[macro["macro_class"] == "Equity"].iloc[0]
        fixed_income = macro[macro["macro_class"] == "Fixed Income"].iloc[0]

        self.assertAlmostEqual(equity["benchmark_contribution_pp"], 6.0)
        self.assertAlmostEqual(equity["fund_contribution_pp"], 4.0)
        self.assertAlmostEqual(equity["active_contribution_pp"], -2.0)
        self.assertAlmostEqual(fixed_income["benchmark_contribution_pp"], -2.0)
        self.assertAlmostEqual(fixed_income["active_contribution_pp"], 1.0)


if __name__ == "__main__":
    unittest.main()
