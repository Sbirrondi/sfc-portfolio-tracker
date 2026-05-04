import unittest

import pandas as pd

from benchmark_lookthrough import (
    compare_group_exposures,
    enrich_benchmark_holdings,
    fund_level1_holdings,
    parse_stockanalysis_holdings,
)


class BenchmarkLookthroughTests(unittest.TestCase):
    def test_parses_stockanalysis_holdings_table(self):
        html = """
        <table>
          <thead><tr><th>No.</th><th>Symbol</th><th>Name</th><th>Weight</th><th>Shares</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>LON: VHVG</td><td>Vanguard FTSE Developed World UCITS ETF</td><td>19.29%</td><td>1,260,255</td></tr>
            <tr><td>2</td><td>ETR: VAGF</td><td>Vanguard Global Aggregate Bond UCITS ETF</td><td>19.23%</td><td>6,046,311</td></tr>
          </tbody>
        </table>
        """

        result = parse_stockanalysis_holdings(html)

        self.assertEqual(result.loc[0, "symbol"], "LON: VHVG")
        self.assertAlmostEqual(result.loc[0, "weight_pct"], 19.29)
        self.assertEqual(result.loc[1, "name"], "Vanguard Global Aggregate Bond UCITS ETF")

    def test_enriches_benchmark_holdings_with_level1_metadata(self):
        raw = pd.DataFrame([
            {"symbol": "LON: VHVG", "name": "Vanguard FTSE Developed World UCITS ETF", "weight_pct": 19.29},
            {"symbol": "ETR: VAGF", "name": "Vanguard Global Aggregate Bond UCITS ETF", "weight_pct": 19.23},
        ])

        result = enrich_benchmark_holdings(raw)

        self.assertEqual(result.loc[0, "macro_class"], "Equity")
        self.assertEqual(result.loc[0, "region"], "Developed World")
        self.assertEqual(result.loc[1, "macro_class"], "Fixed Income")

    def test_builds_fund_level1_holdings_with_cash_weight(self):
        positions = pd.DataFrame([
            {
                "isin": "IE00TEST",
                "name": "S&P Equal Weight",
                "macro_class": "Equity",
                "sector": "US Equity",
                "currency": "EUR",
                "current_value": 600.0,
            },
            {
                "isin": "IT000BTP",
                "name": "Btp Tf 0,25% Mz28 Eur",
                "macro_class": "Fixed Income",
                "sector": "Gov. Italia",
                "currency": "EUR",
                "current_value": 300.0,
            },
        ])

        result = fund_level1_holdings(positions, nav_total=1_000, cash=100)

        self.assertAlmostEqual(result[result["name"] == "S&P Equal Weight"]["weight_pct"].iloc[0], 60.0)
        self.assertEqual(result[result["name"] == "S&P Equal Weight"]["region"].iloc[0], "North America")
        self.assertEqual(result[result["name"] == "Liquidità"]["macro_class"].iloc[0], "Cash")
        self.assertAlmostEqual(result[result["name"] == "Liquidità"]["weight_pct"].iloc[0], 10.0)

    def test_compares_group_exposures_with_active_weight(self):
        fund = pd.DataFrame([
            {"macro_class": "Equity", "weight_pct": 70.0},
            {"macro_class": "Fixed Income", "weight_pct": 20.0},
            {"macro_class": "Cash", "weight_pct": 10.0},
        ])
        benchmark = pd.DataFrame([
            {"macro_class": "Equity", "weight_pct": 60.0},
            {"macro_class": "Fixed Income", "weight_pct": 40.0},
        ])

        result = compare_group_exposures(fund, benchmark, "macro_class")
        equity = result[result["macro_class"] == "Equity"].iloc[0]
        fixed_income = result[result["macro_class"] == "Fixed Income"].iloc[0]

        self.assertAlmostEqual(equity["active_weight_pct"], 10.0)
        self.assertAlmostEqual(fixed_income["active_weight_pct"], -20.0)


if __name__ == "__main__":
    unittest.main()
