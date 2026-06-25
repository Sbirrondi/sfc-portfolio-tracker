import unittest

import pandas as pd

import sector_lookthrough as sl


class SectorLookthroughTests(unittest.TestCase):
    def test_profiles_normalised_to_100(self):
        for name, profile in sl.SECTOR_PROFILES.items():
            self.assertAlmostEqual(sum(profile.values()), 100.0, places=4, msg=name)

    def test_fund_breakdown_buckets_non_equity_and_sums_100(self):
        positions = pd.DataFrame([
            # single stock -> real GICS sector
            {"isin": "X1", "name": "Reply", "macro_class": "Equity", "sector": "Technology",
             "current_value": 100.0},
            # broad equity ETF -> archetype (US broad)
            {"isin": "IE00BFMXXD54", "name": "Vanguard S&P500", "macro_class": "Equity",
             "sector": "US Equity", "current_value": 100.0},
            # bond -> Bonds bucket
            {"isin": "B1", "name": "BTP", "macro_class": "Fixed Income", "sector": "Gov. Italia",
             "current_value": 100.0},
            # commodity -> Commodities bucket
            {"isin": "C1", "name": "Oro", "macro_class": "Alternative", "sector": "Precious Metals",
             "current_value": 50.0},
            # crypto -> Crypto bucket
            {"isin": "K1", "name": "Bitcoin", "macro_class": "Alternative", "sector": "Crypto",
             "current_value": 50.0},
        ])
        nav_total = 500.0  # 400 in positions + 100 cash
        breakdown = sl.fund_sector_breakdown(positions, nav_total, cash=100.0)

        self.assertAlmostEqual(breakdown.sum(), 100.0, places=4)
        self.assertAlmostEqual(breakdown["Bonds"], 20.0, places=4)        # 100/500
        self.assertAlmostEqual(breakdown["Commodities"], 10.0, places=4)  # 50/500
        self.assertAlmostEqual(breakdown["Crypto"], 10.0, places=4)       # 50/500
        self.assertAlmostEqual(breakdown["Cash"], 20.0, places=4)         # 100/500
        # Technology gets the single stock (20%) + part of the S&P500 ETF (20% * ~31%)
        self.assertGreater(breakdown["Technology"], 20.0)

    def test_benchmark_breakdown_scales_bond_sleeve(self):
        holdings = pd.DataFrame([
            {"ticker": "VWRA", "weight_pct": 60.0, "macro_class": "Equity", "region": "Global"},
            {"ticker": "VAGF", "weight_pct": 40.0, "macro_class": "Fixed Income", "region": "Global"},
        ])
        b60 = sl.benchmark_sector_breakdown(holdings, 60, 40)
        b40 = sl.benchmark_sector_breakdown(holdings, 40, 60)
        self.assertAlmostEqual(b60.sum(), 100.0, places=4)
        self.assertAlmostEqual(b40.sum(), 100.0, places=4)
        self.assertAlmostEqual(b60["Bonds"], 40.0, places=4)
        self.assertAlmostEqual(b40["Bonds"], 60.0, places=4)
        # less equity in VNGA40 -> lower Technology weight
        self.assertLess(b40["Technology"], b60["Technology"])

    def test_colors_for_known_and_unknown(self):
        cols = sl.colors_for(["Technology", "Bonds", "ZZZ"])
        self.assertEqual(cols[0], sl.SECTOR_COLORS["Technology"])
        self.assertEqual(cols[2], "#64748b")  # fallback


if __name__ == "__main__":
    unittest.main()
