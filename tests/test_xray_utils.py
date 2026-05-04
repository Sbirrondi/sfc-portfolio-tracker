import unittest

import pandas as pd

from xray_utils import add_xray_sector, build_country_exposure


class XRaySectorTests(unittest.TestCase):
    def test_assigns_broad_equity_etfs_to_multisector_bucket(self):
        positions = pd.DataFrame([
            {
                "isin": "IE00BNGJJT35",
                "name": "Xtrackers S&P 500 Equal Weight",
                "macro_class": "Equity",
                "sector": "US Large Cap",
                "industry": "Large Cap USA",
                "asset_type": "ETF",
                "current_value": 100.0,
            }
        ])

        result = add_xray_sector(positions)

        self.assertEqual(result.loc[0, "xray_sector"], "Equity - Indici multi-settore")

    def test_normalizes_equity_stock_sector_names(self):
        positions = pd.DataFrame([
            {
                "isin": "IT0003549422",
                "name": "Sanlorenzo",
                "macro_class": "Equity",
                "sector": "Consumer Cyclical",
                "industry": "Lusso",
                "asset_type": "Stock",
                "current_value": 100.0,
            }
        ])

        result = add_xray_sector(positions)

        self.assertEqual(result.loc[0, "xray_sector"], "Consumer Discretionary")

    def test_splits_fixed_income_by_issuer_type_and_region(self):
        positions = pd.DataFrame([
            {
                "isin": "IT0005433690",
                "name": "BTP 0,25% Mar 2028",
                "macro_class": "Fixed Income",
                "sector": "Government Bond",
                "industry": "Gov. Italia",
                "country": "Italy",
                "current_value": 100.0,
            },
            {
                "isin": "XS2943818059",
                "name": "Iliad Holding 5.375% 2030",
                "macro_class": "Fixed Income",
                "sector": "Corporate Bond",
                "industry": "Corporate Bond",
                "country": "France",
                "current_value": 200.0,
            },
        ])

        result = add_xray_sector(positions)

        self.assertEqual(result.loc[0, "xray_sector"], "Fixed Income - Gov. Italia")
        self.assertEqual(result.loc[1, "xray_sector"], "Fixed Income - Corporate")

    def test_splits_alternatives_into_economic_buckets(self):
        positions = pd.DataFrame([
            {
                "isin": "US37954Y8710",
                "name": "Global X Uranium ETF",
                "macro_class": "Alternative",
                "sector": "Commodities",
                "industry": "Uranio & Nucleare",
                "current_value": 100.0,
            },
            {
                "isin": "GB00BJYDH287",
                "name": "WisdomTree Bitcoin ETP",
                "macro_class": "Alternative",
                "sector": "Crypto",
                "industry": "Bitcoin",
                "current_value": 200.0,
            },
        ])

        result = add_xray_sector(positions)

        self.assertEqual(result.loc[0, "xray_sector"], "Alternative - Uranio/Nucleare")
        self.assertEqual(result.loc[1, "xray_sector"], "Alternative - Crypto")


class XRayCountryExposureTests(unittest.TestCase):
    def test_country_exposure_maps_countries_and_keeps_regions_unmapped(self):
        positions = pd.DataFrame([
            {
                "isin": "US5949181045",
                "name": "Microsoft",
                "macro_class": "Equity",
                "country": "United States",
                "current_value": 100.0,
            },
            {
                "isin": "LU0908500753",
                "name": "Amundi Stoxx Europe 600",
                "macro_class": "Equity",
                "country": "Europe",
                "current_value": 300.0,
            },
        ])

        result = build_country_exposure(positions)

        usa = result[result["country"] == "United States"].iloc[0]
        europe = result[result["country"] == "Europe"].iloc[0]
        self.assertEqual(usa["iso3"], "USA")
        self.assertTrue(usa["is_mappable"])
        self.assertTrue(pd.isna(europe["iso3"]))
        self.assertFalse(europe["is_mappable"])
        self.assertAlmostEqual(float(usa["weight"]), 25.0)


if __name__ == "__main__":
    unittest.main()
