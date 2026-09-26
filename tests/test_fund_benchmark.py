import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import fund_benchmark as fb


def _official(values, dates):
    return pd.Series(values, index=pd.to_datetime(dates), name=fb.BENCHMARK_NAME)


class FundBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.official = _official([100.0, 101.0, 102.0], ["2026-08-27", "2026-08-28", "2026-08-31"])
        # ETF closes: anchor on 08-31, then two trading days
        self.prices = pd.DataFrame(
            {"V60A.DE": [37.0, 37.0, 37.37, 37.0], "V40A.DE": [31.0, 31.0, 31.0, 31.31]},
            index=pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02"]),
        )

    def test_repo_series_starts_at_base_100(self):
        series = fb.load_official_series()
        self.assertFalse(series.empty)
        self.assertAlmostEqual(series.iloc[0], 100.0)
        self.assertEqual(series.index[0], pd.Timestamp("2023-10-16"))
        self.assertTrue(series.index.is_monotonic_increasing)

    def test_extension_chains_daily_50_50_returns_from_last_level(self):
        ext = fb.extend_series(self.official, self.prices)
        self.assertEqual(list(ext.index), list(pd.to_datetime(["2026-09-01", "2026-09-02"])))
        # 09-01: +1% VNGA60, 0% VNGA40 -> +0.5%
        day1 = 102.0 * 1.005
        self.assertAlmostEqual(ext.iloc[0], day1, places=6)
        # 09-02: VNGA60 back to 37 (-0.990%), VNGA40 +1% -> 50/50 of the two
        r60 = 37.0 / 37.37 - 1
        self.assertAlmostEqual(ext.iloc[1], day1 * (1 + 0.5 * r60 + 0.5 * 0.01), places=6)

    def test_missing_component_price_counts_as_flat_until_next_price(self):
        prices = self.prices.copy()
        prices.loc[pd.Timestamp("2026-09-01"), "V40A.DE"] = np.nan
        ext = fb.extend_series(self.official, prices)
        self.assertAlmostEqual(ext.iloc[0], 102.0 * 1.005, places=6)
        self.assertEqual(len(ext), 2)

    def test_full_series_keeps_history_untouched(self):
        full = fb.benchmark_series(prices=self.prices, official=self.official)
        pd.testing.assert_series_equal(full.loc[self.official.index], self.official, check_freq=False)
        self.assertEqual(full.index.max(), pd.Timestamp("2026-09-02"))

    def test_without_prices_returns_history_only(self):
        full = fb.benchmark_series(prices=pd.DataFrame(), official=self.official, download=False)
        pd.testing.assert_series_equal(full, self.official)

    def test_align_forward_fills_weekends_but_not_past_last_value(self):
        dates = pd.to_datetime(["2026-08-28", "2026-08-29", "2026-08-31", "2026-09-03"])
        aligned = fb.align_to_dates(self.official, dates)
        self.assertEqual(list(aligned.values[:3]), [101.0, 101.0, 102.0])
        self.assertTrue(np.isnan(aligned.values[3]))

    def test_apply_to_nav_history_uses_series_and_sanitizes_later_values(self):
        nav = pd.DataFrame({
            "date": ["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"],
            "nav": [1.0, 2.0, 3.0, 4.0, 5.0],
            # old ETF prices in the column; 09-02 a plausible level, 09-03 an ETF price again
            "benchmark": [37.2, 37.0, 37.1, 102.5, 37.3],
        })
        out = fb.apply_to_nav_history(nav, self.official)
        self.assertEqual(list(out["date"]), list(nav["date"]))
        self.assertEqual(list(out["nav"]), list(nav["nav"]))
        vals = out["benchmark"].tolist()
        self.assertEqual(vals[:2], [101.0, 102.0])
        self.assertEqual(vals[2], 102.0)   # 37.1 rejected -> last valid level
        self.assertEqual(vals[3], 102.5)   # plausible stored level kept
        self.assertEqual(vals[4], 102.5)   # 37.3 rejected -> last valid level

    def test_refresh_nav_history_file_writes_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nav.csv"
            pd.DataFrame({
                "date": ["2026-08-31", "2026-09-01", "2026-09-02"],
                "nav": [10.0, 11.0, 12.0],
                "benchmark": [37.0, 37.37, 37.0],
            }).to_csv(path, index=False)
            original_load = fb.load_official_series
            fb.load_official_series = lambda path=None: self.official
            try:
                last = fb.refresh_nav_history_file(path=path, prices=self.prices, download=False)
            finally:
                fb.load_official_series = original_load
            saved = pd.read_csv(path)
            self.assertAlmostEqual(saved["benchmark"].iloc[0], 102.0)
            self.assertAlmostEqual(saved["benchmark"].iloc[1], round(102.0 * 1.005, 4))
            self.assertAlmostEqual(last, saved["benchmark"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
