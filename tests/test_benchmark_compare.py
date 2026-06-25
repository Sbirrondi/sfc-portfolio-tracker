import unittest

import numpy as np
import pandas as pd

import benchmark_compare as bc


def _nav_df(dates, navs):
    return pd.DataFrame({"date": pd.to_datetime(dates), "nav": navs})


class BenchmarkCompareTests(unittest.TestCase):
    def setUp(self):
        # 5 business-ish days of fund NAV and one benchmark price series
        self.dates = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        self.nav = _nav_df(self.dates, [1000.0, 1010.0, 1005.0, 1020.0, 1100.0])
        # benchmark priced on a slightly different (sparser) calendar -> ffill
        self.bench = pd.DataFrame(
            {"VNGA60": [10.0, 10.5, 11.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
        )

    def test_align_forward_fills_onto_nav_dates(self):
        aligned = bc.align_to_dates(self.bench, self.dates)
        self.assertEqual(list(aligned.index), list(self.dates))
        # 2024-01-02 has no own price -> forward filled from 01-01 (10.0)
        self.assertAlmostEqual(aligned.loc[self.dates[1], "VNGA60"], 10.0)
        # 2024-01-04 forward filled from 01-03 (10.5)
        self.assertAlmostEqual(aligned.loc[self.dates[3], "VNGA60"], 10.5)
        self.assertFalse(aligned["VNGA60"].isna().any())

    def test_period_start_anchors_to_last_data_point(self):
        start = bc.period_start_date(self.dates, "Dall'Inizio", "2024-01-01")
        self.assertEqual(start, pd.Timestamp("2024-01-01"))
        # 1M before the LAST date (2024-01-05), not wall-clock today
        start_1m = bc.period_start_date(self.dates, "1M", "2024-01-01")
        self.assertEqual(start_1m, pd.Timestamp("2024-01-05") - pd.DateOffset(months=1))

    def test_rebased_frame_starts_at_100_for_benchmark(self):
        aligned = bc.align_to_dates(self.bench, self.dates)
        reb = bc.build_rebased_frame(
            self.nav, aligned, "Dall'Inizio", "2024-01-01",
            initial_nav=1000.0, selected_keys=["VNGA60"],
        )
        self.assertAlmostEqual(reb[bc.FUND_LABEL].iloc[0], 100.0)   # 1000/1000*100
        self.assertAlmostEqual(reb[bc.FUND_LABEL].iloc[-1], 110.0)  # 1100/1000*100
        self.assertAlmostEqual(reb["VNGA60"].iloc[0], 100.0)        # 10/10*100
        self.assertAlmostEqual(reb["VNGA60"].iloc[-1], 110.0)       # 11/10*100

    def test_return_table_has_fund_and_benchmark_rows(self):
        aligned = bc.align_to_dates(self.bench, self.dates)
        table = bc.compute_return_table(
            self.nav, aligned, "2024-01-01", 1000.0, ["VNGA60"],
        )
        self.assertIn(bc.FUND_LABEL, table.index)
        self.assertIn(bc.BENCHMARKS["VNGA60"]["label"], table.index)
        self.assertEqual(list(table.columns), bc.PERIODS)
        # since inception: fund +10%, benchmark +10%
        self.assertAlmostEqual(table.loc[bc.FUND_LABEL, "Dall'Inizio"], 10.0, places=4)
        self.assertAlmostEqual(
            table.loc[bc.BENCHMARKS["VNGA60"]["label"], "Dall'Inizio"], 10.0, places=4
        )

    def test_macro_composition_sums_to_100(self):
        comp = bc.macro_composition(60, 40)
        self.assertAlmostEqual(comp["weight_pct"].sum(), 100.0)
        self.assertEqual(set(comp["macro_class"]), {"Equity", "Fixed Income"})

    def test_align_empty_prices_returns_dated_frame(self):
        aligned = bc.align_to_dates(pd.DataFrame(), self.dates)
        self.assertEqual(list(aligned.index), list(self.dates))


if __name__ == "__main__":
    unittest.main()
