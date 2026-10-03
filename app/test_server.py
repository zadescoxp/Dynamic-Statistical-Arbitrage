import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import server
from server import (
    BacktestConfig,
    DynamicStatArbEngine,
    extract_close_prices,
    get_server_address,
    parse_config,
    run_analysis,
)


class ParseConfigTests(unittest.TestCase):
    def test_requires_two_distinct_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            parse_config({"tickers": ["btc-usd", " BTC-USD "]})

    def test_accepts_symbols_and_strategy_parameters(self) -> None:
        config = parse_config(
            {
                "tickers": ["btc-usd", "eth-usd"],
                "initial_capital": "25000",
                "z_entry": "2.2",
                "z_exit": "0.2",
                "z_stop": "3.8",
            }
        )
        self.assertEqual(config.tickers, ["BTC-USD", "ETH-USD"])
        self.assertEqual(config.initial_capital, 25_000.0)
        self.assertEqual(config.z_entry, 2.2)

    def test_rejects_out_of_order_z_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "exit < entry < stop"):
            parse_config(
                {
                    "tickers": ["BTC-USD", "ETH-USD"],
                    "z_entry": 2,
                    "z_exit": 2,
                    "z_stop": 3,
                }
            )

    def test_server_defaults_to_localhost_for_development(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_server_address(), ("127.0.0.1", 8000))

    def test_server_uses_render_host_and_port(self) -> None:
        with patch.dict(os.environ, {"RENDER": "true", "PORT": "10000"}, clear=True):
            self.assertEqual(get_server_address(), ("0.0.0.0", 10000))

    def test_server_rejects_invalid_port(self) -> None:
        with patch.dict(os.environ, {"PORT": "not-a-port"}, clear=True):
            with self.assertRaisesRegex(ValueError, "PORT must be an integer"):
                get_server_address()

    def test_accepts_supported_interval(self) -> None:
        config = parse_config(
            {
                "tickers": ["BTC-USD", "ETH-USD"],
                "start_date": "2026-09-01",
                "interval": "15m",
            }
        )
        self.assertEqual(config.interval, "15m")

    def test_rejects_intraday_history_outside_yahoo_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 59 days"):
            parse_config(
                {
                    "tickers": ["BTC-USD", "ETH-USD"],
                    "start_date": "2024-10-04",
                    "interval": "30m",
                }
            )

    def test_rejects_unknown_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported intervals"):
            parse_config(
                {
                    "tickers": ["BTC-USD", "ETH-USD"],
                    "interval": "5m",
                }
            )


class PriceAndEngineTests(unittest.TestCase):
    def test_extracts_close_field_from_multiindex_download(self) -> None:
        columns = pd.MultiIndex.from_product(
            [["Close", "Open"], ["BTC-USD", "ETH-USD"]]
        )
        data = pd.DataFrame(
            [[100 + index, 50 + index, 99 + index, 49 + index] for index in range(60)],
            columns=columns,
            index=pd.date_range("2025-01-01", periods=60),
        )
        prices, excluded = extract_close_prices(data, ["BTC-USD", "ETH-USD"])
        self.assertEqual(list(prices.columns), ["BTC-USD", "ETH-USD"])
        self.assertEqual(prices.iloc[0].tolist(), [100, 50])
        self.assertEqual(excluded, [])

    def test_pair_engine_returns_trade_sequence_and_final_parameters(self) -> None:
        dates = pd.date_range("2025-01-01", periods=100)
        first = pd.Series(100 + np.linspace(0, 5, 100), index=dates)
        second = pd.Series(50 + np.linspace(0, 2, 100), index=dates)
        trades, equity, half_life, beta = DynamicStatArbEngine().run_pair(first, second)
        self.assertEqual(len(equity), len(trades) + 1)
        self.assertTrue(np.isfinite(beta))
        self.assertEqual(equity[0], 100_000.0)
        self.assertTrue(np.isnan(half_life) or np.isfinite(half_life))

    def test_analysis_pipeline_returns_json_safe_results_and_writes_csv(self) -> None:
        random = np.random.default_rng(4)
        observations = 500
        base = 100 + np.cumsum(random.normal(0, 0.5, observations))
        spread = np.zeros(observations)
        for index in range(1, observations):
            spread[index] = 0.5 * spread[index - 1] + random.normal(0, 0.3)
        first = 2 * base + spread
        index = pd.date_range("2024-01-01", periods=observations)
        columns = pd.MultiIndex.from_product(
            [["Close", "Open"], ["AAA", "BBB"]]
        )
        downloaded = pd.DataFrame(
            np.column_stack([first, base, first - 1, base - 1]),
            columns=columns,
            index=index,
        )
        config = BacktestConfig(
            tickers=["AAA", "BBB"],
            start_date="2024-01-01",
            interval="1d",
            initial_capital=100_000,
            z_entry=2,
            z_exit=0,
            z_stop=3.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "summary.csv"
            with (
                patch.object(server, "YFINANCE_CACHE_CONFIGURED", False),
                patch.object(
                    server.yf, "download", return_value=downloaded
                ) as download_mock,
                patch.object(server.yf, "set_tz_cache_location") as cache_mock,
                patch.object(server, "CSV_PATH", csv_path),
            ):
                result = run_analysis(config)
            self.assertEqual(download_mock.call_args.kwargs["interval"], "1d")
            self.assertFalse(download_mock.call_args.kwargs["threads"])
            cache_mock.assert_called_once_with(
                str(Path(tempfile.gettempdir()) / "dynamic-stat-arb-yfinance-cache")
            )
            json.dumps(result, allow_nan=False)
            self.assertEqual(result["summary"]["pairs_tested"], 1)
            self.assertEqual(len(result["equity_curves"]), 1)
            curve = result["equity_curves"][0]
            self.assertEqual(curve["equity"][0], config.initial_capital)
            self.assertEqual(curve["equity"][-1], curve["final_capital"])
            self.assertTrue(csv_path.exists())
            self.assertIn("Total Trades", csv_path.read_text(encoding="utf-8"))

    def test_price_error_advises_on_rate_limited_download(self) -> None:
        download = pd.DataFrame(
            {
                ("Close", "BTC-USD"): [float("nan")] * 60,
                ("Close", "ETH-USD"): [float("nan")] * 60,
            },
            index=pd.date_range("2025-01-01", periods=60),
        )
        download.columns = pd.MultiIndex.from_tuples(download.columns)
        with self.assertRaisesRegex(ValueError, "throttling \\(HTTP 429\\)"):
            extract_close_prices(download, ["BTC-USD", "ETH-USD"])

    def test_provider_download_exception_has_render_rate_limit_guidance(self) -> None:
        config = BacktestConfig(
            tickers=["BTC-USD", "ETH-USD"],
            start_date="2024-01-01",
            interval="1d",
            initial_capital=100_000,
            z_entry=2,
            z_exit=0,
            z_stop=3.5,
        )
        with (
            patch.object(server, "YFINANCE_CACHE_CONFIGURED", True),
            patch.object(server.yf, "download", side_effect=RuntimeError("HTTP 429")),
        ):
            with self.assertRaisesRegex(ValueError, "Render.*HTTP 429"):
                run_analysis(config)


if __name__ == "__main__":
    unittest.main()
