from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import tempfile
import traceback
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CSV_PATH = PROJECT_DIR / "dynamic_statistical_arbitrage_summary.csv"
CSV_COLUMNS = [
    "Pair",
    "Final Beta (Kalman)",
    "OU Half-Life (Bars)",
    "Initial Capital",
    "Final Capital",
    "Net Profit ($)",
    "Return (%)",
    "Total Trades",
    "Win Rate (%)",
    "Max Drawdown (%)",
]
MAX_REQUEST_BYTES = 16_384
MAX_TICKERS = 40
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.^=_-]{0,24}$")
INTERVAL_MAX_HISTORY_DAYS = {
    "1d": None,
    "1h": 729,
    "30m": 59,
    "15m": 59,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class BacktestConfig:
    tickers: list[str]
    start_date: str
    interval: str
    initial_capital: float
    z_entry: float
    z_exit: float
    z_stop: float


def parse_config(payload: Any) -> BacktestConfig:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    raw_tickers = payload.get("tickers")
    if not isinstance(raw_tickers, list) or any(not isinstance(item, str) for item in raw_tickers):
        raise ValueError("Tickers must be provided as an array of symbols.")

    tickers = list(dict.fromkeys(item.strip().upper() for item in raw_tickers if item.strip()))
    if len(tickers) < 2:
        raise ValueError("Enter at least two distinct ticker symbols.")
    if len(tickers) > MAX_TICKERS:
        raise ValueError(f"Enter no more than {MAX_TICKERS} ticker symbols per run.")
    invalid = [ticker for ticker in tickers if not TICKER_PATTERN.fullmatch(ticker)]
    if invalid:
        raise ValueError(f"Invalid ticker symbol(s): {', '.join(invalid)}")

    start_date = payload.get("start_date", "2024-10-04")
    if not isinstance(start_date, str):
        raise ValueError("Start date must be in YYYY-MM-DD format.")
    try:
        parsed_date = date.fromisoformat(start_date)
    except ValueError as error:
        raise ValueError("Start date must be in YYYY-MM-DD format.") from error
    if parsed_date >= date.today():
        raise ValueError("Start date must be earlier than today.")

    interval = payload.get("interval", "1d")
    if not isinstance(interval, str) or interval not in INTERVAL_MAX_HISTORY_DAYS:
        raise ValueError("Choose one of the supported intervals: 1d, 1h, 30m, or 15m.")
    max_history_days = INTERVAL_MAX_HISTORY_DAYS[interval]
    if max_history_days is not None:
        earliest_date = date.today() - pd.Timedelta(days=max_history_days)
        if parsed_date < earliest_date:
            raise ValueError(
                f"Yahoo Finance provides at most {max_history_days} days of "
                f"{interval} data. Choose a start date on or after {earliest_date}."
            )

    def number(name: str, default: float) -> float:
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a finite number.")
        return value

    initial_capital = number("initial_capital", 100_000.0)
    z_entry = number("z_entry", 2.0)
    z_exit = number("z_exit", 0.0)
    z_stop = number("z_stop", 3.5)

    if initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")
    if not (0 <= z_exit < z_entry < z_stop):
        raise ValueError("Z-score levels must satisfy 0 <= exit < entry < stop.")

    return BacktestConfig(
        tickers=tickers,
        start_date=start_date,
        interval=interval,
        initial_capital=initial_capital,
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
    )


def extract_close_prices(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if download.empty:
        raise ValueError("Yahoo Finance returned no price data for the selected tickers and date.")

    prices: pd.DataFrame
    if isinstance(download.columns, pd.MultiIndex):
        field_level = next(
            (
                level
                for level in range(download.columns.nlevels)
                if "Close" in download.columns.get_level_values(level)
            ),
            None,
        )
        if field_level is None:
            raise ValueError("Yahoo Finance data did not contain closing prices.")
        prices = download.xs("Close", axis=1, level=field_level)
    elif "Close" in download.columns:
        prices = download[["Close"]].copy()
    else:
        raise ValueError("Yahoo Finance data did not contain closing prices.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.columns = [str(column) for column in prices.columns]
    present = [ticker for ticker in tickers if ticker in prices.columns]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if len(present) < 2:
        detail = f" Missing symbols: {', '.join(missing)}." if missing else ""
        raise ValueError(f"Fewer than two selected tickers have usable price data.{detail}")

    prices = prices[present].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).ffill()
    usable = [ticker for ticker in present if prices[ticker].notna().sum() >= 60]
    excluded = [ticker for ticker in present if ticker not in usable]
    if len(usable) < 2:
        raise ValueError("Fewer than two tickers have at least 60 valid closing prices.")
    return prices[usable].dropna(how="all"), missing + excluded


class KalmanFilterHedgeRatio:
    def __init__(self, delta: float = 1e-4, measurement_variance: float = 1e-3) -> None:
        self.delta = delta
        self.measurement_variance = measurement_variance
        self.process_variance = (delta / (1 - delta)) * np.eye(2)
        self.theta = np.zeros(2)
        self.covariance = np.zeros((2, 2))

    def update(self, x: float, y: float) -> tuple[float, float, float]:
        prior_covariance = self.covariance + self.process_variance
        measurement = np.array([1.0, x])
        error = y - np.dot(measurement, self.theta)
        variance = (
            np.dot(measurement, np.dot(prior_covariance, measurement.T))
            + self.measurement_variance
        )
        gain = np.dot(prior_covariance, measurement.T) / variance
        self.theta = self.theta + gain * error
        self.covariance = prior_covariance - np.outer(
            gain, np.dot(measurement, prior_covariance)
        )
        return float(self.theta[0]), float(self.theta[1]), float(error)


def estimate_ou_parameters(spread: pd.Series) -> tuple[float, float, float]:
    if len(spread) < 30:
        return np.nan, np.nan, np.nan
    lagged = spread.shift(1).dropna()
    current = spread.iloc[1:]
    model = sm.OLS(current, sm.add_constant(lagged)).fit()
    intercept, slope = float(model.params.iloc[0]), float(model.params.iloc[1])
    if not 0.0 < slope < 1.0:
        return np.nan, np.nan, np.nan
    theta = -np.log(slope)
    return float(np.log(2.0) / theta), float(intercept / (1.0 - slope)), float(theta)


class DynamicStatArbEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        z_stop: float = 3.5,
    ) -> None:
        self.initial_capital = initial_capital
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop

    def run_pair(
        self,
        s1_series: pd.Series,
        s2_series: pd.Series,
        lookback_window: int = 252,
        rescreen_days: int = 7,
    ) -> tuple[pd.DataFrame, np.ndarray, float, float]:
        aligned = pd.concat([s1_series, s2_series], axis=1).dropna()
        s1_series, s2_series = aligned.iloc[:, 0], aligned.iloc[:, 1]
        if len(aligned) <= 60:
            raise ValueError("Each test pair needs more than 60 valid price observations.")

        kalman = KalmanFilterHedgeRatio()
        capital = self.initial_capital
        equity_curve = [capital]
        trades: list[dict[str, Any]] = []
        in_position = False
        position_type: str | None = None
        entry_p1 = entry_p2 = units1 = units2 = 0.0
        spread_history: list[float] = []
        is_cointegrated = True
        current_half_life = np.nan
        final_beta = np.nan

        for index in range(60):
            kalman.update(float(s2_series.iloc[index]), float(s1_series.iloc[index]))

        for index in range(60, len(s1_series)):
            p1, p2 = float(s1_series.iloc[index]), float(s2_series.iloc[index])
            alpha, beta, _ = kalman.update(p2, p1)
            final_beta = beta
            spread = p1 - (beta * p2 + alpha)
            spread_history.append(spread)

            if index % rescreen_days == 0 and index >= lookback_window:
                s1_window = s1_series.iloc[index - lookback_window : index]
                s2_window = s2_series.iloc[index - lookback_window : index]
                try:
                    _, p_value, _ = coint(s1_window, s2_window)
                    is_cointegrated = bool(p_value < 0.05)
                except (ValueError, np.linalg.LinAlgError):
                    is_cointegrated = False

            if len(spread_history) >= 60:
                recent = pd.Series(spread_history[-60:])
                half_life, _, _ = estimate_ou_parameters(recent)
                if not np.isnan(half_life):
                    current_half_life = half_life
                spread_std = recent.std()
                z_score = (
                    (spread - recent.mean()) / spread_std if spread_std > 0 else 0.0
                )
            else:
                z_score = 0.0

            if not in_position and is_cointegrated and abs(z_score) >= self.z_entry:
                position_type = "LONG_SPREAD" if z_score <= -self.z_entry else "SHORT_SPREAD"
                allocation = capital * 0.15
                units1 = (allocation / 2.0) / p1
                units2 = ((allocation / 2.0) * beta) / p2
                entry_p1, entry_p2 = p1, p2
                in_position = True
            elif in_position:
                take_profit = (
                    (position_type == "LONG_SPREAD" and z_score >= self.z_exit)
                    or (position_type == "SHORT_SPREAD" and z_score <= self.z_exit)
                )
                stop_loss = abs(z_score) >= self.z_stop or not is_cointegrated
                if take_profit or stop_loss:
                    if position_type == "LONG_SPREAD":
                        pnl = units1 * (p1 - entry_p1) + units2 * (entry_p2 - p2)
                    else:
                        pnl = units1 * (entry_p1 - p1) + units2 * (p2 - entry_p2)
                    capital += pnl
                    equity_curve.append(capital)
                    trades.append(
                        {
                            "pnl": pnl,
                            "type": position_type,
                            "reason": (
                                "Coint_Break"
                                if not is_cointegrated
                                else ("SL" if stop_loss else "TP")
                            ),
                            "capital": capital,
                        }
                    )
                    in_position = False

        return (
            pd.DataFrame(trades),
            np.asarray(equity_curve),
            float(current_half_life),
            float(final_beta),
        )


def _pair_p_value(train: pd.DataFrame, first: str, second: str) -> float:
    pair_data = train[[first, second]].dropna()
    if len(pair_data) < 60 or pair_data[first].nunique() < 2 or pair_data[second].nunique() < 2:
        return np.nan
    try:
        return float(coint(pair_data[first], pair_data[second])[1])
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def run_analysis(config: BacktestConfig) -> dict[str, Any]:
    downloaded = yf.download(
        config.tickers,
        start=config.start_date,
        interval=config.interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices, excluded_tickers = extract_close_prices(downloaded, config.tickers)
    split_index = int(len(prices) * 0.8)
    if split_index < 60 or len(prices) - split_index <= 60:
        raise ValueError(
            "Not enough history for the 80/20 train/test split. Choose an earlier start date."
        )

    train, test = prices.iloc[:split_index], prices.iloc[split_index:]
    candidate_pairs: list[tuple[str, str, float]] = []
    symbols = list(prices.columns)
    for first_index, first in enumerate(symbols):
        for second in symbols[first_index + 1 :]:
            p_value = _pair_p_value(train, first, second)
            if not np.isnan(p_value):
                candidate_pairs.append((first, second, p_value))

    if not candidate_pairs:
        raise ValueError("No ticker pairs had enough valid training data to screen.")

    selected = [pair for pair in candidate_pairs if pair[2] < 0.05]
    if not selected:
        selected = sorted(candidate_pairs, key=lambda pair: pair[2])[:10]

    engine = DynamicStatArbEngine(
        initial_capital=config.initial_capital,
        z_entry=config.z_entry,
        z_exit=config.z_exit,
        z_stop=config.z_stop,
    )
    results: list[dict[str, Any]] = []
    chart: list[dict[str, Any]] = []
    equity_curves: list[dict[str, Any]] = []
    for first, second, _ in selected:
        pair_test = test[[first, second]].dropna()
        if len(pair_test) <= 60:
            continue
        trades, equity, half_life, beta = engine.run_pair(
            pair_test[first], pair_test[second]
        )
        final_capital = float(equity[-1])
        if not np.isfinite(equity).all():
            raise ValueError(f"Backtest produced non-finite equity values for {first} / {second}.")
        peak = np.maximum.accumulate(equity)
        max_drawdown = (
            float(np.min((equity - peak) / peak) * 100.0)
            if len(equity) > 1
            else 0.0
        )
        total_trades = len(trades)
        net_profit = final_capital - config.initial_capital
        results.append(
            {
                "Pair": f"{first} / {second}",
                "Final Beta (Kalman)": beta if math.isfinite(beta) else None,
                "OU Half-Life (Bars)": (
                    half_life if math.isfinite(half_life) else None
                ),
                "Initial Capital": config.initial_capital,
                "Final Capital": final_capital,
                "Net Profit ($)": net_profit,
                "Return (%)": net_profit / config.initial_capital * 100.0,
                "Total Trades": total_trades,
                "Win Rate (%)": (
                    float((trades["pnl"] > 0).mean() * 100.0) if total_trades else 0.0
                ),
                "Max Drawdown (%)": max_drawdown,
            }
        )
        chart.append({"pair": f"{first} / {second}", "trades": total_trades})
        equity_curves.append(
            {
                "pair": f"{first} / {second}",
                "final_capital": final_capital,
                "equity": [float(value) for value in equity],
            }
        )

    if not results:
        raise ValueError("No screened pairs had enough test data to run the backtest.")

    results.sort(key=lambda result: result["Net Profit ($)"], reverse=True)
    chart.sort(key=lambda result: result["trades"], reverse=True)
    equity_curves.sort(key=lambda curve: curve["final_capital"])
    save_csv(results)
    return {
        "results": results,
        "trade_counts": chart,
        "equity_curves": equity_curves,
        "summary": {
            "price_rows": len(prices),
            "train_rows": len(train),
            "test_rows": len(test),
            "pairs_screened": len(candidate_pairs),
            "pairs_tested": len(results),
            "cointegrated_pairs": sum(pair[2] < 0.05 for pair in candidate_pairs),
            "interval": config.interval,
            "csv_file": CSV_PATH.name,
        },
        "warnings": (
            [f"No usable price history for: {', '.join(excluded_tickers)}."]
            if excluded_tickers
            else []
        ),
    }


def save_csv(results: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=CSV_PATH.parent,
            prefix=f".{CSV_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(output.getvalue())
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(CSV_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "StatArbApp/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(
                200,
                (APP_DIR / "index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif path == "/api/health":
            self._send_json(200, {"status": "ok"})
        elif path == "/api/download":
            if not CSV_PATH.exists():
                self._send_json(404, {"error": "Run a backtest before downloading a CSV."})
                return
            self._send_bytes(
                200,
                CSV_PATH.read_bytes(),
                "text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{CSV_PATH.name}"'
                },
            )
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/backtest":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            config = parse_config(payload)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as error:
            self._send_json(400, {"error": str(error)})
            return

        try:
            self._send_json(200, run_analysis(config))
        except ValueError as error:
            self._send_json(422, {"error": str(error)})
        except Exception as error:
            logging.error("Backtest failed:\n%s", traceback.format_exc())
            self._send_json(
                502,
                {"error": f"Backtest failed: {error}. Check your connection and ticker symbols."},
            )

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
        self._send_bytes(status, encoded, "application/json; charset=utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format_string % args)


def get_server_address() -> tuple[str, int]:
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError as error:
        raise ValueError("PORT must be an integer between 1 and 65535.") from error
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535.")
    return host, port


def main() -> None:
    host, port = get_server_address()
    server = ThreadingHTTPServer((host, port), AppHandler)
    logging.info("Statistical arbitrage UI running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping statistical arbitrage UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
