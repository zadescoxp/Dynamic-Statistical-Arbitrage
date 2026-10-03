# Dynamic statistical arbitrage UI

For the full strategy explanation, metric definitions, research caveats, and saved figures, see the [project README](../README.md).

This local web app takes Yahoo Finance ticker symbols, screens pairs for cointegration on the first 80% of prices, runs the dynamic Kalman/OU strategy on the remaining 20%, and displays a pair summary, completed-trade-count chart, and combined equity-curve visualization. The equity chart colors each pair from red to green by final capital, marks initial capital with a baseline, and includes an expandable pair/final-capital legend and final-capital scale. Curves track realized capital after each closed trade, matching the notebook’s executed-trade sequence. Each run writes `dynamic_statistical_arbitrage_summary.csv` to the project root.

## Run

From the project root:

```sh
./venv/bin/python app/server.py
```

Then open <http://127.0.0.1:8000>. The server uses the Python standard library; the strategy dependencies are listed in `requirements.txt`.

Enter at least two distinct Yahoo Finance ticker symbols. The default is four crypto symbols. You can change the history start date, initial capital, and entry/exit/stop z-score settings. The maximum is 40 symbols per run.

Supported price intervals are daily (`1d`), hourly (`1h`), 30-minute (`30m`), and 15-minute (`15m`). Yahoo Finance restricts intraday history; the UI automatically moves the start date forward when changing to an interval with a shorter history window. The server validates this limit as well. OU half-life is reported in bars, so the result remains correctly labeled for every interval.

The screening fallback matches the notebook: if no pair passes the 5% cointegration threshold, it tests the 10 pairs with the lowest p-values. CSV download contains one row per tested pair.

Yahoo Finance data access requires an internet connection. Prices use the adjusted closing series (`auto_adjust=True`); no look-ahead fill is applied to missing prices.
