from backtest import run_backtest
import json
print("Running backtest...")
res = run_backtest('RELIANCE', '2023-01-01', '2024-01-01')
print(json.dumps(res, indent=2))
