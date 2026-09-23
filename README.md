# BTC/USDT 1-Hour Realized Volatility Forecasting

This project studies whether short-horizon market information can improve forecasts of BTC/USDT realized volatility beyond a simple volatility-persistence baseline.

Using 5-minute Binance USD-M BTCUSDT perpetual-futures data, the model predicts realized volatility over the following one hour.

## Overview

The forecasting pipeline is:

5-minute BTCUSDT data  
→ 1-hour forward realized volatility  
→ historical + microstructure + derivatives features  
→ purged walk-forward validation

The target is
\[RV_{t,t+1h}
=
\sqrt{\sum_{j=1}^{12} r_{t+j}^2},\]


where \(r_{t+j}\) denotes the 5-minute log return.

## Features

The predictors are divided into three groups:

- **Historical features:** multi-horizon returns, realized volatility, volume, and trading activity.
- **Market microstructure:** order-book depth imbalance and related liquidity variables.
- **Derivatives positioning:** funding rate, open interest, and changes in open interest.

## Models

Four models are compared under the same out-of-sample validation framework:

- volatility-persistence baseline
- Ridge regression
- gradient-boosted trees
- PyTorch MLP

Because adjacent one-hour forward-volatility labels overlap, the evaluation uses **purged expanding walk-forward validation** to avoid train-test leakage.

## Main Results

| Model | RMSE | MAE | QLIKE |
|---|---:|---:|---:|
| GBT | 0.001397 | 0.000989 | **0.299618** |
| Ridge | **0.001389** | **0.000942** | 0.335431 |
| Persistence | 0.001577 | 0.001085 | 0.614677 |
| MLP | 0.001749 | 0.001249 | 0.731160 |

Relative to the persistence baseline:

- Ridge improves RMSE by approximately **11.9%**.
- GBT improves RMSE by approximately **11.4%** and QLIKE by approximately **51.3%**.
- The MLP underperforms the persistence benchmark out of sample.

Feature-ablation results indicate that historical variables contain most of the predictive information, while adding market-microstructure variables provides a modest additional improvement. Funding-rate and open-interest variables do not provide robust incremental value in the current sample.

## Additional Analysis

The notebook also includes:

- feature-group ablation;
- volatility-regime diagnostics;
- RMSE, MAE, and QLIKE evaluation;
- a simple volatility-targeting experiment including turnover and transaction costs.

The volatility-targeting exercise shows that better statistical forecasts do not automatically translate into better trading performance.

## Data

The project uses publicly available Binance USD-M futures data, including:

- 5-minute BTCUSDT klines;
- funding rates;
- open interest;
- historical order-book depth data.

The main microstructure specification uses aggregated bid/ask depth around the order book rather than strict top-of-book imbalance.

## Repository

The main analysis is contained in:

`BTCUSDT_Volatility_Forecasting_Experiment.ipynb`

## Requirements

Main Python packages:

```text
numpy
pandas
matplotlib
scikit-learn
torch
requests
