
# ALFA Forex Price Prediction 🚀💹

[Python](https://img.shields.io/badge/Python-3.8+-blue)

[PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) 

[Plotly](https://img.shields.io/badge/Plotly-Interactive-green) 

[License](https://img.shields.io/badge/License-MIT-brightgreen)

**Attention-enhanced LSTM (ALFA) for Accurate Forex Price Prediction**

This project implements **ALFA** — an **Attention-based Long Short-Term Memory** model designed specifically for **next-day OHLC (Open, High, Low, Close) price prediction** in the Forex market. Tested on **USD/JPY daily data**, it intelligently focuses on key temporal patterns using an attention mechanism layered on top of a multi-layer LSTM.

The model predicts **log returns** for better stationarity, then reconstructs realistic absolute prices. It also generates **MACD-based buy/sell signals** on predicted prices for potential trading insights.

![USD/JPY Overview and Analysis](https://fusionmarkets.com/static_images/screenshot_9e9a62dae8.png)

![USD/JPY Overview and Analysis](https://fusionmarkets.com/static_images/9_cwz_Wn_Udz3_Y_Ay_T_Du1e_M1fb_Hz_AZ_OTFI_be993601dc.png)

## 🌟 Key Features

Attention Mechanism → Dynamically weights important time steps in LSTM hidden states, filtering noise and improving focus on critical patterns.

!["C:\Users\mier3\Downloads\Gemini_Generated_Image_rrltrprrltrprrlt.png"](C:\Users\mier3\Downloads\Gemini_Generated_Image_rrltrprrltrprrlt.png)

**Rich Technical Indicators** → SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Momentum as input features.

**Stationary Training** → Predicts log returns (not raw prices) + strict time-based train/val/test split to prevent data leakage.

**Full OHLC Prediction** → Outputs predicted Open, High, Low, and Close for the next day.

**Interactive Visualizations** → Powered by Plotly: two detailed charts showing actual vs predicted prices, high/low ranges, and MACD crossover buy/sell signals.

## 📊 Example Results (Similar to Your Output Charts)

Your code generates **three interactive Plotly charts**:

1. Full comparison with actual high/low background shading.
2. Clean actual vs predicted close with high/low bands.
3. Enhanced version with range selector and unified hover.

XAUUSD 2022-2025 Predictive Trajectory Flow Line

![](XAUUSD%20Result.png)

EURUSD 2022-2025 Predictive Trajectory Flow Line

![](EURUSD%20Result.png)

USDJPY 2022-2025 Predictive Trajectory Flow Line

![](USDJPY%20Result.png)

## 🚀 Quick Start

### Requirements

#### **PyTorch Version & Python Compatibility**

- **Latest Version:** PyTorch 2.9 / 2.10
- **Python Requirement:** **Python 3.10** to **Python 3.13** (3.14 is currently experimental).
  - *Note:* PyTorch has dropped support for Python 3.8 and 3.9. It is recommended to use Python 3.10 or 3.11 for the best stability.

#### **CUDA (GPU Driver) Requirements**

Requirements vary depending on the CUDA version you select in your installation command:

- **CUDA 12.x Series (Mainstream Recommendation):** Supports **CUDA 12.1**, **12.4**, **12.6**, and **12.8**.
  - This is the standard choice, especially for RTX 40 and 50 series cards.
- **CUDA 11.8:** Still supported for legacy reasons or specific compatibility needs, but is gradually being phased out.

#### **GPU (Graphics Card) Requirements**

The threshold for PyTorch has risen depending on the CUDA package you choose.

- **Minimum Architecture Requirement:**
  - **NVIDIA GPUs:** Generally requires **Compute Capability 6.0 (Pascal Architecture)** or higher.
  - **Critical Warning:** If you build/install PyTorch with **CUDA 12.8** or higher, official support for **Maxwell (GTX 900 series)** and **Pascal (GTX 1000 series)** has been removed.
  - **If you use a GTX 1080Ti / 1060:** You **must** choose the **CUDA 12.6** or **CUDA 11.8** version of PyTorch. Do not install the CUDA 12.8+ build.
  - **AMD GPUs (ROCm):** Supports RX 6000/7000 series and select older models (requires the ROCm version of PyTorch).
- **VRAM (Video Memory):**
  - **Minimum:** 4GB (Only for learning simple models).
  - **Recommended:** 8GB+ (For running Stable Diffusion or Quantized LLMs).
  - **Production:** 24GB+.

#### **CPU (Processor) Requirements**

- **Instruction Set:** Must support **AVX2** instructions.
  - Most Intel CPUs (Haswell architecture [~2013] and later) and AMD CPUs (Ryzen series) support this.
- **Architecture:** x86_64 or ARM64 (e.g., Apple Silicon M1/M2/M3/M4).

| 库名             | 用途                                                         | 安装命令                   |
| ---------------- | ------------------------------------------------------------ | -------------------------- |
| **torch**        | PyTorch Deep Learning Framework (Models, Training)           | `pip install torch`        |
| **pandas**       | Data reading, cleaning, and processing DataFrame             | `pip install pandas`       |
| **numpy**        | Numerical computation, array operations                      | `pip install numpy`        |
| **matplotlib**   | Although Plotly is primarily used in the code, it still needs to be imported after | `pip install matplotlib`   |
| **plotly**       | Interactive charts (go.Figure, Scatter, etc.)                | `pip install plotly`       |
| **scikit-learn** | Provide `MinMaxScaler` and `StandardScaler`                  | `pip install scikit-learn` |

#### **Recommended one-time installation command (most commonly used single command)**

```python
pip install torch pandas numpy matplotlib plotly scikit-learn
```

### Run the Project

```python
python Main.py
```

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Forex trading involves high risk. Past performance and model predictions are not indicative of future results. Never trade with money you cannot afford to lose.

## 🤝 Contribute

Feel free to Star ⭐, Fork, or open Issues/PRs! Ideas welcome: add more currency pairs, backtesting, or ensemble models.

**Master the Forex market with ALFA — where attention meets prediction!** 💰📈