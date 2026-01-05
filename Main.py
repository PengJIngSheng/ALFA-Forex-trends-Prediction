import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Device check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 Using device: {device}')

# ==========================================
# 1. 数据准备与清洗
# ==========================================
try:
    df = pd.read_csv('EURUSD_D1_DATA.csv')
    df.rename(columns={'Date/Time': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Difference %' in df.columns:
        df['Close'] = df['Open'] * (1 + df['Difference %'] / 100)
except FileNotFoundError:
    print("⚠️ CSV not found. Generating synthetic data for demonstration...")
    dates = pd.date_range(start="2020-01-01", end="2024-01-01")
    prices = [1800]
    for _ in range(len(dates) - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    # 生成 OHLC 模拟结构
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.002))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005)))

df = df.sort_values('Date').reset_index(drop=True)


# --- 技术指标计算 (保留你的逻辑) ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA12'] = ema(df['Close'], 12)
df['EMA26'] = ema(df['Close'], 26)
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal_Line'] = ema(df['MACD'], 9)
df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


df['RSI'] = calculate_rsi(df['Close'])
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
df['TR'] = np.maximum(df['High'] - df['Low'],
                      np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
df['ATR'] = df['TR'].rolling(window=14).mean()
df['Momentum'] = df['Close'] - df['Close'].shift(4)

df = df.dropna().reset_index(drop=True)

# ==========================================
# 2. 关键修改：特征工程与非随机切分
# ==========================================
feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ATR',
                   'Momentum']

# 这里的 Log Return 是为了让模型学习“变化率”，而不是绝对价格
# Target: 下一天的 OHLC 相对于 今天 Close 的变化率
df['Ret_Open'] = np.log(df['Open'].shift(-1) / df['Close'])
df['Ret_High'] = np.log(df['High'].shift(-1) / df['Close'])
df['Ret_Low'] = np.log(df['Low'].shift(-1) / df['Close'])
df['Ret_Close'] = np.log(df['Close'].shift(-1) / df['Close'])

# 去除最后一行 NaN (因为 shift(-1))
df_model = df.iloc[:-1].copy().reset_index(drop=True)

# 严格时间切分 (Strict Sequential Split)
train_size = int(0.8 * len(df_model))
val_size = int(0.1 * len(df_model))
test_size = len(df_model) - train_size - val_size

# Scaler 只在训练集拟合 (防止数据泄露)
scaler_X = StandardScaler()  # 金融数据推荐用 StandardScaler
X_data = df_model[feature_columns].values
X_train_fit = X_data[:train_size]
scaler_X.fit(X_train_fit)
X_scaled = scaler_X.transform(X_data)

# 准备 Target (变化率，不需要缩放，因为本身就很小)
y_data = df_model[['Ret_Open', 'Ret_High', 'Ret_Low', 'Ret_Close']].values

sequence_length = 30


def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


# 全量序列化
X_seq, y_seq = create_sequences(X_scaled, y_data, sequence_length)

# 重新计算切分点 (因为序列化少掉了 sequence_length 数据)
train_idx = train_size - sequence_length
val_idx = train_idx + val_size

# 切分 Tensor
X_train = torch.FloatTensor(X_seq[:train_idx])
y_train = torch.FloatTensor(y_seq[:train_idx])
X_val = torch.FloatTensor(X_seq[train_idx:val_idx])
y_val = torch.FloatTensor(y_seq[train_idx:val_idx])
X_test = torch.FloatTensor(X_seq[val_idx:])
y_test = torch.FloatTensor(y_seq[val_idx:])

# DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)  # 训练集内部可以打乱
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")


# ==========================================
# 3. 模型定义 (ALFA - LSTM with Attention)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        scores = self.attention_fc(hidden_states).squeeze(-1)
        alpha = F.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)
        return context, alpha


class ALFA(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.attention = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        context, _ = self.attention(lstm_out)
        prediction = self.linear(context)
        return prediction


model = ALFA(input_size=len(feature_columns)).to(device)
criterion = nn.MSELoss()  # 回归问题用 MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. 训练过程
# ==========================================
epochs = 5000
best_val_loss = float('inf')

print("🚀 Starting training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            val_loss += criterion(out, by).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))
print("✅ Training finished.")

# ==========================================
# 5. 预测与价格还原 (The Reconstruction)
# ==========================================
model.eval()
all_preds_log_ret = []

# 对整个数据集进行预测 (为了画图)
full_loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)), batch_size=32,
                         shuffle=False)

with torch.no_grad():
    for bx, _ in full_loader:
        bx = bx.to(device)
        out = model(bx)
        all_preds_log_ret.append(out.cpu().numpy())

all_preds_log_ret = np.vstack(all_preds_log_ret)

# --- 关键步骤：从 Log Return 还原回 绝对价格 ---
# 预测的是 t+1 相对 t 的变化。所以 Pred_Price_t+1 = Real_Price_t * exp(Pred_Ret)
# 我们需要对应的基准价格 (Base Prices)
base_prices = df_model['Close'].values[sequence_length:]  # 对应的"前一天"收盘价

pred_open = base_prices * np.exp(all_preds_log_ret[:, 0])
pred_high = base_prices * np.exp(all_preds_log_ret[:, 1])
pred_low = base_prices * np.exp(all_preds_log_ret[:, 2])
pred_close = base_prices * np.exp(all_preds_log_ret[:, 3])

# 构建用于画图的 DataFrame
df_with_predictions = df_model.iloc[sequence_length:].copy()
df_with_predictions['Pred_Open'] = pred_open
df_with_predictions['Pred_High'] = pred_high
df_with_predictions['Pred_Low'] = pred_low
df_with_predictions['Pred_Close'] = pred_close

# --- 重新计算技术指标 (用于原本的画图逻辑) ---
df_with_predictions['Pred_EMA12'] = ema(df_with_predictions['Pred_Close'], 12)
df_with_predictions['Pred_EMA26'] = ema(df_with_predictions['Pred_Close'], 26)
df_with_predictions['Pred_MACD'] = df_with_predictions['Pred_EMA12'] - df_with_predictions['Pred_EMA26']
df_with_predictions['Pred_Signal_Line'] = ema(df_with_predictions['Pred_MACD'], 9)

# 交易信号逻辑
df_with_predictions['Buy_Signal'] = np.where(
    (df_with_predictions['Pred_MACD'] > df_with_predictions['Pred_Signal_Line']) &
    (df_with_predictions['Pred_MACD'].shift(1) <= df_with_predictions['Pred_Signal_Line'].shift(1)), 1, 0
)
df_with_predictions['Sell_Signal'] = np.where(
    (df_with_predictions['Pred_MACD'] < df_with_predictions['Pred_Signal_Line']) &
    (df_with_predictions['Pred_MACD'].shift(1) >= df_with_predictions['Pred_Signal_Line'].shift(1)), 1, 0
)

# ==========================================
# 6. 可视化 (已添加真实 High/Low)
# ==========================================
print("\nGenerating visualization charts...")
df_sampled = df_with_predictions.copy()

fig_price = go.Figure()

# --- 新增部分：真实的 High 和 Low 范围 ---
# 使用浅蓝色，细线，稍微透明一点，作为背景参考通道
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['High'],
    mode='lines', name='Actual High',
    line=dict(color='lightblue', width=1), opacity=0.6
))
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Low'],
    mode='lines', name='Actual Low',
    line=dict(color='lightblue', width=1), opacity=0.6
))
# ------------------------------------

# 1. 真实收盘价线 (主线，深蓝色，加粗)
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Close'],
    mode='lines', name='Actual Close',
    line=dict(color='blue', width=2)
))

# 2. 预测收盘价线 (主预测线，红色虚线)
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_Close'],
    mode='lines', name='Predicted Close',
    line=dict(color='red', width=2, dash='dash')
))

# 3. 预测的高低范围 (High/Low，橙色和紫色细线)
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_High'],
    mode='lines', name='Predicted High',
    line=dict(color='orange', width=1), showlegend=True
))
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_Low'],
    mode='lines', name='Predicted Low',
    line=dict(color='purple', width=1), showlegend=True
))

# 4. 买卖信号
buy_signals = df_sampled[df_sampled['Buy_Signal'] == 1]
sell_signals = df_sampled[df_sampled['Sell_Signal'] == 1]

if not buy_signals.empty:
    fig_price.add_trace(go.Scatter(
        x=buy_signals['Date'], y=buy_signals['Close'],
        mode='markers', name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: %{y:.2f}'
    ))

if not sell_signals.empty:
    fig_price.add_trace(go.Scatter(
        x=sell_signals['Date'], y=sell_signals['Close'],
        mode='markers', name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        hovertemplate='<b>Sell Signal</b><br>Date: %{x}<br>Price: %{y:.2f}'
    ))

# 布局设置
fig_price.update_layout(
    title={
        'text': 'ALFA Model Prediction vs Actual OHLC Range',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Date', yaxis_title='Price',
    hovermode='x unified', height=750, template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.8)"  # 图例背景半透明
    ),
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date",
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
)

# 优化坐标轴显示
fig_price.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)')
fig_price.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)')

fig_price.show()
print("✅ Done. Visualization updated with Actual High/Low ranges.")
print("\nGenerating visualization charts...")
df_sampled = df_with_predictions.copy()  # 如果数据量太大可做采样

fig_price = go.Figure()

# 1. 真实价格线
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Close'],
    mode='lines', name='Actual Close',
    line=dict(color='blue', width=2)
))

# 2. 预测价格线
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_Close'],
    mode='lines', name='Predicted Close',
    line=dict(color='red', width=2, dash='dash')
))

# 3. 预测的高低范围 (High/Low)
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_High'],
    mode='lines', name='Predicted High',
    line=dict(color='orange', width=1), showlegend=True
))
fig_price.add_trace(go.Scatter(
    x=df_sampled['Date'], y=df_sampled['Pred_Low'],
    mode='lines', name='Predicted Low',
    line=dict(color='purple', width=1), showlegend=True
))

# 4. 买卖信号
buy_signals = df_sampled[df_sampled['Buy_Signal'] == 1]
sell_signals = df_sampled[df_sampled['Sell_Signal'] == 1]

if not buy_signals.empty:
    fig_price.add_trace(go.Scatter(
        x=buy_signals['Date'], y=buy_signals['Close'],
        mode='markers', name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

if not sell_signals.empty:
    fig_price.add_trace(go.Scatter(
        x=sell_signals['Date'], y=sell_signals['Close'],
        mode='markers', name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

# 布局设置
fig_price.update_layout(
    title='ALFA Model Price Prediction (Stationary Training)',
    xaxis_title='Date', yaxis_title='Price',
    hovermode='x unified', height=700, template='plotly_white',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(rangeslider=dict(visible=True), type="date")
)

fig_price.show()
print("✅ Done. Price chart generated with logical integrity.")
