import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn

# =================================================
# APP CONFIG
# =================================================
st.set_page_config(
    page_title="AI Market Trend Analysis",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI Market Trend Analysis Dashboard")

st.caption(
    "Industry-grade AI system for demand forecasting, business insights, "
    "seasonal scenario simulation, and deep learning comparison."
)

# =================================================
# SIDEBAR CONTROLS
# =================================================
st.sidebar.header("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Market Sales CSV",
    type=["csv"]
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=30,
    value=14
)
selected_season = st.sidebar.selectbox(
    "Season Impact Summary",
    ["Normal", "Monsoon (+15%)", "Festival (+25%)", "Off-season (-10%)"]
)


st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Dataset requirements**:\n"
    "- Columns: `Date`, `Sales`\n"
    "- Daily / regular time series data\n\n"
    "The system automatically trains AI models and generates insights."
)

if not uploaded_file:
    st.info("⬅️ Upload a CSV file from the sidebar to begin.")
    st.stop()

# =================================================
# LOAD DATA
# =================================================
df = pd.read_csv(uploaded_file)

# =================================================
# DATA VALIDATION & CLEANING
# =================================================
st.markdown("---")
st.subheader("🧹 Data Validation & Cleaning")

if not {"Date", "Sales"}.issubset(df.columns):
    st.error("CSV must contain `Date` and `Sales` columns.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

removed_rows = df.isna().any(axis=1).sum()
df.dropna(inplace=True)
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df))
c2.metric("Removed Rows", removed_rows)
c3.metric(
    "Date Range",
    f"{df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')}"
)

# =================================================
# FEATURE ENGINEERING
# =================================================
st.markdown("---")
st.subheader("🧠 Feature Engineering")

window = 7 if len(df) >= 30 else max(2, len(df) // 2)

df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year
df["day_of_week"] = df["Date"].dt.dayofweek
df["lag_1"] = df["Sales"].shift(1)
df["rolling_avg"] = df["Sales"].rolling(window).mean()

before = len(df)
df.dropna(inplace=True)

st.caption(f"Rolling window: {window} | Rows dropped due to lag features: {before - len(df)}")
st.dataframe(df.head(), use_container_width=True)

features = ["day", "month", "year", "day_of_week", "lag_1", "rolling_avg"]
X = df[features]
y = df["Sales"]

# =================================================
# TRAIN / TEST SPLIT
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =================================================
# CLASSICAL MODELS
# =================================================
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

lr_mae = mean_absolute_error(y_test, lr_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

# =================================================
# TRANSFORMER MODEL
# =================================================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

def make_sequences(X, y, seq_len=7):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X.iloc[i:i+seq_len].values)
        ys.append(y.iloc[i+seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = make_sequences(X, y)
split = int(0.8 * len(X_seq))

X_tr, X_te = X_seq[:split], X_seq[split:]
y_tr, y_te = y_seq[:split], y_seq[split:]

X_tr = torch.tensor(X_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
X_te = torch.tensor(X_te, dtype=torch.float32)
y_te = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

transformer = TimeSeriesTransformer(X_tr.shape[2])
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.MSELoss()

for _ in range(20):
    optimizer.zero_grad()
    loss = criterion(transformer(X_tr), y_tr)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    t_preds = transformer(X_te).numpy().flatten()

transformer_rmse = mean_squared_error(y_te.numpy(), t_preds, squared=False)

# =================================================
# MODEL PERFORMANCE
# =================================================
st.markdown("---")
st.subheader("🤖 Model Performance Comparison")

c1, c2, c3 = st.columns(3)
c1.metric("Linear Regression RMSE", f"{lr_rmse:.2f}")
c2.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
c3.metric("Transformer RMSE", f"{transformer_rmse:.2f}")

best_model = lr if lr_rmse <= rf_rmse else rf
best_name = "Linear Regression" if lr_rmse <= rf_rmse else "Random Forest"

st.success(f"Final forecasting model used: **{best_name}**")

# =================================================
# FUTURE FORECAST
# =================================================
st.markdown("---")
st.subheader("📈 Future Sales Forecast")

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)

future_df = pd.DataFrame({"Date": future_dates})
future_df["day"] = future_df["Date"].dt.day
future_df["month"] = future_df["Date"].dt.month
future_df["year"] = future_df["Date"].dt.year
future_df["day_of_week"] = future_df["Date"].dt.dayofweek
future_df["lag_1"] = df["Sales"].iloc[-1]
future_df["rolling_avg"] = df["rolling_avg"].iloc[-1]

future_df["Predicted_Sales"] = best_model.predict(future_df[features])
st.dataframe(future_df, use_container_width=True)

# =================================================
# BUSINESS INSIGHTS
# =================================================
st.markdown("---")
st.subheader("🧠 Business Insights")

hist_growth = (df["Sales"].iloc[-1] - df["Sales"].iloc[0]) / df["Sales"].iloc[0] * 100
forecast_growth = (
    (future_df["Predicted_Sales"].iloc[-1] - df["Sales"].iloc[-1])
    / df["Sales"].iloc[-1] * 100
)
volatility = df["Sales"].pct_change().std() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Historical Growth (%)", f"{hist_growth:.2f}")
c2.metric("Forecast Growth (%)", f"{forecast_growth:.2f}")
c3.metric("Demand Volatility (%)", f"{volatility:.2f}")

if forecast_growth > 5:
    st.success("📈 Demand expected to grow → Increase inventory & marketing.")
elif forecast_growth < -5:
    st.warning("📉 Demand may decline → Optimize inventory & reduce risk.")
else:
    st.info("➖ Demand appears stable → Maintain current strategy.")

# =================================================
# MULTI-SCENARIO SIMULATION
# =================================================
st.markdown("---")
st.subheader("🌦️ Multi-Scenario Demand Simulation")

scenarios = {
    "Normal": 1.00,
    "Monsoon (+15%)": 1.15,
    "Festival (+25%)": 1.25,
    "Off-season (-10%)": 0.90
}

scenario_df = pd.DataFrame({"Date": future_df["Date"]})
for name, m in scenarios.items():
    scenario_df[name] = future_df["Predicted_Sales"] * m

display_df = scenario_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
st.dataframe(display_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df["Date"], df["Sales"], label="Actual Sales", linewidth=2)

for name in scenarios:
    ax.plot(scenario_df["Date"], scenario_df[name], linestyle="--", label=name)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# =================================================
# CLEAR SEASONAL IMPACT OUTPUT (ADD-ON ONLY)
# =================================================
st.markdown("---")
st.subheader("🌦️ Seasonal Impact Summary")

# Compare selected season against Normal
normal_avg = scenario_df["Normal"].mean()
season_avg = scenario_df[selected_season].mean()

change_pct = ((season_avg - normal_avg) / normal_avg) * 100

if change_pct > 5:
    st.success(
        f"📈 During **{selected_season}**, sales are expected to "
        f"**INCREASE** by approximately **{change_pct:.2f}%**."
    )
elif change_pct < -5:
    st.warning(
        f"📉 During **{selected_season}**, sales are expected to "
        f"**DECREASE** by approximately **{abs(change_pct):.2f}%**."
    )
else:
    st.info(
        f"➖ During **{selected_season}**, sales are expected to "
        f"**REMAIN STABLE**."
    )

# =================================================
# EXPORT
# =================================================
st.markdown("---")
st.subheader("📤 Export Forecast")

export_df = scenario_df.copy()
export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")

st.download_button(
    "⬇️ Download Multi-Scenario Forecast CSV",
    export_df.to_csv(index=False),
    file_name="multi_scenario_forecast.csv",
    mime="text/csv"
)

st.success("✅ Analysis completed successfully")
# =================================================
# KPI CARD – SEASONAL IMPACT SUMMARY
# =================================================
st.markdown("---")
st.subheader("📌 Seasonal Impact KPI")

normal_avg = scenario_df["Normal"].mean()
season_avg = scenario_df[selected_season].mean()
change_pct = ((season_avg - normal_avg) / normal_avg) * 100

# KPI decision logic
if change_pct > 5:
    status = "📈 SALES INCREASING"
    color = "#16a34a"   # green
    recommendation = "Increase inventory and marketing efforts."
elif change_pct < -5:
    status = "📉 SALES DECREASING"
    color = "#ea580c"   # orange
    recommendation = "Reduce inventory risk and control costs."
else:
    status = "➖ SALES STABLE"
    color = "#2563eb"   # blue
    recommendation = "Maintain current strategy and monitor trends."

# KPI Card UI
st.markdown(
    f"""
    <div style="
        background-color:#0f172a;
        border-left:6px solid {color};
        padding:20px;
        border-radius:12px;
        margin-top:10px;
    ">
        <h3 style="margin-bottom:8px;">🌦️ {selected_season} Season</h3>
        <h2 style="color:{color}; margin-bottom:8px;">{status}</h2>
        <p style="font-size:18px;">
            Change vs Normal: <b>{change_pct:.2f}%</b>
        </p>
        <p style="margin-top:10px;">
            🧠 <b>Recommendation:</b> {recommendation}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
