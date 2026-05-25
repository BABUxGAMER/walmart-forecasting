import pandas as pd
import joblib
import os

# =====================================================
# 1. CONFIG
# =====================================================
DATA_PATH = "data.csv"
MODELS_DIR = "models"
STORE_PREFIX = "M"

# =====================================================
# 2. USER INPUT (NUMERIC STORE ID)
# =====================================================
store_num = input("Enter Store Number(1-100): ").strip()

if not store_num.isdigit():
    raise ValueError("Store number must be numeric (e.g. 1)")

branch = f"{STORE_PREFIX}{int(store_num):03d}"
model_path = os.path.join(MODELS_DIR, f"store_{branch}_model.joblib")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No trained model found for {branch}")

# =====================================================
# 3. LOAD MODEL
# =====================================================
bundle = joblib.load(model_path)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

# =====================================================
# 4. LOAD & PREPARE DATA (MATCH TRAINING)
# =====================================================
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")

df["unit_price"] = (
    df["unit_price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "")
    .astype(float)
)

df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

df["sales"] = df["unit_price"] * df["quantity"]

df = df.dropna(subset=["date", "Branch", "sales"])

# Filter store
store_df = df[df["Branch"] == branch].copy()

if store_df.empty:
    raise ValueError(f"No data found for {branch}")

city = store_df["City"].iloc[0]

# =====================================================
# 5. WEEKLY AGGREGATION
# =====================================================
weekly = (
    store_df
    .set_index("date")
    .resample("W")["sales"]
    .sum()
    .to_frame("weekly_sales")
)

if len(weekly) < 8:
    raise ValueError("Not enough weekly history to forecast")

# =====================================================
# 6. FEATURE ENGINEERING (MATCH TRAINING)
# =====================================================
weekly["lag_1"] = weekly["weekly_sales"].shift(1)
weekly["lag_2"] = weekly["weekly_sales"].shift(2)
weekly["lag_4"] = weekly["weekly_sales"].shift(4)
weekly["ma_4"] = weekly["weekly_sales"].shift(1).rolling(4).mean()

weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
weekly["year"] = weekly.index.year

weekly = weekly.dropna()

# =====================================================
# 7. CURRENT WEEK SALES
# =====================================================
current_week_sales = weekly["weekly_sales"].iloc[-1]

# =====================================================
# 8. FORECAST NEXT WEEK
# =====================================================
X_last = weekly[feature_cols].iloc[[-1]]
next_week_forecast = model.predict(X_last)[0]

# =====================================================
# 9. PERCENTAGE CHANGE
# =====================================================
pct_change = (
    ((next_week_forecast - current_week_sales) / current_week_sales) * 100
    if current_week_sales != 0 else 0
)

# =====================================================
# 10. OUTPUT
# =====================================================
print()
print(f"Store        : {branch} ({city})")
print(f"Current Week : ${current_week_sales:,.2f}")
print(f"Next Week    : ${next_week_forecast:,.2f}")

direction = "increase" if pct_change >= 0 else "decrease"
print(f"Change       : {pct_change:+.2f}% ({direction})\n")

