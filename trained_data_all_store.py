import pandas as pd
import joblib
import os
import glob

# =====================================================
# 1. CONFIG
# =====================================================
DATA_PATH = "data.csv"
MODELS_DIR = "models"
OUTPUT_CSV = "store_next_week_forecasts.csv"

# =====================================================
# 2. LOAD & PREPARE DATA (MUST MATCH TRAINING)
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

# Branch → City map (optional but useful)
branch_city = (
    df[["Branch", "City"]]
    .drop_duplicates()
    .set_index("Branch")["City"]
    .to_dict()
)

# =====================================================
# 3. LOAD ALL STORE MODELS
# =====================================================
model_files = glob.glob(os.path.join(MODELS_DIR, "store_*_model.joblib"))

if not model_files:
    raise FileNotFoundError("No trained store models found.")

results = []

# =====================================================
# 4. RUN PREDICTIONS PER STORE
# =====================================================
for model_path in model_files:
    # Extract Branch from filename
    # store_WALM001_model.joblib → WALM001
    branch = os.path.basename(model_path).split("_")[1]
    city = branch_city.get(branch, "Unknown")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    store_df = df[df["Branch"] == branch].copy()
    if store_df.empty:
        continue

    weekly = (
        store_df
        .set_index("date")
        .resample("W")["sales"]
        .sum()
        .to_frame("weekly_sales")
    )

    if len(weekly) < 8:
        continue

    # Feature engineering (MUST MATCH TRAINING)
    weekly["lag_1"] = weekly["weekly_sales"].shift(1)
    weekly["lag_2"] = weekly["weekly_sales"].shift(2)
    weekly["lag_4"] = weekly["weekly_sales"].shift(4)
    weekly["ma_4"] = weekly["weekly_sales"].shift(1).rolling(4).mean()

    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["year"] = weekly.index.year

    weekly = weekly.dropna()
    if weekly.empty:
        continue

    X_last = weekly[feature_cols].iloc[[-1]]
    forecast = model.predict(X_last)[0]

    results.append({
        "Branch": branch,
        "City": city,
        "Last_Observed_Week": weekly.index[-1].date().isoformat(),
        "Forecast_Next_Week_Sales": float(forecast)
    })

# =====================================================
# 5. FORMAT & SAVE RESULTS
# =====================================================
results_df = pd.DataFrame(results)

results_df["Forecast_Next_Week_Sales_$"] = (
    results_df["Forecast_Next_Week_Sales"]
    .astype(float)
    .map(lambda x: f"${x:,.2f}")
)

results_df = results_df.sort_values(
    "Forecast_Next_Week_Sales",
    ascending=False
)

results_df.to_csv(OUTPUT_CSV, index=False)

print("\nStore-level forecasts saved to:", OUTPUT_CSV)
print(results_df[["Branch", "City", "Forecast_Next_Week_Sales_$"]])
