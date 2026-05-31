import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import os
import glob
from datetime import datetime

st.set_page_config(page_title="Walmart Store + Product Forecasts", layout="wide")

# Custom CSS to improve layout, remove headers/menus, and make selectbox uneditable
st.markdown(
    """
    <style>
    /* Completely remove the Streamlit header and main menu from the layout */
    header {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }
    
    /* Make the selectbox uneditable (only selectable) */
    .stSelectbox div[data-baseweb="select"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
        user-select: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data.csv")
models_dir = os.path.join(script_dir, "models")


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df["unit_price"] = (
        df["unit_price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["sales"] = df["unit_price"] * df["quantity"]
    return df.dropna(subset=["date", "Branch", "category", "sales"])


@st.cache_data
def load_sales_data() -> pd.DataFrame:
    if not os.path.exists(data_path):
        st.error("data.csv not found in Walmart directory.")
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    return clean_sales_data(df)


@st.cache_resource(max_entries=10)
def load_product_models() -> list[dict]:
    model_files = sorted(glob.glob(os.path.join(models_dir, "product_*_model.joblib")))
    bundles = []

    for model_file in model_files:
        try:
            bundle = joblib.load(model_file)
            category = bundle.get("category") if isinstance(bundle, dict) else None
            if not category:
                category = (
                    os.path.basename(model_file)
                    .replace("product_", "")
                    .replace("_model.joblib", "")
                    .replace("_", " ")
                )
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else ["lag_1", "lag_2", "lag_4", "ma_4", "weekofyear", "year"]
            bundles.append({
                "category": category,
                "model": model,
                "feature_cols": feature_cols,
                "path": model_file,
            })
        except Exception as e:
            st.warning(f"Unable to load product model {model_file}: {e}")
    return bundles


@st.cache_resource(max_entries=10)
def load_store_models() -> dict[str, dict]:
    model_files = sorted(glob.glob(os.path.join(models_dir, "store_*_model.joblib")))
    store_models = {}

    for model_file in model_files:
        try:
            bundle = joblib.load(model_file)
            branch = os.path.basename(model_file).split("_")[1]
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else ["lag_1", "lag_2", "lag_4", "ma_4", "weekofyear", "year"]
            store_models[branch] = {
                "model": model,
                "feature_cols": feature_cols,
                "path": model_file,
            }
        except Exception as e:
            st.warning(f"Unable to load store model {model_file}: {e}")
    return store_models


def is_holiday_week() -> bool:
    today = datetime.now()
    month = today.month
    day = today.day
    holidays = [
        (1, 1),
        (7, 4),
        (11, 27),
        (12, 25),
        (12, 26),
    ]
    for holiday_month, holiday_day in holidays:
        if month == holiday_month and abs(day - holiday_day) <= 3:
            return True
    if month == 11 and 22 <= day <= 28:
        return True
    return False


def get_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.set_index("date")
        .resample("W")
        .agg(
            weekly_sales=("sales", "sum"),
            weekly_quantity=("quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_rating=("rating", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            weekly_transactions=("invoice_id", "count"),
        )
    )
    weekly["week_end"] = weekly.index
    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter
    weekly["year"] = weekly.index.year
    return weekly


def build_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    features = weekly.copy()
    features["lag_1"] = features["weekly_sales"].shift(1)
    features["lag_2"] = features["weekly_sales"].shift(2)
    features["lag_4"] = features["weekly_sales"].shift(4)
    features["ma_4"] = features["weekly_sales"].shift(1).rolling(4).mean()
    return features


def simple_forecast_fallback(weekly: pd.DataFrame) -> float | None:
    if weekly.empty or "weekly_sales" not in weekly:
        return None
    last_sales = weekly["weekly_sales"].dropna()
    if last_sales.empty:
        return None
    if len(last_sales) >= 3:
        return float(last_sales.tail(3).mean())
    return float(last_sales.iloc[-1])


def predict_next_week(model, feature_cols: list[str], weekly: pd.DataFrame) -> float:
    x_last = weekly[feature_cols].iloc[[-1]]
    return max(0.0, float(model.predict(x_last)[0]))


def predict_multiple_weeks(model, feature_cols: list[str], weekly: pd.DataFrame, steps: int = 3) -> list[float]:
    df_temp = weekly.copy()
    predictions = []
    
    for step in range(steps):
        if step > 0:
            last_idx = df_temp.index[-1]
            next_idx = last_idx + pd.Timedelta(days=7)
            
            new_row = df_temp.iloc[[-1]].copy()
            new_row.index = [next_idx]
            
            new_row["weekly_sales"] = predictions[-1]
            df_temp = pd.concat([df_temp, new_row])
            
            df_temp["lag_1"] = df_temp["weekly_sales"].shift(1)
            df_temp["lag_2"] = df_temp["weekly_sales"].shift(2)
            df_temp["lag_4"] = df_temp["weekly_sales"].shift(4)
            df_temp["ma_4"] = df_temp["weekly_sales"].shift(1).rolling(4).mean()
            
            df_temp["weekofyear"] = df_temp.index.isocalendar().week.astype(int)
            df_temp["month"] = df_temp.index.month
            df_temp["quarter"] = df_temp.index.quarter
            df_temp["year"] = df_temp.index.year
            
        x_last = df_temp[feature_cols].iloc[[-1]]
        pred = max(0.0, float(model.predict(x_last)[0]))
        predictions.append(pred)
        
    return predictions


def fallback_multiple_weeks(weekly: pd.DataFrame, steps: int = 3) -> list[float]:
    val = simple_forecast_fallback(weekly)
    if val is None:
        return [0.0] * steps
    return [val] * steps



def format_currency(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"${value:,.2f}"


def branch_to_store_name(branch: str, city: str) -> str:
    try:
        store_number = int(branch.replace("M", ""))
        return f"Store {store_number} - {city}"
    except Exception:
        return f"{branch} - {city}"


# --- Page layout ---
st.title("🛒 Walmart Store + Product Forecasts")
st.markdown(
    "Select a store from the dropdown, then review the store-level forecast and product-level category predictions for that store."
)

# --- Load data and models ---
df = load_sales_data()
product_models = load_product_models()
product_model_map = {bundle["category"]: bundle for bundle in product_models}
store_models = load_store_models()

if df.empty:
    st.error("Unable to load the Walmart sales data. Please check data.csv.")
    st.stop()

store_meta = (
    df[["Branch", "City"]]
    .drop_duplicates()
    .assign(store_number=lambda d: d["Branch"].str.replace("M", "").astype(int))
    .sort_values("store_number")
    .reset_index(drop=True)
)
store_meta["display"] = store_meta.apply(lambda row: branch_to_store_name(row["Branch"], row["City"]), axis=1)

col_select, _ = st.columns([1, 2])
with col_select:
    selected_store_display = st.selectbox(
        "Select Store",
        options=store_meta["display"].tolist(),
        index=0,
    )
selected_store_branch = store_meta.loc[store_meta["display"] == selected_store_display, "Branch"].iloc[0]
selected_store_city = store_meta.loc[store_meta["display"] == selected_store_display, "City"].iloc[0]
selected_store_number = int(selected_store_branch.replace("M", ""))

store_df = df[df["Branch"] == selected_store_branch].copy()
store_sales_total = store_df["sales"].sum()
product_groups = (
    store_df.groupby("category", observed=False)["sales"].sum().reset_index().rename(columns={"sales": "total_sales"})
)
product_groups["model_available"] = product_groups["category"].isin(product_model_map)
product_groups["total_sales_formatted"] = product_groups["total_sales"].map(format_currency)

store_model_bundle = store_models.get(selected_store_branch)
store_weekly = get_weekly_sales(store_df) if not store_df.empty else pd.DataFrame()
store_forecasts = []
store_forecast_note = ""
store_dates = []

if store_model_bundle is not None and not store_weekly.empty:
    feature_weekly = build_weekly_features(store_weekly)
    available_features = feature_weekly.dropna()
    if not available_features.empty:
        try:
            store_forecasts = predict_multiple_weeks(store_model_bundle["model"], store_model_bundle["feature_cols"], available_features, steps=3)
            store_forecast_note = "Store forecast based on the trained model."
        except Exception as e:
            store_forecasts = fallback_multiple_weeks(store_weekly, steps=3)
            store_forecast_note = f"Model failed, using fallback from recent history: {e}"
    else:
        store_forecasts = fallback_multiple_weeks(store_weekly, steps=3)
        store_forecast_note = "Not enough store history for full forecast features; using recent weekly sales as fallback."
else:
    store_forecasts = fallback_multiple_weeks(store_weekly, steps=3) if not store_weekly.empty else []
    store_forecast_note = "No store-level model found for this selection; using recent weekly sales as fallback." if store_forecasts else "No store-level forecast available."

store_forecast = store_forecasts[0] if len(store_forecasts) > 0 else None
store_forecast_delta = 0.0
if store_forecast is not None and not store_weekly.empty:
    last_store_week = store_weekly["weekly_sales"].iloc[-1]
    store_forecast_delta = ((store_forecast - last_store_week) / last_store_week * 100) if last_store_week != 0 else 0

if not store_weekly.empty:
    last_week_end = store_weekly.index[-1]
    store_dates = [
        (last_week_end + pd.Timedelta(days=7)).date().isoformat(),
        (last_week_end + pd.Timedelta(days=14)).date().isoformat(),
        (last_week_end + pd.Timedelta(days=21)).date().isoformat()
    ]
else:
    store_dates = ["Week 1", "Week 2", "Week 3"]


# --- Store and Product metrics ---

col1, col2 = st.columns([2, 1])
with col1:
    st.metric("Selected Store", selected_store_display)
    st.metric("Total Store Sales", format_currency(store_sales_total))
with col2:
    st.metric("Product Categories in Store", len(product_groups))
    st.metric("Product Categories Modeled", int(product_groups["model_available"].sum()))

# Initialize session state for showing extra weeks
if "show_extra_weeks" not in st.session_state:
    st.session_state.show_extra_weeks = False

# Action bar to show/hide more weeks
col_btn, _ = st.columns([1, 2])
with col_btn:
    if st.button(
        "📅 Show More 2 Weeks" if not st.session_state.show_extra_weeks else "📅 Show Next Week Only",
        width="stretch"
    ):
        st.session_state.show_extra_weeks = not st.session_state.show_extra_weeks
        st.rerun()

if store_forecast is not None:
    if st.session_state.show_extra_weeks and len(store_forecasts) >= 3:
        st.subheader("📈 Store-Level Next 3-Week Forecast")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label=f"1st Week ({store_dates[0]})",
                value=format_currency(store_forecasts[0]),
                delta=f"{store_forecast_delta:.2f}%"
            )
        with m_col2:
            wk2_delta = ((store_forecasts[1] - store_forecasts[0]) / store_forecasts[0] * 100) if store_forecasts[0] != 0 else 0
            st.metric(
                label=f"2nd Week ({store_dates[1]})",
                value=format_currency(store_forecasts[1]),
                delta=f"{wk2_delta:.2f}% (vs Wk 1)"
            )
        with m_col3:
            wk3_delta = ((store_forecasts[2] - store_forecasts[1]) / store_forecasts[1] * 100) if store_forecasts[1] != 0 else 0
            st.metric(
                label=f"3rd Week ({store_dates[2]})",
                value=format_currency(store_forecasts[2]),
                delta=f"{wk3_delta:.2f}% (vs Wk 2)"
            )
    else:
        st.subheader("📈 Store-Level Next Week Forecast")
        st.metric("Forecast", format_currency(store_forecast), delta=f"{store_forecast_delta:.2f}%")
    
    if store_forecast_note:
        st.info(store_forecast_note)
else:
    with st.expander("Store-level forecast status"):
        st.write(store_forecast_note)

st.markdown("---")

st.subheader("Products and Category Forecasts for This Store")
product_rows = []
for _, row in product_groups.sort_values("total_sales", ascending=False).iterrows():
    category = row["category"]
    total_sales = float(row["total_sales"])
    bundle = product_model_map.get(category)
    forecasts = []
    previous_week_date = "-"
    previous_week_sales = None
    current_week_date = "-"
    current_week_sales = None
    next_week_date = "-"
    next_week_plus_1_date = "-"
    next_week_plus_2_date = "-"

    category_df = store_df[store_df["category"] == category].copy()
    category_weekly = get_weekly_sales(category_df)
    
    if len(category_weekly) >= 1:
        current_week_sales = float(category_weekly["weekly_sales"].iloc[-1])
        current_week_date = category_weekly.index[-1].date().isoformat()
        next_week_date = (category_weekly.index[-1] + pd.Timedelta(days=7)).date().isoformat()
        next_week_plus_1_date = (category_weekly.index[-1] + pd.Timedelta(days=14)).date().isoformat()
        next_week_plus_2_date = (category_weekly.index[-1] + pd.Timedelta(days=21)).date().isoformat()
        if len(category_weekly) >= 2:
            previous_week_sales = float(category_weekly["weekly_sales"].iloc[-2])
            previous_week_date = category_weekly.index[-2].date().isoformat()
        else:
            previous_week_date = (category_weekly.index[-1] - pd.Timedelta(days=7)).date().isoformat()
            previous_week_sales = current_week_sales
    elif not category_df.empty:
        last_date = category_df["date"].max()
        if pd.notna(last_date):
            current_week_date = last_date.date().isoformat()
            current_week_sales = float(category_df["sales"].sum())
            next_week_date = (last_date + pd.Timedelta(days=7)).date().isoformat()
            next_week_plus_1_date = (last_date + pd.Timedelta(days=14)).date().isoformat()
            next_week_plus_2_date = (last_date + pd.Timedelta(days=21)).date().isoformat()
            previous_week_date = (last_date - pd.Timedelta(days=7)).date().isoformat()
            previous_week_sales = current_week_sales

    feature_weekly = build_weekly_features(category_weekly)
    available_features = feature_weekly.dropna()
    if bundle is not None and not available_features.empty:
        try:
            forecasts = predict_multiple_weeks(bundle["model"], bundle["feature_cols"], available_features, steps=3)
        except Exception:
            forecasts = fallback_multiple_weeks(category_weekly, steps=3)
    else:
        forecasts = fallback_multiple_weeks(category_weekly, steps=3)

    forecast = forecasts[0] if len(forecasts) > 0 else None
    forecast_plus_1 = forecasts[1] if len(forecasts) > 1 else None
    forecast_plus_2 = forecasts[2] if len(forecasts) > 2 else None

    current_week_delta = None
    if current_week_sales is not None and previous_week_sales is not None:
        if previous_week_sales != 0:
            current_week_delta = ((current_week_sales - previous_week_sales) / previous_week_sales * 100)
        else:
            current_week_delta = 0.0

    forecast_delta = None
    if forecast is not None and current_week_sales is not None:
        if current_week_sales != 0:
            forecast_delta = ((forecast - current_week_sales) / current_week_sales * 100)
        else:
            forecast_delta = 0.0

    forecast_plus_1_delta = None
    if forecast_plus_1 is not None and forecast is not None:
        if forecast != 0:
            forecast_plus_1_delta = ((forecast_plus_1 - forecast) / forecast * 100)
        else:
            forecast_plus_1_delta = 0.0

    forecast_plus_2_delta = None
    if forecast_plus_2 is not None and forecast_plus_1 is not None:
        if forecast_plus_1 != 0:
            forecast_plus_2_delta = ((forecast_plus_2 - forecast_plus_1) / forecast_plus_1 * 100)
        else:
            forecast_plus_2_delta = 0.0

    product_rows.append({
        "Category": category,
        "Total Sales": total_sales,
        "Total Sales Display": format_currency(total_sales),
        "Previous Week Date": previous_week_date,
        "Previous Week Sales": format_currency(previous_week_sales) if previous_week_sales is not None else "-",
        "Current Week Date": current_week_date,
        "Current Week Sales": format_currency(current_week_sales) if current_week_sales is not None else "-",
        "Current Week % Change": f"{current_week_delta:.2f}%" if current_week_delta is not None else "-",
        "Next 1st Week Date": next_week_date,
        "Next 1st Week Forecast": format_currency(forecast) if forecast is not None else "-",
        "Next 1st Week % Change": f"{forecast_delta:.2f}%" if forecast_delta is not None else "-",
        "Next 2nd Week Date": next_week_plus_1_date,
        "Next 2nd Week Forecast": format_currency(forecast_plus_1) if forecast_plus_1 is not None else "-",
        "Next 2nd Week % Change": f"{forecast_plus_1_delta:.2f}%" if forecast_plus_1_delta is not None else "-",
        "Next 3rd Week Date": next_week_plus_2_date,
        "Next 3rd Week Forecast": format_currency(forecast_plus_2) if forecast_plus_2 is not None else "-",
        "Next 3rd Week % Change": f"{forecast_plus_2_delta:.2f}%" if forecast_plus_2_delta is not None else "-",
    })

product_report = pd.DataFrame(product_rows)
if product_report.empty:
    st.warning("No product categories were found for this store.")
    st.markdown("---")
    st.subheader("All Product Categories in Selected Store")
    st.dataframe(
        product_groups[["category", "total_sales_formatted"]]
        .rename(columns={"category": "Category", "total_sales_formatted": "Total Sales"}),
        width="stretch"
    )
else:
    if st.session_state.get("show_extra_weeks", False):
        display_cols = [
            "Category",
            "Total Sales Display",
            "Previous Week Date",
            "Previous Week Sales",
            "Current Week Date",
            "Current Week Sales",
            "Current Week % Change",
            "Next 1st Week Date",
            "Next 1st Week Forecast",
            "Next 1st Week % Change",
            "Next 2nd Week Date",
            "Next 2nd Week Forecast",
            "Next 2nd Week % Change",
            "Next 3rd Week Date",
            "Next 3rd Week Forecast",
            "Next 3rd Week % Change",
        ]
    else:
        display_cols = [
            "Category",
            "Total Sales Display",
            "Previous Week Date",
            "Previous Week Sales",
            "Current Week Date",
            "Current Week Sales",
            "Current Week % Change",
            "Next 1st Week Date",
            "Next 1st Week Forecast",
            "Next 1st Week % Change",
        ]
        
    st.dataframe(
        product_report[display_cols].rename(columns={"Total Sales Display": "Total Sales"})
    )
    
    chart_data = product_report.copy()
    chart_data["Next 1st Week Forecast Numeric"] = chart_data["Next 1st Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("All Product Categories in Selected Store")
        st.dataframe(
            product_groups[["category", "total_sales_formatted"]]
            .rename(columns={"category": "Category", "total_sales_formatted": "Total Sales"}),
            width="stretch"
        )
    with col2:
        if st.session_state.get("show_extra_weeks", False):
            chart_data["Next 2nd Week Forecast Numeric"] = chart_data["Next 2nd Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
            chart_data["Next 3rd Week Forecast Numeric"] = chart_data["Next 3rd Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
            chart_data_clean = chart_data.dropna(subset=["Next 1st Week Forecast Numeric", "Next 2nd Week Forecast Numeric", "Next 3rd Week Forecast Numeric"])
            
            if not chart_data_clean.empty:
                st.subheader("Forecast Distributions")
                tab1, tab2, tab3 = st.tabs([
                    f"Next 1st Week ({store_dates[0]})",
                    f"Next 2nd Week ({store_dates[1]})",
                    f"Next 3rd Week ({store_dates[2]})"
                ])
                with tab1:
                    fig1 = alt.Chart(chart_data_clean).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="Next 1st Week Forecast Numeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Category")),
                        tooltip=["Category", "Next 1st Week Forecast Numeric"]
                    ).properties(
                        title="Next 1st Week Forecast Distribution"
                    )
                    st.altair_chart(fig1, width="stretch")
                
                with tab2:
                    fig2 = alt.Chart(chart_data_clean).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="Next 2nd Week Forecast Numeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Category")),
                        tooltip=["Category", "Next 2nd Week Forecast Numeric"]
                    ).properties(
                        title="Next 2nd Week Forecast Distribution"
                    )
                    st.altair_chart(fig2, width="stretch")
                    
                with tab3:
                    fig3 = alt.Chart(chart_data_clean).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="Next 3rd Week Forecast Numeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Category")),
                        tooltip=["Category", "Next 3rd Week Forecast Numeric"]
                    ).properties(
                        title="Next 3rd Week Forecast Distribution"
                    )
                    st.altair_chart(fig3, width="stretch")
        else:
            chart_data_clean = chart_data.dropna(subset=["Next 1st Week Forecast Numeric"])
            if not chart_data_clean.empty:
                st.subheader("Forecast Distributions")
                fig1 = alt.Chart(chart_data_clean).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="Next 1st Week Forecast Numeric", type="quantitative"),
                    color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Category")),
                    tooltip=["Category", "Next 1st Week Forecast Numeric"]
                ).properties(
                    title="Next 1st Week Forecast Distribution"
                )
                st.altair_chart(fig1, width="stretch")

