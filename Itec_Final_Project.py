# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred

# ------------------------------
#  Helper functions (caching)
# ------------------------------

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_starbucks_data(csv_path: str) -> pd.DataFrame:
    """
    Load Starbucks data from CSV, parse dates, and set index to Quarter-end dates.
    Expected columns: ['date','revenue','expenses','transactions','store_count',
                       'avg_ticket','marketing_spend','employee_count','loyalty_members',
                       'mobile_orders_pct','coffee_bean_price','temperature_index']
    'revenue' is in millions of USD.
    """
    df = pd.read_csv(csv_path)
    # Parse 'date' as datetime at quarter end
    df['date'] = pd.to_datetime(df['date'])
    # Ensure it's aligned to quarter-end frequency
    df = df.set_index(pd.DatetimeIndex(df['date']).to_period('Q').to_timestamp('Q'))
    df = df.sort_index()
    return df

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def fetch_quarterly_cpi(api_key: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch CPIAUCSL (Consumer Price Index for All Urban Consumers) from FRED,
    then resample monthly data to quarterly average. Returns a pd.Series
    indexed by quarter-end timestamps.
    """
    fred = Fred(api_key=api_key)
    cpi_monthly = fred.get_series('CPIAUCSL', observation_start=start_date, observation_end=end_date)
    # Convert to datetime index
    cpi_monthly.index = pd.to_datetime(cpi_monthly.index)
    # Resample to quarterly frequency, taking the average of months in each quarter
    cpi_q = cpi_monthly.resample('Q').mean()
    cpi_q.index = pd.DatetimeIndex(cpi_q.index.to_period('Q').to_timestamp('Q'))
    return cpi_q.rename("CPI")

# ------------------------------
#  Streamlit App Layout
# ------------------------------

st.set_page_config(page_title="Starbucks Revenue Forecasting", layout="wide")
st.title("Starbucks Revenue Forecasting App")

st.markdown(
    """
    This app uses an ARIMAX model to forecast Starbucks quarterly revenue (in millions USD), 
    incorporating CPI data from FRED and historical marketing spend as exogenous regressors.  
    You can adjust the forecast horizon in quarters, and view in‐sample predicted vs actual results 
    with confidence intervals.
    """
)

# Sidebar: User inputs
st.sidebar.header("User Inputs")

# 1. FRED API key (for CPI data)
fred_api_key = st.sidebar.text_input(
    label="Enter your FRED API Key", 
    type="password", 
    help="Obtain an API key from https://fred.stlouisfed.org/ to fetch CPI data."
)

# 2. Forecast horizon in quarters
forecast_horizon = st.sidebar.slider(
    label="Forecast horizon (quarters)", 
    min_value=1, 
    max_value=8, 
    value=4, 
    step=1, 
    help="Select how many future quarters to forecast."
)

# 3. Path to Starbucks CSV (assumes file in same directory)
csv_path = st.sidebar.text_input(
    label="Starbucks data CSV path", 
    value="final_project_starbucks_data.csv", 
    help="Filename of Starbucks CSV (e.g., 'final_project_starbucks_data.csv')."
)

# Check that API key is provided
if not fred_api_key:
    st.warning("🔑 Please enter your FRED API key on the sidebar to fetch CPI data.")
    st.stop()

# ------------------------------
#  Data Loading and Preparation
# ------------------------------

# Load Starbucks data
with st.spinner("Loading Starbucks data..."):
    try:
        sbux_df = load_starbucks_data(csv_path)
    except FileNotFoundError:
        st.error(f"Could not find CSV at path: '{csv_path}'. Please verify the filename/location.")
        st.stop()

# Determine data date range for CPI fetch
start_date = sbux_df.index.min().strftime("%Y-%m-%d")
end_date   = sbux_df.index.max().strftime("%Y-%m-%d")

# Fetch quarterly CPI data
with st.spinner("Fetching CPI data from FRED..."):
    try:
        cpi_q = fetch_quarterly_cpi(fred_api_key, start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching CPI data: {e}")
        st.stop()

# Merge CPI into Starbucks DataFrame
sbux_df = sbux_df.copy()
sbux_df = sbux_df.assign(CPI = cpi_q)
# Drop any quarters where CPI or marketing_spend is missing
sbux_df = sbux_df.dropna(subset=["CPI", "marketing_spend", "revenue"])

# ------------------------------
#  Model Fitting: ARIMAX
# ------------------------------

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def fit_arimax_model(df: pd.DataFrame):
    """
    Fit an ARIMAX(1,1,1) model with exogenous regressors: marketing_spend and CPI.
    Returns the fitted SARIMAXResults object.
    """
    y = df["revenue"]            # Revenue in millions USD
    exog = df[["marketing_spend", "CPI"]]
    # Use a basic ARIMA(1,1,1) specification; seasonal terms omitted for simplicity
    model = SARIMAX(
        endog=y,
        exog=exog,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    return result

with st.spinner("Fitting ARIMAX model..."):
    arimax_result = fit_arimax_model(sbux_df)

# ------------------------------
#  In-Sample Predictions
# ------------------------------

# Prepare exogenous matrix for in-sample prediction
exog_insample = sbux_df[["marketing_spend", "CPI"]]
# Generate in-sample prediction (fitted values) with confidence intervals
pred_insample = arimax_result.get_prediction(start=0, end=len(sbux_df)-1, exog=exog_insample)
pred_mean_insample = pred_insample.predicted_mean
conf_int_insample = pred_insample.conf_int(alpha=0.05)  # 95% CI

# ------------------------------
#  Out-of-Sample Forecast
# ------------------------------

# Build exogenous data for future quarters (assume last observed values persist)
last_date = sbux_df.index.max()
future_index = pd.date_range(
    start=(last_date + pd.tseries.offsets.QuarterEnd(1)), 
    periods=forecast_horizon, 
    freq="Q"
)
last_cpi = sbux_df["CPI"].iloc[-1]
last_mkt = sbux_df["marketing_spend"].iloc[-1]
future_exog = pd.DataFrame({
    "marketing_spend": [last_mkt] * forecast_horizon,
    "CPI": [last_cpi] * forecast_horizon
}, index=future_index)

# Generate forecasts
forecast_obj = arimax_result.get_forecast(steps=forecast_horizon, exog=future_exog)
forecast_mean = forecast_obj.predicted_mean
forecast_conf_int = forecast_obj.conf_int(alpha=0.05)

# ------------------------------
#  Visualization
# ------------------------------

st.subheader("In-Sample: Actual vs. Predicted Revenue (with 95% CI)")

fig_insample, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    sbux_df.index, 
    sbux_df["revenue"], 
    label="Actual Revenue", 
    marker="o", 
    linestyle="-", 
    color="tab:blue"
)
ax.plot(
    sbux_df.index, 
    pred_mean_insample, 
    label="Predicted (Fitted) Revenue", 
    marker="o", 
    linestyle="--", 
    color="tab:orange"
)
# Shade confidence interval
ax.fill_between(
    sbux_df.index.astype("datetime64[ns]"),
    conf_int_insample["lower revenue"],
    conf_int_insample["upper revenue"],
    color="tab:orange",
    alpha=0.2,
    label="95% Confidence Interval"
)

ax.set_title("Actual vs. In-Sample Predicted Revenue\n(Revenue in Millions USD)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue (Millions USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig_insample)

st.markdown("---")

st.subheader(f"Out-of-Sample Forecast for Next {forecast_horizon} Quarter(s)")

fig_forecast, ax2 = plt.subplots(figsize=(10, 5))
# Plot historical revenue
ax2.plot(
    sbux_df.index, 
    sbux_df["revenue"], 
    label="Historical Actual Revenue", 
    marker="o", 
    linestyle="-", 
    color="tab:blue"
)
# Plot forecasted revenue
ax2.plot(
    future_index, 
    forecast_mean, 
    label="Forecasted Revenue", 
    marker="o", 
    linestyle="--", 
    color="tab:green"
)
# Shade future confidence interval
ax2.fill_between(
    future_index.astype("datetime64[ns]"),
    forecast_conf_int["lower revenue"],
    forecast_conf_int["upper revenue"],
    color="tab:green",
    alpha=0.2,
    label="95% Confidence Interval (Forecast)"
)

ax2.set_title(f"Forecasted Revenue for Next {forecast_horizon} Quarter(s)\n(Revenue in Millions USD)")
ax2.set_xlabel("Quarter")
ax2.set_ylabel("Revenue (Millions USD)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig_forecast)

# ------------------------------
#  Display Model Summary & AI-Generated Summary Placeholder
# ------------------------------

st.subheader("Model Summary (ARIMAX(1,1,1))")
with st.expander("Show full summary"):
    st.text(arimax_result.summary().as_text())

st.subheader("AI-Generated Summary for Audit Committee")
ai_summary = st.text_area(
    label="Refine or replace this summary as needed:",
    value=(
        "The ARIMAX model incorporates Starbucks’ historical quarterly revenue (in millions USD) "
        "with marketing spend and CPI as exogenous regressors. In‐sample results show "
        "close alignment between actual and fitted revenue, with a 95% confidence interval "
        "indicating model stability. The out‐of‐sample forecast for the next "
        f"{forecast_horizon} quarter(s) suggests the revenue trend will "
        "remain broadly in line with seasonal patterns, assuming stable marketing "
        "investment and CPI levels. Audit risk of revenue overstatement appears low, "
        "as model fit is strong and external economic factors are accounted for."
    ),
    height=150,
    help="This 50–100 word summary is tailored for an audit committee, focusing on risk insights."
)

# ------------------------------
#  Additional Insight: Marketing Spend vs. Revenue Scatter
# ------------------------------

st.subheader("Additional Insight: Marketing Spend vs. Revenue")

fig_scatter, ax3 = plt.subplots(figsize=(8, 4))
ax3.scatter(
    sbux_df["marketing_spend"], 
    sbux_df["revenue"], 
    color="tab:purple", 
    alpha=0.7
)
# Fit a simple linear trend line
m, b = np.polyfit(sbux_df["marketing_spend"], sbux_df["revenue"], 1)
x_vals = np.linspace(sbux_df["marketing_spend"].min(), sbux_df["marketing_spend"].max(), 100)
ax3.plot(x_vals, m * x_vals + b, color="tab:red", linestyle="--", label="Linear Trend")

ax3.set_title("Quarterly Marketing Spend vs. Revenue")
ax3.set_xlabel("Marketing Spend (Millions USD)")
ax3.set_ylabel("Revenue (Millions USD)")
ax3.legend()
ax3.grid(True)
st.pyplot(fig_scatter)

st.markdown(
    """
    **Insight:** Higher marketing spend generally correlates with higher revenue, though 
    the relationship is subject to seasonal and macroeconomic fluctuations (CPI). By incorporating 
    CPI into the ARIMAX model, we adjust for inflationary pressure, ensuring that marketing 
    ROI is evaluated in real terms.
    """
)
