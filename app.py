# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('airmiles.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.set_index('Date')
    df = df.rename(columns={'airmiles': 'y'})
    
    # Force monthly frequency and interpolate missing values
    df = df.asfreq('MS')  
    df['y'] = df['y'].interpolate()  # In case of missing months

    return df


df = load_data()

# Sidebar
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select from Here:", [
    "Dataset Preview", 
    "Exploratory Data Analysis", 
    "Seasonality Analysis", 
    "Model Assessment", 
    "Forecast Future"
])

st.title("Air Miles Forecasting Dashboard")

# 1. Dataset Preview
if options == "Dataset Preview":
    st.subheader("Raw Data")
    st.write(df.head())
    st.write(df.describe())

# 2. Exploratory Data Analysis
elif options == "Exploratory Data Analysis":
    st.subheader("Monthly Air Miles Line Plot")
    st.line_chart(df['y'])

    st.subheader("Autocorrelation Plot (ACF)")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(df['y'], lags=40, ax=ax)
    st.pyplot(fig)

    st.subheader("Partial Autocorrelation Plot (PACF)")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(df['y'], lags=30, ax=ax)
    st.pyplot(fig)

# 3. Seasonality Analysis
elif options == "Seasonality Analysis":
    st.subheader("Monthly Seasonality")
    df_monthly = df.asfreq('MS')
    fig, ax = plt.subplots()
    month_plot(df_monthly['y'], ax=ax)
    st.pyplot(fig)

    st.subheader("Quarterly Seasonality")
    df_quarterly = df.resample('QE').mean().asfreq('QE')
    fig, ax = plt.subplots()
    quarter_plot(df_quarterly['y'], ax=ax)
    st.pyplot(fig)

    st.subheader("Seasonal Decomposition")
    result = seasonal_decompose(df['y'], model='multiplicative', period=12)
    fig = result.plot()
    fig.set_size_inches(10, 8)
    st.pyplot(fig)

# 4. Model Assessment
elif options == "Model Assessment":
    st.subheader("Holt-Winters Model Evaluation")
    
    # Train-test split
    train, test = df.iloc[:-12, :], df.iloc[-12:, :]

    # Model
    model = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=12).fit()
    predictions = model.forecast(steps=12)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train, label='Train')
    ax.plot(test, label='Test')
    ax.plot(predictions, label='Prediction')
    ax.set_title('Train, Test, and Prediction')
    ax.legend()
    st.pyplot(fig)

    # Metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="MAE", value=f"{mae:.2f}")

    with col2:
        st.metric(label="RMSE", value=f"{rmse:.2f}")

    with col3:
        st.metric(label="MAPE (%)", value=f"{mape * 100:.2f}")



# 5. Forecast Future
elif options == "Forecast Future":
    st.subheader("Holt-Winters Forecasting")

    forecast_horizon = st.slider("Select Forecast Horizon (months):", min_value=1, max_value=36, value=12)

    st.write("### Adjust Holt-Winters Smoothing Parameters")

    # Default parameters
    default_alpha = 0.2
    default_beta = 0.1
    default_gamma = 0.1

    # Initialize session state
    if "alpha" not in st.session_state:
        st.session_state.alpha = default_alpha
    if "beta" not in st.session_state:
        st.session_state.beta = default_beta
    if "gamma" not in st.session_state:
        st.session_state.gamma = default_gamma

    # Reset button
    if st.button("Reset Parameters"):
        st.session_state.alpha = default_alpha
        st.session_state.beta = default_beta
        st.session_state.gamma = default_gamma
        st.success("Parameters reset to default values.")

    # Sliders with session state
    alpha = st.slider("Alpha (Level Smoothing):", min_value=0.01, max_value=1.0, value=st.session_state.alpha, key="alpha")
    beta = st.slider("Beta (Trend Smoothing):", min_value=0.01, max_value=1.0, value=st.session_state.beta, key="beta")
    gamma = st.slider("Gamma (Seasonal Smoothing):", min_value=0.01, max_value=1.0, value=st.session_state.gamma, key="gamma")

    # Fit model
    model = ExponentialSmoothing(
        df['y'],
        trend='mul',
        seasonal='mul',
        seasonal_periods=12
    ).fit(
        smoothing_level=alpha,
        smoothing_slope=beta,
        smoothing_seasonal=gamma
    )

    forecast = model.forecast(steps=forecast_horizon)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['y'], label='Historical')
    ax.plot(forecast, label='Forecast')
    ax.set_title(f"Forecast for the Next {forecast_horizon} Months")
    ax.legend()
    st.pyplot(fig)

    # Display forecast
    st.write("**Forecasted Values:**")
    st.write(forecast)
