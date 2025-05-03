# ✈️ Air Miles Forecasting Dashboard
### This project is an interactive time series forecasting dashboard developed using Streamlit. It predicts future air travel demand based on monthly air miles data using the Holt-Winters Exponential Smoothing method. I built this project to enhance my knowledge in time series analysis, incorporating concepts and techniques I learned through hands-on experimentation and with the help of a tutorial.

## 🔍 Features
📊 Forecast monthly air miles for up to 3 years (1–36 months)

🛠️ Adjust smoothing parameters (alpha, beta, and gamma) dynamically

🔁 Reset parameters to default values using a one-click button

📈 Visualize historical trends, autocorrelation, partial autocorrelation, and seasonality (monthly and quarterly)

📉 Decompose the time series into trend, seasonal, and residual components

🧪 Evaluate model accuracy using MAE, RMSE, and MAPE

⚡ Real-time interactive dashboard powered by Streamlit

## 📦 Technologies Used
Python – Core programming language

Streamlit – For building the interactive dashboard

Pandas – Data manipulation and preprocessing

Matplotlib – Visualization of plots and charts

Statsmodels – Time series modeling and diagnostics (ACF, PACF, Holt-Winters, decomposition)

Scikit-learn – For computing evaluation metrics (MAE, RMSE, MAPE)

NumPy – Efficient numerical operations

## 📊 Modules Overview
Dataset Preview – Quick glance at raw data and statistical summary

Exploratory Data Analysis – Line plots, ACF, and PACF analysis

Seasonality Analysis – Month and quarter plots + seasonal decomposition

Model Assessment – Train/test evaluation of Holt-Winters forecasting

Forecast Future – Adjustable forecast horizon and smoothing parameters with an interactive forecast plot

## 🔮 Future Work
Implement other forecasting models (e.g., ARIMA, Prophet) for comparison

## 📁 Dataset
The dataset used (airmiles.csv) contains monthly air passenger miles. It is preprocessed to ensure consistent monthly frequency and interpolates missing values if any.

