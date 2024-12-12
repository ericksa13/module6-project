import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# title
st.title("Stock Price Prediction App")

# default csv
DEFAULT_CSV = "TSLA.csv"
default_data = pd.read_csv(DEFAULT_CSV)
default_data['Date'] = pd.to_datetime(default_data['Date'])

# upload csv file
st.header("Upload your stock data")
uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

if uploaded_file is not None:
    # read and display the uploaded csv
    data = pd.read_csv(uploaded_file)
    st.write("Using uploaded dataset.")
else:
    data = default_data.copy()
    st.write("Using default dataset: TSLA.csv.")

st.write(data.head())

# ensure date column is datetime format
data['Date'] = pd.to_datetime(data['Date'])

# choose date to predict
st.header("Select Date to Predict Stock Price")
available_dates = data['Date'].dt.date.unique()
default_date = available_dates[-1]
selected_date = st.selectbox("Choose a date to predict:", available_dates, index=len(available_dates) - 1)

# convert selected date to datetime
selected_date = datetime.strptime(str(selected_date), "%Y-%m-%d")

# find the index of selected date
selected_index = data[data['Date'] == selected_date].index[0]

# data up to the selected date
data_up_to_selected_date = data.iloc[:selected_index]

# data for plot
data_for_plot = data.iloc[:selected_index + 1]

features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
target = 'Close'

data_up_to_selected_date['Next Close'] = data_up_to_selected_date[target].shift(-1)
data_up_to_selected_date.dropna(inplace=True)

# training data
X = data_up_to_selected_date[features]
y = data_up_to_selected_date['Next Close']

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# XGBoost
xgb = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
xgb.fit(X_train, y_train)

# predictions on the selected date
selected_row = data.iloc[selected_index]
X_selected = selected_row[features].values.reshape(1, -1)
X_selected_scaled = scaler.transform(X_selected)

rf_pred = rf.predict(X_selected_scaled)[0]
lr_pred = lr.predict(X_selected_scaled)[0]
dt_pred = dt.predict(X_selected_scaled)[0]
xgb_pred = xgb.predict(X_selected_scaled)[0]

# actual closing price
actual_price = selected_row['Close']

# calculate SMA and EMA
sma_period = 20
ema_period = 20

data_for_plot['SMA'] = data_for_plot['Close'].rolling(window=sma_period).mean()
data_for_plot['EMA'] = data_for_plot['Close'].ewm(span=ema_period, adjust=False).mean()

# previous day using row index
if selected_index > 0:
    prev_day_close = data.iloc[selected_index - 1]['Close']
    price_direction = "up" if actual_price > prev_day_close else "down"
else:
    st.error("No previous data available for the selected date.")
    st.stop()

# predicted the correct direction or not
rf_direction = "up" if rf_pred > prev_day_close else "down"
lr_direction = "up" if lr_pred > prev_day_close else "down"
dt_direction = "up" if dt_pred > prev_day_close else "down"
xgb_direction = "up" if xgb_pred > prev_day_close else "down"

# check if the predictions are correct
rf_correct = "Correct Direction" if rf_direction == price_direction else "Incorrect Direction"
lr_correct = "Correct Direction" if lr_direction == price_direction else "Incorrect Direction"
dt_correct = "Correct Direction" if dt_direction == price_direction else "Incorrect Direction"
xgb_correct = "Correct Direction" if xgb_direction == price_direction else "Incorrect Direction"

# color the text
rf_color = 'green' if rf_direction == price_direction else 'red'
lr_color = 'green' if lr_direction == price_direction else 'red'
dt_color = 'green' if dt_direction == price_direction else 'red'
xgb_color = 'green' if xgb_direction == price_direction else 'red'

# MAE for each model
rf_pred_test = rf.predict(X_test)
lr_pred_test = lr.predict(X_test)
dt_pred_test = dt.predict(X_test)
xgb_pred_test = xgb.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred_test)
lr_mae = mean_absolute_error(y_test, lr_pred_test)
dt_mae = mean_absolute_error(y_test, dt_pred_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred_test)

st.header("Model Predictions")

st.markdown(f"**Random Forest Predicted Closing Price**: ${rf_pred:.2f} <span style='color:{rf_color};'>({rf_correct})</span>", unsafe_allow_html=True)
st.markdown(f"- MAE: {rf_mae:.2f}")

st.markdown(f"**Linear Regression Predicted Closing Price**: ${lr_pred:.2f} <span style='color:{lr_color};'>({lr_correct})</span>", unsafe_allow_html=True)
st.markdown(f"- MAE: {lr_mae:.2f}")

st.markdown(f"**Decision Tree Predicted Closing Price**: ${dt_pred:.2f} <span style='color:{dt_color};'>({dt_correct})</span>", unsafe_allow_html=True)
st.markdown(f"- MAE: {dt_mae:.2f}")

st.markdown(f"**XGBoost Predicted Closing Price**: ${xgb_pred:.2f} <span style='color:{xgb_color};'>({xgb_correct})</span>", unsafe_allow_html=True)
st.markdown(f"- MAE: {xgb_mae:.2f}")

# ensemble prediction (average of all model predictions)
ensemble_pred = np.mean([rf_pred, lr_pred, dt_pred, xgb_pred])
st.markdown(f"**Ensemble Predicted Closing Price**: ${ensemble_pred:.2f}")

st.markdown(f"**Actual Closing Price on {selected_date.date()}: ${actual_price:.2f}**")

# plot the actual stock prices up to the selected date
plt.figure(figsize=(10, 6))
plt.plot(data_for_plot['Date'], data_for_plot['Close'], label='Actual Close Price')

# plot SMA and EMA
plt.plot(data_for_plot['Date'], data_for_plot['SMA'], label=f'{sma_period}-day SMA', linestyle='--')
plt.plot(data_for_plot['Date'], data_for_plot['EMA'], label=f'{ema_period}-day EMA', linestyle='-.')

# predicted closing prices for each model
plt.scatter(selected_date, rf_pred, color='r', label=f"RF Predicted Close on {selected_date.date()} (${rf_pred:.2f})", zorder=5)
plt.scatter(selected_date, lr_pred, color='b', label=f"LR Predicted Close on {selected_date.date()} (${lr_pred:.2f})", zorder=5)
plt.scatter(selected_date, dt_pred, color='y', label=f"DT Predicted Close on {selected_date.date()} (${dt_pred:.2f})", zorder=5)
plt.scatter(selected_date, xgb_pred, color='c', label=f"XGB Predicted Close on {selected_date.date()} (${xgb_pred:.2f})", zorder=5)
plt.scatter(selected_date, ensemble_pred, color='g', label=f"Ensemble Predicted Close on {selected_date.date()} (${ensemble_pred:.2f})", zorder=5)

plt.title("Stock Price and Model Predictions with Technical Indicators")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt)