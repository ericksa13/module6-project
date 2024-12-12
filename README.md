# Stock Price Prediction App

This is a web-based application built with Streamlit for predicting stock prices using historical data. The app allows users to upload a custom stock dataset or use a default dataset (`TSLA.csv`), select a date, and view predictions for the stock's closing price using various machine learning models. Additionally, it provides insights through technical indicators such as Simple Moving Average (SMA) and Exponential Moving Average (EMA).

---

## Features

- **Upload Stock Dataset**: Users can upload their own CSV file containing historical stock data.
- **Default Dataset**: If no file is uploaded, the app defaults to using the provided `TSLA.csv` dataset.
- **Select Prediction Date**: Users can select any date within the dataset, with the default set to the last available date.
- **Machine Learning Models**:
  - Random Forest
  - Linear Regression
  - Decision Tree
  - XGBoost
- **Ensemble Prediction**: Combines the predictions from all models for a more robust estimate.
- **Direction Accuracy**: Determines if each model correctly predicts whether the stock price will go up or down from the previous day.
- **Technical Indicators**:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
- **Visualizations**:
  - Line plot of historical prices, SMA, and EMA.
  - Scatter points for predicted closing prices of each model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ericksa13/module6-project.git
   cd module6-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Usage

1. **Upload Dataset**:
   - Upload a CSV file containing the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
   - If no file is uploaded, the app will use the default `TSLA.csv` dataset.

2. **Select Prediction Date**:
   - Use the dropdown menu to select the date for which to predict the closing price.
   - The default is the last day in the dataset.

3. **View Predictions**:
   - Predictions from each model are displayed along with their Mean Absolute Error (MAE).
   - The ensemble prediction combines all models.
   - Direction accuracy is shown for each model with green (correct) or red (incorrect) labels.

4. **Technical Indicators**:
   - View the SMA and EMA overlaid on the stock price graph.

---

## File Structure

```
.
├── streamlit_app.py      # Streamlit app code
├── TSLA.csv              # Default dataset
├── requirements.txt      # Python dependencies
└── README.md             # This file
```