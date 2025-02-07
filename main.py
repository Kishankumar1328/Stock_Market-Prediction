import streamlit as st
import yfinance as yf
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import altair as alt

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                       torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Streamlit UI
st.title("Stock Price Prediction Dashboard")

# Stock selection
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", "AAPL")

# Fetch historical stock data
data = yf.download(ticker, period="1y", interval="1d")
if data.empty:
    st.error("Invalid ticker or no data available!")
else:
    st.success("Data fetched successfully!")

    # Preprocess data
    stock_data = data[['High', 'Low', 'Open', 'Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(stock_data.values)

    train_window = 7
    def create_sequences(data, tw):
        sequences = []
        L = len(data)
        for i in range(L - tw):
            seq = data[i:i + tw]
            label = data[i + tw][3]
            sequences.append((seq, label))
        return sequences

    data_sequences = create_sequences(scaled_data, train_window)
    train_size = int(len(data_sequences) * 0.8)
    train_data = data_sequences[:train_size]
    test_data = data_sequences[train_size:]

    # Model setup
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, labels in train_data:
            seq = torch.FloatTensor(seq)
            labels = torch.FloatTensor([labels])

            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    st.write(f"Training Completed. Final loss: {epoch_loss / len(train_data):.6f}")

    # Evaluation
    model.eval()
    predictions = []
    actual_values = []
    for seq, label in test_data:
        seq = torch.FloatTensor(seq)
        with torch.no_grad():
            pred = model(seq).item()
        predictions.append(pred)
        actual_values.append(label)

    # Rescale predictions and actual values back to original scale
    predictions = scaler.inverse_transform(np.c_[np.zeros(len(predictions)), np.zeros(len(predictions)), np.zeros(len(predictions)), predictions])[:, 3]
    actual_values = scaler.inverse_transform(np.c_[np.zeros(len(actual_values)), np.zeros(len(actual_values)), np.zeros(len(actual_values)), actual_values])[:, 3]

    # Metrics
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot Predictions vs Actual Prices using Altair
    df_predictions = pd.DataFrame({"Actual Prices": actual_values, "Predicted Prices": predictions})
    df_predictions.reset_index(inplace=True)
    df_predictions.rename(columns={"index": "Time"}, inplace=True)

    chart = alt.Chart(df_predictions).transform_fold(
        ["Actual Prices", "Predicted Prices"],
        as_=["Category", "Price"]
    ).mark_line().encode(
        x="Time:Q",
        y="Price:Q",
        color="Category:N"
    ).properties(
        title="Actual vs Predicted Stock Prices"
    )

    st.altair_chart(chart, use_container_width=True)

    # Display Historical Candle Chart
    data.reset_index(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    candle_chart = alt.Chart(data.tail(60)).mark_rule(color="black").encode(
        x="Date:T", y="Low:Q", y2="High:Q"
    ).properties(
        title="Candle Chart (Last 60 Days)"
    ) + alt.Chart(data.tail(60)).mark_bar().encode(
        x="Date:T", y="Open:Q", y2="Close:Q", color=alt.condition("datum.Open < datum.Close", alt.value("green"), alt.value("red"))
    )

    st.altair_chart(candle_chart, use_container_width=True)

    # Display Stock Data Table
    st.subheader("Historical Stock Data")
    st.write(data.tail())
