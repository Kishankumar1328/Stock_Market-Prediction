import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import openai

# Set up your OpenAI GPT API key (ensure it's kept secure)
openai.api_key = "Your GPT API KEY"

st.title("Stock Dashboard")

# Sidebar input for ticker and date range
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, MSFT)")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Validate input and handle errors
if not ticker:
    st.warning("Please enter a valid ticker symbol.")
elif start_date >= end_date:
    st.warning("End date must be after the start date.")
else:
    # Download data
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # Check if data is empty
    if data.empty:
        st.warning(f"No data available for {ticker} between {start_date} and {end_date}.")
    else:
        # Plot data (using 'Close' as a fallback for missing 'Adj Close')
        fig = px.line(data, x=data.index, y=data.get('Adj Close', data['Close']), title=f"{ticker} Price Movement")
        st.plotly_chart(fig, use_container_width=True)

        # Pricing Data Section
        st.header("Price Movement")
        data['%change'] = data['Close'].pct_change()  # Use Close for % change calculation
        data.dropna(inplace=True)
        st.dataframe(data)  # Display the data in a structured format
        annual_returns = data['%change'].mean() * 252 * 100
        st.write("Annual Return is", f"{annual_returns:.2f}%")
        stdev = np.std(data['%change']) * np.sqrt(252)
        st.write("Standard Deviation is", f"{stdev * 100:.2f}%")
        st.write('Risk Adj. Return is', f"{annual_returns / stdev:.2f}")

        # Fundamental Data Section
        st.header("Fundamental Data")

        # Use a secure method to retrieve your API key, such as environment variables
        key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your actual Alpha Vantage API key
        fd = FundamentalData(key, output_format='pandas')

        st.subheader('Balance Sheet')
        try:
            balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
            bs = balance_sheet.T[2:]
            bs.columns = list(balance_sheet.T.iloc[0])
            st.write(bs)
        except Exception as e:
            st.warning(f"Error fetching balance sheet: {e}")

        st.subheader('Income Statement')
        try:
            income_statement = fd.get_income_statement_annual(ticker)[0]
            is1 = income_statement.T[2:]
            is1.columns = list(income_statement.T.iloc[0])
            st.write(is1)
        except Exception as e:
            st.warning(f"Error fetching income statement: {e}")

        st.subheader('Cash Flow Statement')
        try:
            cash_flow = fd.get_cash_flow_annual(ticker)[0]
            cf = cash_flow.T[2:]
            cf.columns = list(cash_flow.T.iloc[0])
            st.write(cf)
        except Exception as e:
            st.warning(f"Error fetching cash flow statement: {e}")

        # User input for market questions
        user_input = st.text_input("Ask me anything about stocks or the market.")

        # Query ChatGPT for response
        if user_input:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=user_input,
                    max_tokens=150
                )
                st.write("ChatGPT's Response:")
                st.write(response['choices'][0]['text'].strip())
            except Exception as e:
                st.warning(f"Error querying ChatGPT: {e}")
