import pandas as pd
import matplotlib.pyplot as mp
import yfinance as yf
import numpy as np

# useless function since it just adds hassle to managing the read_csv params
# def load_data_from_csv(file_path, header=None, index_col=None):
#     df = pd.read_csv(file_path, header=header, index_col=index_col)
#     return df

# useless to try incorporating ranges of data to plot, way more convenient to just plot the given data
# def plot_daily_data(df_to_plot, col_to_plot, day_start, day_end=0):
#     if day_end == 0:
#         df_to_plot.iloc[day_end:, col_to_plot].plot()
#     else:
#         df_to_plot.iloc[day_start:day_end, col_to_plot].plot()
#     mp.ylabel('Price')
#     #mp.xticks(rotation=90)
#     mp.show()

def plot_and_show(df_to_plot, title, ylabel, xlabel):
    df_to_plot.plot()
    mp.title(title)
    mp.ylabel(ylabel)
    mp.xlabel(xlabel)
    mp.show()

# market generation using normally distributed random variables (https://towardsdatascience.com/how-to-simulate-a-stock-market-with-less-than-10-lines-of-python-code-5de9336114e5)
def generate_market(start_price, trend, volatility, size):
    # np.random.seed(0) #use this for reproducibility
    returns = np.random.normal(loc=trend, scale=volatility, size=size)
    price = start_price*(1+returns).cumprod()
    return pd.DataFrame(price)


#read TSLA tick data and plot it
df = pd.read_csv(r"..\..\TradeData\tick\TSLA_2020-07-01.csv", header=0)
tick_plot_df = df[" Price"][0:50]
plot_and_show(tick_plot_df, "Tick prices of TSLA", 'Price', 'Ticks')
#plot_tick_data(df, 1, 100)

#read MSFT minute data and plot it
df = yf.download(tickers="MSFT", period="5d", interval="1m")
# data["Close"]["2022-02-25"]
minute_plot_df = df.Close[df.index.day == 25]
plot_and_show(minute_plot_df, "Minute close prices of MSFT", 'Price', 'Minutes')

#read AAPL daily data and plot it
df = yf.download('AAPL', period='ytd')
daily_plot_df = df.Close["2021-02-24":"2022-02-24"]
plot_and_show(daily_plot_df, "Daily close prices of AAPL", 'Price', 'Date')


# mu = 0.001
# sigma = 0.01
# start_price = 5
market_array = generate_market(5, 0.001, 0.01, 100)
plot_and_show(market_array, "Generated tick price data", "Price", "Ticks")
