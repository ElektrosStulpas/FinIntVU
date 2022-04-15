import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# buy when intersect with upper band, sell when already bought and intersect with middle band

#data load up
df_AAPL = yf.download('AAPL', period='10y', interval="1d")
df_INTC = yf.download('INTC', period='1y', interval="1d")
df_MSFT = yf.download('MSFT', period='1y', interval="1d")

#Keltner channel
# either trend trade where follow candles outside bounds
# or flat market trade where bounce between bounds
def Keltner_calc(data, kc_multiplier, timePeriod):
    typicalPrice = (data.High + data.Low + data.Close)/3

    tr1 = data.High - data.Low
    tr2 = abs(data.High - data.Close.shift())
    tr3 = abs(data.Close.shift() - data.Low)
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1).max(axis=1)
    atr = tr.ewm(span=timePeriod).mean()

    kc_middle = typicalPrice.ewm(span=timePeriod).mean()
    print(type(kc_middle))
    kc_upper = kc_middle + (kc_multiplier * atr)
    kc_lower = kc_middle - (kc_multiplier * atr)

    return kc_middle[1:], kc_upper[1:], kc_lower[1:]


def position_calc(df_to_calc, KC_middle, KC_upper, KC_lower):
    position = np.zeros(len(df_to_calc[1:]))
    # since we're doing close, this means we're executing at end of day
    for i, today_price in enumerate(df_to_calc[1:].Close):
        if position[i] == 0: #today position was none
            if today_price > KC_upper[i]:
                position[i+1] = 1 # if price passes upper band, uptrend expected, so we buy
            if today_price < KC_lower[i]:
                position[i+1] = -1 # if price passes lower band, downtrend expected, so we short sell
            continue

        if position[i] == 1: # today position was long
            if today_price < KC_middle[i]:
                position[i+1] = 0 # if we pass middle, sell position as we expect a downtrend
            else:
                position[i+1] = position[i] # continue position
            continue

        if position[i] == -1: # today position was short
            if today_price > KC_middle[i]:
                position[i+1] = 0 # if we pass middle, downtrend ends so we buy to cover
            else:
                position[i+1] = position[i] # continue position
            continue

    return position


def PnL_calc(df_to_calc, position):
    PnL = np.ediff1d(df_to_calc.Close) * position
    return np.cumsum(PnL)


def plot_Keltner(df_to_plot, ticker, kc_shift, period):
    middle, upper, lower = Keltner_calc(df_to_plot, kc_shift, period)
    position = position_calc(df_to_plot, middle, upper, lower)

    print(position)
    shift_position = position
    buy_index = np.where((position[1:] != shift_position[:-1]) & (position[1:] > shift_position[:-1]))[0] + 1 # could add one here
    sell_index = np.where((position[:-1] != shift_position[1:]) & (position[:-1] > shift_position[1:]))[0] + 1

    plt.figure(figsize=(12, 5))
    plt.plot(df_to_plot.Close, color='blue')
    plt.plot(middle, color='black')
    plt.plot(upper, color='black')
    plt.plot(lower, color='black')

    plt.scatter(df_to_plot.index[buy_index], df_to_plot.Close[buy_index], marker='^', color='green')
    plt.scatter(df_to_plot.index[sell_index], df_to_plot.Close[sell_index], marker='v', color='red')

    plt.fill_between(df_to_plot[1:].index, upper, lower, color='grey', alpha=0.3)
    plt.title(ticker + " price with Keltner channel")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.grid()
    plt.show()


# plot_Keltner(df_AAPL, 'AAPL', 2, 20)
# plot_Keltner(df_INTC, 'INTC', 2, 20)
# plot_Keltner(df_MSFT, 'MSFT', 2, 20)



# position = np.zeros(len(df_AAPL[1:]))
# # since we're doing close, this means we're executing at end of day
# for i, today_price in enumerate(df_AAPL[1:].Close):
#     if position[i] == 0: #today position was none
#         if today_price > upper[i]:
#             position[i+1] = 1 # if price passes upper band, uptrend expected, so we buy
#         if today_price < lower[i]:
#             position[i+1] = -1 # if price passes lower band, downtrend expected, so we short sell
#         continue

#     if position[i] == 1: # today position was long
#         if today_price < middle[i]:
#             position[i+1] = 0 # if we pass middle, sell position as we expect a downtrend
#         else:
#             position[i+1] = position[i] # continue position
#         continue

#     if position[i] == -1: # today position was short
#         if today_price > middle[i]:
#             position[i+1] = 0 # if we pass middle, downtrend ends so we buy to cover
#         else:
#             position[i+1] = position[i] # continue position
#         continue

middle, upper, lower = Keltner_calc(df_AAPL, 2, 20)
position = position_calc(df_AAPL, middle, upper, lower)
profit = PnL_calc(df_AAPL, position)
plt.plot(profit)
plt.show()


shift_position = position
buy_indexes = np.where((position[1:] != shift_position[:-1]) & (position[1:] > shift_position[:-1]))[0] + 1 # could add one here
sell_indexes = np.where((position[:-1] != shift_position[1:]) & (position[:-1] > shift_position[1:]))[0] + 1

# series = np.array([0, 1, 1, 1, 0, 0])
# yesterseries = series
# buy_indexes = np.where((series[1:] != yesterseries[:-1]) & (series[1:] > yesterseries[:-1]))[0] + 1
# print(series[:-1])
# print(yesterseries[1:])
# print(buy_indexes)

plot_Keltner(df_AAPL, 'AAPL', 2, 20)