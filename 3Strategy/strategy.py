import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

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
    kc_upper = kc_middle + (kc_multiplier * atr)
    kc_lower = kc_middle - (kc_multiplier * atr)

    return kc_middle[1:], kc_upper[1:], kc_lower[1:]


def position_calc(df_to_calc, KCchannels, stop_loss, take_profit):
    KC_middle, KC_upper, KC_lower = KCchannels
    position = np.zeros(len(df_to_calc))
    SLTP_indices = []
    # since we're doing close, this means we're executing at end of day
    for i, today_price in enumerate(df_to_calc[1:].Close):
        if position[i] == 0: #today position was none
            if today_price > KC_upper[i]:
                position[i+1] = 1 # if price passes upper band, uptrend expected, so we buy
                entry_price = today_price
            if today_price < KC_lower[i]:
                position[i+1] = -1 # if price passes lower band, downtrend expected, so we short sell
                entry_price = today_price
            continue

        if position[i] == 1: # today position was long
            if today_price < KC_middle[i]:
                position[i+1] = 0 # if we pass middle, sell position as we expect a downtrend
            elif today_price >= entry_price * (1+take_profit) or today_price <= entry_price * (1-stop_loss):
                position[i+1] = 0 # sell as stop_loss or take_profit was triggered
                SLTP_indices.append(i)
                entry_price = 0
            else:
                position[i+1] = position[i] # continue position
            continue

        if position[i] == -1: # today position was short
            if today_price > KC_middle[i]:
                position[i+1] = 0 # if we pass middle, downtrend ends so we buy to cover
            elif today_price <= entry_price * (1-take_profit) or today_price >= entry_price * (1+stop_loss):
                position[i+1] = 0
                SLTP_indices.append(i)
                entry_price = 0
            else:
                position[i+1] = position[i] # continue position
            continue

    return position[1:], np.array(SLTP_indices, dtype=np.int64)+1


def PnL_calc(df_to_calc, position, buy_indices, sell_indice, fees):
    PnL = (df_to_calc.Close).diff()[1:] * position
    #adjust sell and buy indexes to day before since actual position change will be for tommorow, but fees are on day close before
    PnL.iloc[buy_indices-1] -= fees
    PnL.iloc[sell_indice-1] -= fees
    return PnL


def plot_Keltner_with_Profit(df_to_plot, ticker, KCchannels, signal_indices, profit):
    middle, upper, lower = KCchannels
    buy_index, sell_index, SLTP_index = signal_indices
    fig, ax = plt.subplots(2, figsize=(12, 5), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    close_line, = ax[0].plot(df_to_plot.Close, color='blue', label="Close price")
    ax[0].plot(middle, color='black')
    ax[0].plot(upper, color='black')
    ax[0].plot(lower, color='black')
    ax[0].fill_between(df_to_plot[1:].index, upper, lower, color='grey', alpha=0.3)

    buy_marker = ax[0].scatter(df_to_plot.index[buy_index], df_to_plot.Close[buy_index], marker='^', color='green', label='Buy signal')
    sell_marker = ax[0].scatter(df_to_plot.index[sell_index], df_to_plot.Close[sell_index], marker='v', color='red', label='Sell signal')
    SLTP_marker = ax[0].scatter(df_to_plot.index[SLTP_index], df_to_plot.Close[SLTP_index], marker='X', color='purple', label='SLTP signal', alpha=0.5)

    ax[0].set_title(ticker + " price with Keltner channel")
    ax[0].set_ylabel("Price")
    ax[0].grid()
    ax[0].legend(handles = [close_line, buy_marker, sell_marker, SLTP_marker])

    ax[1].plot(profit, color='black')
    ax[1].set_ylabel("Profit")
    ax[1].set_xlabel("Time")
    ax[1].grid()
    plt.show()

def Keltner_strat(df_to_strat, keltner_multiplier, keltner_time):
    KCchannels = Keltner_calc(df_to_strat, keltner_multiplier, keltner_time)
    position, SLTP_indices = position_calc(df_to_strat, KCchannels, 0.2, 0.5)
    shift_position = position
    buy_index = np.where((position[1:] != shift_position[:-1]) & (position[1:] > shift_position[:-1]))[0] + 1
    sell_index = np.where((position[:-1] != shift_position[1:]) & (position[:-1] > shift_position[1:]))[0] + 1
    profit = PnL_calc(df_to_strat, position, buy_index, sell_index, 0.1)
    return profit, KCchannels, (buy_index, sell_index, SLTP_indices)


profit, KCchannels, signal_indices = Keltner_strat(df_AAPL, 0.103058, 1)
plot_Keltner_with_Profit(df_AAPL, "AAPL", KCchannels, signal_indices, profit.cumsum())

# df_AAPL = df_AAPL[1:]
# df_AAPL['KC_middle'] = middle[1:]
# df_AAPL['KC_upper'] = upper[1:]
# df_AAPL['KC_lower'] = lower[1:]
# df_AAPL['position'] = position
# df_AAPL['price diff'] = (df_AAPL.Close).diff()
# df_AAPL['profit'] = profit
# df_AAPL

# series = np.array([0, 1, 1, 1, 0, 0])
# yesterseries = series
# buy_indexes = np.where((series[1:] != yesterseries[:-1]) & (series[1:] > yesterseries[:-1]))[0] + 1
# print(series[:-1])
# print(yesterseries[1:])
# print(buy_indexes)

#OPTIMIZATION

def annual_sharpe(profit, n = 253):
    return np.sqrt(n) * profit.mean() / profit.std()

annual_sharpe(profit, 253)

rng = np.random.default_rng()
optim_results = []
for i in range(10000):
    keltner_time = rng.integers(1, 70)
    keltner_multiplier = (rng.random()+0.01)*5
    profit, _, _ = Keltner_strat(df_AAPL.head(), keltner_multiplier, keltner_time)
    profit_over_time = profit.cumsum()
    if profit_over_time[-1] == 0:
        continue
    sharpe = annual_sharpe(profit)
    optim_results.append([keltner_time, keltner_multiplier, profit_over_time[-1], sharpe])

results_df = pd.DataFrame(optim_results)
print("idk, smth")