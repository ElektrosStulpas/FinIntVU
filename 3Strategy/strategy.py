import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

#data load up
df_AAPL = yf.download('AAPL', period='10y', interval="1d")
df_INTC = yf.download('INTC', period='10y', interval="1d")
df_MSFT = yf.download('MSFT', period='10y', interval="1d")

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


def position_calc(df_to_calc, KCchannels, SLTP):
    stop_loss, take_profit = SLTP
    KC_middle, KC_upper, KC_lower = KCchannels
    df_len = len(df_to_calc[1:])
    position = np.zeros(df_len)
    SLTP_indices = []
    # since we're doing close, this means we're executing at end of day
    for i, today_price in enumerate(df_to_calc[1:].Close):
        if i+1 >= df_len: continue
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

    return position, np.array(SLTP_indices, dtype=np.int64)+1


def PnL_calc(df_to_calc, position, buy_indices, sell_indice, fees):
    PnL = (df_to_calc.Close).diff()[1:] * position
    #adjust sell and buy indexes to day before since actual position change will be for tommorow, but fees are on day close before
    PnL.iloc[buy_indices-1] -= fees
    PnL.iloc[sell_indice-1] -= fees
    return PnL


def plot_Keltner_with_Profit(df_to_plot, ticker, KCchannels, signal_indices, profit):
    middle, upper, lower = KCchannels
    buy_index, sell_index, SLTP_index = signal_indices
    fig, ax = plt.subplots(2, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    close_line, = ax[0].plot(df_to_plot.Close, color='blue', label="Close price")
    ewm_line, = ax[0].plot(middle, color='orange', label="Moving average")
    upper_line, = ax[0].plot(upper, color='green', label="Upper band")
    lower_line, = ax[0].plot(lower, color='red', label="Lower band")
    ax[0].fill_between(df_to_plot[1:].index, upper, lower, color='grey', alpha=0.3)

    buy_marker = ax[0].scatter(df_to_plot.index[buy_index], df_to_plot.Close[buy_index], marker='^', color='green', label='Buy signal')
    sell_marker = ax[0].scatter(df_to_plot.index[sell_index], df_to_plot.Close[sell_index], marker='v', color='red', label='Sell signal')
    SLTP_marker = ax[0].scatter(df_to_plot.index[SLTP_index], df_to_plot.Close[SLTP_index], marker='X', color='purple', label='SLTP signal', alpha=0.5, s=70)

    ax[0].set_title(ticker + " price with Keltner channel")
    ax[0].set_ylabel("Price")
    ax[0].grid()
    ax[0].legend(handles = [close_line, ewm_line, upper_line, lower_line, buy_marker, sell_marker, SLTP_marker])

    ax[1].plot(profit, color='black')
    ax[1].set_ylabel("Profit")
    ax[1].set_xlabel("Time")
    ax[1].grid()
    plt.show()


def Keltner_strat(df_to_strat, keltner_multiplier, keltner_time, SLTP, fees):
    KCchannels = Keltner_calc(df_to_strat, keltner_multiplier, keltner_time)
    position, SLTP_indices = position_calc(df_to_strat, KCchannels, SLTP)
    shift_position = position
    buy_index = np.where((position[1:] != shift_position[:-1]) & (position[1:] > shift_position[:-1]))[0] + 1
    sell_index = np.where((position[:-1] != shift_position[1:]) & (position[:-1] > shift_position[1:]))[0] + 1
    profit = PnL_calc(df_to_strat, position, buy_index, sell_index, fees)
    return profit, KCchannels, (buy_index, sell_index, SLTP_indices)


profit, KCchannels, signal_indices = Keltner_strat(df_AAPL, 2, 20, SLTP=(0.1, 0.5), fees=0.1)
plot_Keltner_with_Profit(df_AAPL, "AAPL", KCchannels, signal_indices, profit.cumsum())

middle, upper, lower = KCchannels
position, SLTP_indices = position_calc(df_AAPL, KCchannels, SLTP=(0.1, 0.5))

df_AAPL = df_AAPL[1:]
df_AAPL['KC_middle'] = middle[1:]
df_AAPL['KC_upper'] = upper[1:]
df_AAPL['KC_lower'] = lower[1:]
df_AAPL['position'] = position
df_AAPL['price diff'] = (df_AAPL.Close).diff()
df_AAPL['profit'] = profit
df_AAPL

# series = np.array([0, 1, 1, 1, 0, 0])
# yesterseries = series
# buy_indexes = np.where((series[1:] != yesterseries[:-1]) & (series[1:] > yesterseries[:-1]))[0] + 1
# print(series[:-1])
# print(yesterseries[1:])
# print(buy_indexes)

#OPTIMIZATION

def annual_sharpe(profit, n = 253):
    return np.sqrt(n) * profit.mean() / profit.std()


def testing_parameters_random(df_to_test, test_range):
    rng = np.random.default_rng()
    optim_results = []
    for i in range(test_range):
        keltner_time = rng.integers(1, 100)
        keltner_multiplier = round((rng.random()+0.1)*4, 2)
        profit, _, _ = Keltner_strat(df_to_test, keltner_multiplier, keltner_time, SLTP=(0.1, 0.5), fees=0.1)
        profit_over_time = profit.cumsum()
        if profit_over_time[-1] == 0:
            continue
        sharpe = annual_sharpe(profit)
        optim_results.append([keltner_time, keltner_multiplier, profit_over_time[-1], round(sharpe, 2)])

    results_df = pd.DataFrame(optim_results, columns=['KC Time', 'KC Multiplier', 'Profit', 'Sharpe'])
    return results_df


def testing_parameters_brute(df_to_test, kc_time_start, kc_time_stop, kc_multi_start, kc_multi_stop):
    # kc_multi_gen = (x/100 for x in range(kc_multi_start, kc_multi_stop, 5)) # for some reason using named generator in for it uses it up once and doesn't refresh from the start
    optim_results = []

    for keltner_time in range(kc_time_start, kc_time_stop+1):
        for keltner_multiplier in (x/100 for x in range(kc_multi_start, kc_multi_stop+1, 5)):
            profit, _, _ = Keltner_strat(df_to_test, keltner_multiplier, keltner_time, SLTP=(0.15, 0.5), fees=0.1)
            profit_over_time = profit.cumsum()
            if profit_over_time[-1] == 0:
                continue
            sharpe = annual_sharpe(profit)
            optim_results.append([keltner_time, keltner_multiplier, profit_over_time[-1], round(sharpe, 2)])

    results_df = pd.DataFrame(optim_results, columns=['KC Time', 'KC Multiplier', 'Profit', 'Sharpe'])
    return results_df


def sim_results_heatmap(df_to_draw, x_axis_col_name, y_axis_col_name, values_col_name):
    table = df_to_draw.pivot(index=y_axis_col_name, columns=x_axis_col_name, values=values_col_name)
    fig, ax0 = plt.subplots(figsize=(14, 8))
    im = ax0.pcolormesh(table)
    fig.colorbar(im, ax=ax0)
    plt.yticks(np.arange(0,len(table.index))+0.5, table.index)
    plt.ylabel(y_axis_col_name)
    plt.xticks(np.arange(0,len(table.columns.values))+0.5, table.columns.values)
    plt.xlabel(x_axis_col_name)
    plt.show()


def profit_before_and_after(df_to_calc, KC_params_before, KC_params_after, SLTP, fees):
    keltner_multiplier_before, keltner_time_before = KC_params_before
    keltner_multiplier_after, keltner_time_after = KC_params_after

    profit_before, _, _ = Keltner_strat(df_to_calc, keltner_multiplier_before, keltner_time_before, SLTP, fees)
    profit_after, _, _ = Keltner_strat(df_to_calc, keltner_multiplier_after, keltner_time_after, SLTP, fees)

    fig, ax = plt.subplots(figsize=(12, 5))
    before_optim, = ax.plot(profit_before.cumsum(), color='blue', label="Before optimization")
    after_optim, = ax.plot(profit_after.cumsum(), color='purple', label="After optimization")
    ax.legend(handles = [before_optim, after_optim])
    ax.set_ylabel("Profit")
    ax.set_xlabel("Time")
    ax.grid()
    plt.show()


# sim_results = testing_parameters_random(df_AAPL, 10000)
# sim_results.sort_values('Sharpe', ascending=False, inplace=True)
# sim_results = sim_results[:50]
# sim_results.drop_duplicates(subset=['KC Multiplier'], keep='first', inplace=True)
# sim_results_heatmap(sim_results, 'KC Time', 'KC Multiplier', 'Sharpe')

sim_results = testing_parameters_brute(df_AAPL, 10, 50, 250, 550)
sim_results_heatmap(sim_results, 'KC Time', 'KC Multiplier', 'Sharpe')

profit_before_and_after(df_AAPL, KC_params_before=(2, 20), KC_params_after=(3.1, 20), SLTP=(0.15, 0.5), fees=0.1)
