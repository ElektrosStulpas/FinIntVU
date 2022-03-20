import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

#data load up
df_AAPL = yf.download('AAPL', period='1y', interval="1d")
df_INTC = yf.download('INTC', period='1y', interval="1d")
df_MSFT = yf.download('MSFT', period='1y', interval="1d")

# WMA
#buy/sell on price cross with wma, trend direction
#buy lows on up, sell highs on down
def WMA_calc(data, timePeriod):
    weights_mask = np.arange(1, timePeriod+1) #arange is not inclusive for last
    denom = weights_mask.sum() #can either sum all or do the whole n*(n+1)/2
    WMA = np.zeros(len(data))

    for i in range(len(data[timePeriod-1:])): 
        period_data = data[i:timePeriod+i]
        WMA[timePeriod+i-1] = (period_data * weights_mask).sum()/denom

    WMA = pd.Series(WMA)
    WMA.index = data.index
    return WMA[timePeriod-1:]


def plot_WMA(df_price_to_plot, WMA_period, ticker):
    df_WMA = WMA_calc(df_price_to_plot, WMA_period)
    plt.figure(figsize=(12, 5))
    plt.plot(df_price_to_plot)
    plt.plot(df_WMA)
    plt.title(ticker + " price with WMA")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.grid()
    plt.show()


plot_WMA(df_AAPL.Close, 20, "AAPL")
plot_WMA(df_INTC.Close, 20, "INTC")
plot_WMA(df_MSFT.Close, 20, "MSFT")



#MFI
#overbought/oversold, divergences and failure swings
def MFI_calc(data, timePeriod):
    MFI = np.zeros(len(data))
    typicalPrice = (data.High + data.Low + data.Close)/3
    moneyFlow = typicalPrice * data.Volume

    negativeMoneyFlow = np.zeros(len(moneyFlow))
    positiveMoneyFlow = np.zeros(len(moneyFlow))

    #could probably do this with shift
    negMoneyFlowIndices = np.where(np.ediff1d(typicalPrice) < 0)[0] + 1 # indexes where current typical price was lower than the day before
    posMoneyFlowIndices = np.where(np.ediff1d(typicalPrice) > 0)[0] + 1 # indexes where current typical price was higher than the day before

    negativeMoneyFlow[negMoneyFlowIndices] = moneyFlow[negMoneyFlowIndices]
    positiveMoneyFlow[posMoneyFlowIndices] = moneyFlow[posMoneyFlowIndices]

    for i in range(len(data[timePeriod-1:])):
        negMFforPeriod = negativeMoneyFlow[i:timePeriod+i].sum() # instead of slicing I could throw out last and add in next
        posMFforPeriod = positiveMoneyFlow[i:timePeriod+i].sum()

        moneyRatio = negMFforPeriod/posMFforPeriod #flipped values from original formula

        MFI[timePeriod+i-1] = 100/(1+moneyRatio)

    print(type(MFI))
    MFI = pd.Series(MFI)
    MFI.index = data.index

    return MFI[timePeriod-1:]


def get_MFI_bound(df, bound):
    boundArray = np.repeat(bound, len(df))
    boundSeries = pd.Series(boundArray)
    boundSeries.index = df.index
    return boundSeries


def plot_MFI(df_to_plot, MFI_period, ticker, MFI_upper, MFI_lower):
    df_MFI = MFI_calc(df_to_plot, MFI_period)
    MFI_up = get_MFI_bound(df_to_plot, MFI_upper)
    MFI_low = get_MFI_bound(df_to_plot, MFI_lower)
    fig, ax = plt.subplots(2, figsize=(12, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax[0].plot(df_to_plot.Close, color='blue')
    ax[0].set_title(ticker + " price with MFI")
    ax[0].set_ylabel("Price")
    ax[0].grid()

    ax[1].plot(df_MFI, color='black')
    ax[1].plot(MFI_up, color='grey')
    ax[1].plot(MFI_low, color='grey')
    ax[1].set_ylabel("MFI")
    ax[1].set_xlabel("Time")
    ax[1].grid()
    plt.show()


plot_MFI(df_AAPL, 14, "AAPL", 80, 20)
plot_MFI(df_INTC, 14, "INTC", 80, 20)
plot_MFI(df_MSFT, 14, "MSFT", 80, 20)



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


def plot_Keltner(df_to_plot, ticker, kc_shift, period):
    middle, upper, lower = Keltner_calc(df_to_plot, kc_shift, period)
    plt.figure(figsize=(12, 5))
    plt.plot(df_to_plot.Close, color='blue')
    plt.plot(middle, color='black')
    plt.plot(upper, color='green')
    plt.plot(lower, color='red')
    plt.fill_between(df_to_plot[1:].index, upper, lower, color='grey', alpha=0.3)
    plt.title(ticker + " price with Keltner channel")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.grid()
    plt.show()


plot_Keltner(df_AAPL, 'AAPL', 2, 20)
plot_Keltner(df_INTC, 'INTC', 2, 20)
plot_Keltner(df_MSFT, 'MSFT', 2, 20)