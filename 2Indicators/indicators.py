import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

def plot(df_to_plot, title, ylabel, xlabel):
    df_to_plot.plot()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.show()


# WMA
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
    plt.plot(df_price_to_plot)
    plt.plot(df_WMA)
    plt.title(ticker + " price with WMA")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.show()


#MFI
def MFI_calc(data, timePeriod):
    MFI = np.zeros(len(data))
    typicalPrice = (data.High + data.Low + data.Close)/3
    moneyFlow = typicalPrice * data.Volume

    negativeMoneyFlow = np.zeros(len(moneyFlow))
    positiveMoneyFlow = np.zeros(len(moneyFlow))

    negMoneyFlowIndices = np.where(np.ediff1d(typicalPrice) < 0)[0] + 1 # indexes where current typical price was lower than the day before
    posMoneyFlowIndices = np.where(np.ediff1d(typicalPrice) > 0)[0] + 1 # indexes where current typical price was higher than the day before

    negativeMoneyFlow[negMoneyFlowIndices] = moneyFlow[negMoneyFlowIndices]
    positiveMoneyFlow[posMoneyFlowIndices] = moneyFlow[posMoneyFlowIndices]

    for i in range(len(data[timePeriod-1:])):
        negMFforPeriod = negativeMoneyFlow[i:timePeriod+i].sum() # instead of slicing I could throw out last and add in next
        posMFforPeriod = positiveMoneyFlow[i:timePeriod+i].sum()

        moneyRatio = posMFforPeriod/negMFforPeriod

        MFI[timePeriod+i-1] = 100 - (100/(1+moneyRatio))

    MFI = pd.Series(MFI)
    MFI.index = data.index

    return MFI[timePeriod-1:]


def plot_MFI(df_to_plot, MFI_period, ticker):
    df_MFI = MFI_calc(df, MFI_period)
    fig, ax = plt.subplots(2)
    ax[0].plot(df_to_plot.Close, 'r')
    ax[1].plot(df_MFI, 'b')
    # fig.title(ticker + " price with WMA")
    # plt.ylabel("Price")
    # plt.xlabel("Time") #TODO figure out how to format this
    plt.show()


df = yf.download('AAPL', period='5y', interval="1d")
plot_WMA(df.Close, 20, "APPL")
plot_MFI(df, 14, "APPL")


fig, ax = plt.subplots(2)  # figura viena, asiu gali buti daug
#fig, ax = plt.subplots(2, 3, sharex=True, sharey=True) # sujungtos visos asys. Naudinga darant zoom in
ax[0].plot(df.Close["2018-02-24":"2022-02-24"], 'r')
ax[1].plot(WMA_calc(df.Close["2018-02-24":"2022-02-24"], 20), 'g')
plt.show()

#Keltner channel

headDF = df[:50]

def Keltner_calc(data, kc_multiplier, kc_timePeriod, atr_timePeriod):
    tr1 = data.High - data.Low
    tr2 = abs(data.High - data.Close.shift())
    tr3 = abs(data.Low - data.Close.shift())
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_timePeriod).mean()

    kc_middle = data.Close.ewm(kc_timePeriod).mean()
    kc_upper = data.Close.ewm(kc_timePeriod).mean() + kc_multiplier * atr
    kc_lower = data.Close.ewm(kc_timePeriod).mean() - kc_multiplier * atr

    return kc_middle, kc_upper, kc_lower


middle, upper, lower = Keltner_calc(df["2021-03-17":], 2, 20, 10)
plt.plot(df.Close["2021-03-17":])
plt.plot(middle)
plt.plot(upper)
plt.plot(lower)
plt.show()