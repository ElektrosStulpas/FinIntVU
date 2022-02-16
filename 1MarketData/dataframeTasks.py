import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as mp
dates = pd.date_range('20190214', periods=6)
numbers = np.matrix( [[ 101, 103], [105.5, 75], [102, 80.3], [100, 85], [110, 98], [109.6, 125.7 ]] )
df = pd.DataFrame(numbers, index=dates, columns=['A', 'B'])

#1 eilute, kurios indeksas yra '2019-02-18'
df.loc[dates[4]]
df.loc['2019-02-18']

#2 eilute, kurios indekso data yra datetime.datetime(2019,2,18)
df.loc[datetime.datetime(2019,2,18)]

#3 eilute, kuri priespaskutine nuo galo (be indekso)
df.iloc[-2]

#4 pirmos dvi eilutes ir stulpelis 'B' (be indekso)
df.iloc[0:2, 1]

#5 isrusiuoti df pagal 'B' mazejanciai
df = df.nlargest(len(df), 'B')
len(df.index)
df.shape[0]

#6 rasti stulpelio 'A' max reiksme
df['A'].max()

#7 padvigubinti stulpelio 'A' max reiksme
df.at[df['A'].idxmax(), 'A'] = df['A'].max() * 2

#8 gauti eilutes kur 'A' reiksmes > 105
df['A'][df['A'] > 105]
df[df['A'] > 105]

#9 nupiesti stulpelio 'A' reiksmes
df['A'].plot()
mp.show()

#10 istrinti eilutes kur 'B' reiksmes > 'A' reiksmes
df.drop(df[df['B'] > df['A']].index)