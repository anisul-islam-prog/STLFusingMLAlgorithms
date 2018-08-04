import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
#from pandas import datetime
#from pandas.tseries.t
from sklearn.preprocessing import MinMaxScaler
#from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import Series

data = pd.read_csv(
    r'E:\Thesis Content\ukdale\house_1\channel_7.dat',
    delimiter=' ',
    header=None,
    names=['date', 'KWh'],
    dtype={'date': np.int64, 'KWh': np.float64},
    index_col='date'
    ) #initially KWh column contains Ws in 6 second interval, later it will be converted to KWh

data.index = pd.to_datetime((data.index.values), unit='s')
#data.head(5)
#before_process = data
after_process=data
#before_process = before_process.resample('d').sum()
#before_process['KWh'] = round(((before_process.KWh * 6) / (1000 * 3600)) , 3)
#before_process.head(5)
after_process = after_process.drop(after_process[(after_process.KWh < 10) | (after_process.KWh > 4000) ].index)
after_process = after_process.resample('d').sum()
#after_process.head(5)
after_process['KWh'] = round(((after_process.KWh * 6) / (1000 * 3600)) , 3)
after_process.head(5)

after_process.to_csv(path_or_buf=r'E:\Thesis Content\ukdale CSV\Without Noise\Tvday.csv', sep = ',' , index_label = 'date')


#rcParams['figure.figsize'] = 16, 10
#plt.subplot(2, 1, 1)
#plt.scatter(before_process.index ,before_process['KWh'].values, s=10)
#plt.title('Before and After Pre Processing')
#plt.ylabel('KWh')
#plt.subplot(2, 1, 2)
#plt.scatter(after_process.index ,after_process['KWh'].values, s=10)
#plt.xlabel('Date')
#plt.ylabel('KWh')
#plt.show()