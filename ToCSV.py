import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
data = pd.read_csv(
    r'E:\Thesis Content\ukdale\house_1\channel_3.dat',
    delimiter=' ',
    header=None,
    names=['Date', 'Power'],
    dtype={'Date': np.int64, 'Power': np.float64},
    index_col='Date'
)
data.index = pd.to_datetime((data.index.values), unit='s')
data=data.resample('d').sum()
ts1=round((data['Power']*0.783*6)/(1000*3600),3)

ts1.fillna(0, inplace=True)
ts1.to_csv(path=r'E:\Thesis Content\ukdale CSV\house_1\Channel_3_solar_thermal_pump.csv',sep=',', header=True)


#x=pd.DataFrame(data=channel3['Date'])
#y=pd.DataFrame(data=channel3['Power']).values


#y_rbf = svr_rbf.fit(x, y).predict(x)
#y_lin = svr_lin.fit(x, y).predict(x)
#y_poly = svr_poly.fit(x, y).predict(x)
#lw = 2
#plt.figure(figsize=(12, 7))
#plt.scatter(x, y, color='darkorange', label='Power')
#plt.plot(x, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.plot(x, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(x, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
#plt.xlabel('Date')
#plt.ylabel('Power')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()
