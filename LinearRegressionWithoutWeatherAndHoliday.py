import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import math
from sklearn import preprocessing
import datetime


#Initializing SVR Kernels
clf= LinearRegression()


#Reading Data
df =pd.read_csv(r'C:\Users\FAHAD\PycharmProjects\Thesis\mainconcat_v8_Without.csv',
                index_col=0,
                parse_dates=True)
df.dropna(inplace=True)


#Declaring forcast variables
forecast_col='Total_power_mainMeter'
forecast_out = 2 #int(math.ceil(0.0001 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)



#Dividing into Train and Test Set
X = np.array(df.drop(['label','Total_power_mainMeter'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['Total_power_mainMeter'])
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#Trainnig a Model
ClflinR = clf.fit(X_train, y_train)


#Testing a Model
y_predict_lin = ClflinR.predict(X_test)
msr=mean_squared_error(y_test, y_predict_lin)
mar=mean_absolute_error(y_test, y_predict_lin)
rms_lin = sqrt(msr)
print('lin_score_')
print(ClflinR.score(X_test,y_test))
print('_Mean_Absolute_Error')
print(mar)
print('_Mean_Squared_Error')
print(msr)
print('_RMS_linear_')
print(rms_lin)

forecast_set=ClflinR.predict(X_test)
twoday_forecast_set=ClflinR.predict(X_lately)
twoday_y=y[-2:]
twoday_X=df.index
twoday_X=twoday_X[-2:]

#Plotting Graph
style.use('ggplot')
df['Forecast'] = np.nan


#Creating Unix Timestap Values for graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Total_power_mainMeter'].plot(color='black',label='Power')
df['Forecast'].plot(color='green', lw=2, label='Linear Regression')
plt.xlabel('Date')
plt.ylabel('Power')
plt.title('Linear Regression')
plt.legend(loc=4)
plt.show()
plt.close()
plt.bar(twoday_X,twoday_y,color='orange',label='True Power')
plt.bar(twoday_X,twoday_forecast_set,color='blue',label='Predicted Power')
plt.xlabel('Date')
plt.ylabel('Power')
plt.title('Linear Regression')
plt.legend()
plt.show()