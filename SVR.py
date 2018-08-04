import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from sklearn import preprocessing
import datetime

#Initializing SVR Kernels
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01, epsilon=0.2)
#svr_lin = SVR(kernel='linear', C=1e3, epsilon=0.2)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3, epsilon=0.2)
#clf= LinearRegression()


#Reading Data
df =pd.read_csv(r'C:\Users\FAHAD\PycharmProjects\Thesis\mainconcat_v4.csv',
                index_col=0,
                parse_dates=True)
df.dropna(inplace=True)



#Seeing Data Structure
#print(df.head())
#df.plot(style='k.')
#plt.show()



#Selecting features
df = df[['Total_power_mainMeter']]

#Declaring forcast variables
forecast_col='Total_power_mainMeter'
forecast_out = 1 #int(math.ceil(0.0001 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
#df.drop(['Total_power_mainMeter'],1)
#df.dropna(inplace=True)




#Dividing into Train and Test Set
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=121234)


#Trainnig a Model
#linR = svr_lin.fit(X_train, y_train)
#rbfR = svr_rbf.fit(X_train, y_train)
#polyR = svr_poly.fit(X_train, y_train)

#Getting Result of a model
#forecast_set_poly = polyR.predict(X_lately)
#forecast_set_lin = linR.predict(X_lately)
#forecast_set_rbf = rbfR.predict(X_lately)

#Testing a Model
#y_predict_lin = linR.predict(X_test)
#y_predict_poly = polyR.predict(X_test)
#y_predict_rbf = rbfR.predict(X_test)
#rms_lin = sqrt(mean_squared_error(y_test, y_predict_lin))
#rms_poly = sqrt(mean_squared_error(y_test, y_predict_poly))
#rms_rbf = sqrt(mean_squared_error(y_test, y_predict_rbf))
#print('SVR_lin_score_')
#print(linR.score(X_test,y_test))
#print('_rms_linear_')
#print(rms_lin)
#print('SVR_poly_score_')
#print(polyR.score(X_test,y_test))
#print('_rms_poly_')
#print(rms_poly)
#print('SVR_rbf_score_')
#print(rbfR.score(X_test,y_test))
#print('_rms_rbf_')
#print(rms_rbf)




#Plotting Graph
style.use('ggplot')
#df['Forecast'] = np.nan


#Creating Unix Timestap Values for graph
#last_date = df.iloc[-1].name
#last_unix = last_date #last_date.timestamp()
#one_day = 86400
#next_unix = last_unix + one_day

#for i in forecast_set_lin:
#    next_date = datetime.datetime.fromtimestamp(next_unix)
#    next_unix += 86400
#    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

plt.figure(figsize=(12, 7))
df['Total_power_mainMeter'].plot(label='Power')
#df['Forecast'].plot(color='c', lw=lw, label='Linear model')
#df['Forecast'].plot(color='navy', lw=2, label='RBF model')

#df['Forecast'].plot(color='navy', lw=lw, label='RBF model')
#plt.figure(figsize=(12, 7))
#plt.scatter(X_train, y_train, color='darkorange', label='Power')
#plt.plot(X_train, rbfR.predict(X_train), color='navy', lw=lw, label='RBF model')
#plt.plot(X_train, linR.predict(X_train), color='c', lw=lw, label='Linear model')
#plt.plot(X_train, polyR.predict(X_train), color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Power in KWH')
plt.title('Power Consumption With Time')
plt.legend(loc=4)
plt.show()
