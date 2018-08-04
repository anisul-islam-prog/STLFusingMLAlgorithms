import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np




df =pd.read_csv(r'C:\Users\FAHAD\PycharmProjects\Thesis\mainconcat_v4.csv',
                index_col=0,
                parse_dates=True)
df.dropna(inplace=True)
style.use('')
df['Total_power_mainMeter'].plot()
plt.show()