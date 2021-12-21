import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, sys
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.constrained_layout.use'] = True

# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
canal_dock_df = pd.read_csv('./station_dataset/grand_canal_dock.csv',
                            usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
                            parse_dates=['TIME'], index_col="TIME")
# kilmainham_lane = pd.read_csv('/content/drive/MyDrive/ML Final/kilmainham_lane.csv', usecols = [2,7], parse_dates=[1])
df = canal_dock_df

# convert date time column to timestamp using pandas
timestamp_array = pd.array(
    pd.DatetimeIndex(df.index).astype(np.int64)) / 1000000000  # convert nanoseconds to seconds
timestamp_interval = timestamp_array[1] - timestamp_array[0]  # 300 seconds, 5 minutes

# start date, end date
start = pd.to_datetime('04−02−2020', format='%d−%m−%Y')
end = pd.to_datetime('14−03−2020', format='%d−%m−%Y')

t_start = (pd.DatetimeIndex([start]).astype(np.int64) / 1000000000).values[0]
t_end = (pd.DatetimeIndex([end]).astype(np.int64) / 1000000000).values[0]

t = np.extract([(timestamp_array.to_numpy() >= t_start) & (timestamp_array.to_numpy() <= t_end)], timestamp_array)
# convert timestamp to days
t = (t - t[0]) / 60 / 60 / 24

y = np.extract([(timestamp_array.to_numpy() >= t_start) & (timestamp_array.to_numpy() <= t_end)], df.iloc[:, 1]).astype(
    np.int64)

q = 2
lag = 3
stride = 1
# number of samples per week
week = math.floor(7 * 24 * 60 * 60 / timestamp_interval)
length = y.size - week - lag * week - q
XX = y[q:q + length:stride]

# for i in range(1, lag):
    # X = y[i * week + q:i * week + q + length:stride]
    # XX = np.column_stack((XX, X))

# number of samples per day
# d = math.floor(24 * 60 * 60 / timestamp_interval)
# for i in range(0, lag):
    # X = y[i * d + q:i * d + q + length:stride]
    # XX = np.column_stack((XX, X))

for i in range(1, lag):
    X = y[i:i + length:stride]
    XX = np.column_stack((XX, X))

yy = y[lag * week + week + q:lag * week + week + q + length:stride]
tt = t[lag * week + week + q:lag * week + week + q + length:stride]
from sklearn.model_selection import train_test_split

train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
# train = np.arange(0,yy.size)
from sklearn.linear_model import Ridge

model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
print(model.intercept_, model.coef_)
y_pred = model.predict(XX)
print(math.sqrt(mean_squared_error(y_pred, yy)))
plt.scatter(t, y, color='black')
plt.scatter(tt, y_pred, color='red')
plt.xlabel('time(days)')
plt.ylabel('  # bikes')
plt.legend(['training data', 'predictions'], loc='upper right')
day = math.floor(24 * 60 * 60 / timestamp_interval)  # number of samples per day
# plt.xlim((4 * 7, 4 * 7 + 4))
plt.show()
