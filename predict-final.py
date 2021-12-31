import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from datetime import datetime

df = pd.read_csv('./station_dataset/grand_canal_dock.csv',
                 usecols=['TIME', 'AVAILABLE BIKES'],
                 parse_dates=['TIME'], index_col="TIME")

# convert timestamp data to unix timestamp in seconds
total_time = pd.array(df.index.astype(np.int64)) / 1000000000
# calculate interval (300 seconds)
interval = total_time[1] - total_time[0]

# select start date and end date
start_date = datetime(day=4, month=2, year=2020)
end_date = datetime(day=14, month=3, year=2020)
start_timestamp = (pd.DatetimeIndex([start_date]).astype(np.int64) / 1000000000).values[0]
end_timestamp = (pd.DatetimeIndex([end_date]).astype(np.int64) / 1000000000).values[0]

time = np.extract([(total_time.to_numpy() >= start_timestamp) & (total_time.to_numpy() <= end_timestamp)], total_time)
# seconds to days
time = (time - time[0]) / 60 / 60 / 24
occupancy = np.extract([(total_time.to_numpy() >= start_timestamp) & (total_time.to_numpy() <= end_timestamp)],
                       df['AVAILABLE BIKES']).astype(np.int64)

q = 2
lag = 4
stride = 1
# intervals per week
day_seconds = 24 * 60 * 60
day_interval = math.floor(day_seconds / interval)  # day_interval = 288
week_interval = day_interval * 7

length = occupancy.size - lag * week_interval
XX = occupancy[q:q + length:stride]

for i in range(1, lag):
    X = occupancy[i * week_interval:i * week_interval + length:stride]
    XX = np.column_stack((XX, X))

# number of samples per day
for i in range(1, lag):
    X = occupancy[i * day_interval:i * day_interval + length:stride]
    XX = np.column_stack((XX, X))

for i in range(1, lag):
    X = occupancy[i:i + length:stride]
    XX = np.column_stack((XX, X))

y = occupancy[lag * week_interval:lag * week_interval + length:stride]
t = time[lag * week_interval:lag * week_interval + length:stride]

train, test = train_test_split(np.arange(0, y.size), test_size=0.2)

cv = KFold(n_splits=10, shuffle=False, random_state=None)
grid_search = GridSearchCV(
    estimator=Ridge(fit_intercept=True),
    cv=cv,
    scoring='r2',
    param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
)

grid_search.fit(XX[train], y[train])
best_model = grid_search.best_estimator_
coeff = [round(x, 4) for x in best_model.coef_]
print(f'Best Model with coeff: {coeff}')

y_pred = best_model.predict(XX)
print(math.sqrt(mean_squared_error(y_pred, y)))
plt.plot(time, occupancy, color='black')
plt.plot(t, y_pred, color='red')
plt.xlabel('time(days)')
plt.ylabel('#bikes')
plt.legend(['training data', 'predictions'], loc='upper right')
day = math.floor(24 * 60 * 60 / interval)  # number of samples per day
plt.xlim((4 * 7, 4 * 7 + 4))
plt.show()
