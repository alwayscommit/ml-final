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

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True

# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
# feature selection
# original_df = pd.read_csv('./station_dataset/grand_canal_dock.csv',
#                           usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
#                           parse_dates=['TIME'], index_col="TIME")
original_df = pd.read_csv('./station_dataset/brookfield-road.csv',
                          usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
                          parse_dates=['TIME'], index_col="TIME")

original_df = original_df[(original_df.index >= '2020-01-23') & (original_df.index <= '2020-04-01')]

# feature engineering
# original_df['OCCUPANCY'] = original_df['AVAILABLE BIKES'] / original_df['BIKE STANDS']
original_df['OCCUPANCY'] = original_df['AVAILABLE BIKES']
df = original_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1)

df.index = df.index - pd.to_timedelta(df.index.second, unit='s')
df["TIME"] = df.index

# 10 minutes, 30 minutes, 60 minutes
step = 2
# range(1, 4) so lag is basically 3
lag = 4
dd = 1

# k-3-q, k-6-q, k-9-q
for i in range(1, lag):
    col_name = 'k-' + str(3 * i) + '-' + 'q'
    df[col_name + 'time'] = df["TIME"].shift((3 * i) + step)
    df[col_name] = df['OCCUPANCY'].shift((3 * i) + step)

# number of 5 minutes intervals in a day 288
for i in range(1, lag):
    col_name = 'k-' + str(i) + 'd'
    df[col_name + 'time'] = df["TIME"].shift(288 * i)
    df[col_name] = df['OCCUPANCY'].shift(288 * i)

for i in range(1, lag):
    col_name = 'k-' + str(i) + 'w'
    df[col_name + 'time'] = df["TIME"].shift(288 * 7 * i)
    df[col_name] = df['OCCUPANCY'].shift(288 * 7 * i)

df['OCCUPANCY FUTURE TIME'] = df["TIME"].shift(-288 * 7)
df['OCCUPANCY_FUTURE'] = df['OCCUPANCY'].shift(-288 * 7)

df.dropna(inplace=True)
df.to_csv('my_csv.csv')

X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
# X = df[['k-1w', 'k-2w', 'k-3w']]
# X = df[['k-1d', 'k-2d', 'k-3d']]
# X = df[['k-3-q', 'k-6-q', 'k-9-q']]
# X = df[['k-3-q', 'k-1d', 'k-3d', 'k-1w', 'k-2w']]

y = df['OCCUPANCY_FUTURE']

X_train, X_test, yTrain, yTest = (train_test_split(X, y, random_state=0))

reg = LinearRegression().fit(X_train, yTrain)
print(reg.coef_)

model = Ridge(fit_intercept=False).fit(X_train, yTrain)

y_pred = model.predict(X_test)
print(math.sqrt(mean_squared_error(yTest, y_pred)))

# feature_weights_table = PrettyTable(
#     ['Step(q)', 'Intercept', 'k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w'])
#
# def report_parameters(trained_model, step_q):
#     feature_weights_table.add_row(
#         [step_q, trained_model.intercept_, trained_model.coef_[0], trained_model.coef_[1],
#          trained_model.coef_[2],
#          trained_model.coef_[3],
#          trained_model.coef_[4],
#          trained_model.coef_[5],
#          trained_model.coef_[6],
#          trained_model.coef_[7],
#          trained_model.coef_[8]])


# report_parameters(model, step)
# print(feature_weights_table)

plt.scatter(original_df.index, original_df['OCCUPANCY'], color='blue')
plt.scatter(X_test.index, y_pred, color='yellow')
plt.xlabel("time(days)")
plt.ylabel("Bike Occupancy")
plt.title("Grand Canal Dock Predictions (" + str(step * 5) + " minutes ahead)")
plt.legend(["training data", "predictions"], loc='upper right')
plt.xlim([pd.to_datetime('2020-03-13', format='%Y-%m-%d'),
          pd.to_datetime('2020-03-29', format='%Y-%m-%d')])
plt.xticks(rotation=15)
plt.show()
