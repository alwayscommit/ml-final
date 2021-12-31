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
#                           usecols=['TIME', 'AVAILABLE BIKES'],
#                           parse_dates=['TIME'], index_col="TIME")
original_df = pd.read_csv('./station_dataset/brookfield-road.csv',
                          usecols=['TIME', 'AVAILABLE BIKES'],
                          parse_dates=['TIME'], index_col="TIME")

original_df = original_df[(original_df.index >= '2020-02-04') & (original_df.index <= '2020-03-14')]

# feature engineering
# original_df['OCCUPANCY'] = original_df['AVAILABLE BIKES'] / original_df['BIKE STANDS']
# original_df['OCCUPANCY'] = original_df['AVAILABLE BIKES']
# df = original_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1)

df = original_df

df.index = df.index - pd.to_timedelta(df.index.second, unit='s')
df["TIME"] = df.index

# 10 minutes, 30 minutes, 60 minutes
step = 12
# range(1, 4) so lag is basically 3
lag = 4
dd = 1

for i in range(1, 3):
    col_name = 'k-' + str(i) + '-' + 'q'
    df[col_name + 'time'] = df["TIME"].shift(i + step)
    df[col_name] = df['AVAILABLE BIKES'].shift(i + step)

# k-3-q, k-6-q, k-9-qx
for i in range(0, lag):
    col_name = 'k-' + str(3 * i) + '-' + 'q'
    df[col_name + 'time'] = df["TIME"].shift((3 * i) + step)
    df[col_name] = df['AVAILABLE BIKES'].shift((3 * i) + step)

# number of 5 minutes intervals in a day 288
for i in range(1, lag):
    col_name = 'k-' + str(i) + 'd'
    df[col_name + 'time'] = df["TIME"].shift(288 * i)
    df[col_name] = df['AVAILABLE BIKES'].shift(288 * i)

for i in range(1, lag):
    col_name = 'k-' + str(i) +  'w'
    df[col_name + 'time'] = df["TIME"].shift(288 * 7 * i)
    df[col_name] = df['AVAILABLE BIKES'].shift(288 * 7 * i)

df['OCCUPANCY FUTURE TIME'] = df["TIME"].shift(-288 * 7)
df['OCCUPANCY_FUTURE'] = df['AVAILABLE BIKES'].shift(-288 * 7)

# df['OCCUPANCY FUTURE TIME'] = df["TIME"]
# df['OCCUPANCY_FUTURE'] = df['AVAILABLE BIKES']

df.dropna(inplace=True)
df.to_csv('my_csv.csv')

#most recent trend
# X = df[['k-0-q', 'k-1-q', 'k-2-q']]

#trend with a lag of 3
# X = df[['k-0-q', 'k-3-q', 'k-6-q', 'k-9-q']]

#day seasonality
# X = df[['k-1d', 'k-2d', 'k-3d']]

# Week Seasonality
# X = df[['k-1w', 'k-2w', 'k-3w']]

X = df[['k-0-q', 'k-1-q', 'k-2-q', 'k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]

y = df['OCCUPANCY_FUTURE']

X_train, X_test, yTrain, yTest = (train_test_split(X, y, random_state=0))

linear_reg_model = LinearRegression().fit(X_train, yTrain)
# print(reg.coef_)

model = Ridge(fit_intercept=False).fit(X_train, yTrain)

y_pred = model.predict(X_test)

feature_weights_table = PrettyTable(
    [ 'Step(q)', 'RMSE', 'k-0-q', 'k-1-q', 'k-2-q', 'k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w',
     'k-3w'])

# feature_weights_table = PrettyTable(
#     ['Step(q)', 'RMSE', 'k-1w', 'k-2w', 'k-3w'])

# def report_parameters(trained_model, step_q):
#     feature_weights_table.add_row(
#         [step_q, math.sqrt(mean_squared_error(yTest, y_pred)), trained_model.coef_[0],
#          trained_model.coef_[1],
#          trained_model.coef_[2]])

def report_parameters(trained_model, step_q):
    feature_weights_table.add_row(
        [step_q, math.sqrt(mean_squared_error(yTest, y_pred)), trained_model.coef_[0],
         trained_model.coef_[1],
         trained_model.coef_[2],
         trained_model.coef_[3],
         trained_model.coef_[4],
         trained_model.coef_[5],
         trained_model.coef_[6],
         trained_model.coef_[7],
         trained_model.coef_[8],
         trained_model.coef_[9],
         trained_model.coef_[10],
         trained_model.coef_[11]])

# report_parameters("Ridge", model, step)
report_parameters(linear_reg_model, step)
print(feature_weights_table)

plt.scatter(original_df.index, original_df['AVAILABLE BIKES'], color='blue')
plt.scatter(X_test.index, y_pred, color='yellow')
plt.xlabel("time(days)")
plt.ylabel("Bike Occupancy")
plt.title("Grand Canal Dock Predictions (" + str(step * 5) + " minutes ahead)")
plt.legend(["training data", "predictions"], loc='upper right')
# plt.xlim([pd.to_datetime('2020-03-13', format='%Y-%m-%d'),
#           pd.to_datetime('2020-03-25', format='%Y-%m-%d')])
plt.xticks(rotation=15)
plt.show()
