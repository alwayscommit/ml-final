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

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True


def extract_station(dublin_bikes_df, station_name):
    return dublin_bikes_df[dublin_bikes_df['NAME'] == station_name]


def select_features(station_df):
    # drop NAME as is it not required
    df = station_df.drop(columns=['NAME'])
    # df.reset_index(drop=True, inplace=True)

    # to perform shift operations on this column
    df['TIME'] = df.index

    # add a unix timestamp column
    # df['TIMESTAMP'] = pd.array(df.index.astype(np.int64)) / 1000000000

    # time in days
    # df['TIME_IN_DAYS'] = (df['TIMESTAMP'] - df['TIMESTAMP'][0]) / 60 / 60 / 24

    # add a day of week column
    df['DAY_OF_WEEK'] = df.index.dayofweek
    # df = df[df['DAY_OF_WEEK'] < 5]

    return df


def select_datetime_range(df, start_string, end_string):
    print(str.format("Selecting date range from {} to {}", start_string, end_string))
    start_date = pd.to_datetime(start_string, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_string, format='%Y-%m-%d')
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df


def plot_date_range(df):
    plt.scatter(df.index, df['AVAILABLE BIKES'], color='blue', marker='.')
    plt.xticks(rotation=15)
    plt.show()


def extract_feature_set(original_df, step_size):
    df = original_df.copy()
    # 10 minutes, 30 minutes, 60 minutes
    step = step_size
    # range(1, 4) so lag is basically 3
    lag = 4
    dd = 3
    # k-3-q, k-6-q, k-9-qx
    for i in range(1, lag):
        col_name = 'k-' + str((dd * i)) + '-' + 'q'
        df[col_name + '-TIME'] = df["TIME"].shift((dd * i) + step)
        df[col_name] = df['AVAILABLE BIKES'].shift((dd * i) + step)

    # number of 5 minutes intervals in a day 288
    for i in range(1, 5):
        col_name = 'k-' + str(5-i) + 'd'
        df[col_name + '-TIME'] = df["TIME"].shift(-288 * i)
        df[col_name] = df['AVAILABLE BIKES'].shift(-288 * i)

    for i in range(1, 5):
        col_name = 'k-' + str(i) + 'w'
        df[col_name + '-TIME'] = df["TIME"].shift((288 * 7 * i) - 288 * 7)
        df[col_name] = df['AVAILABLE BIKES'].shift((288 * 7 * i) - 288 * 7)

    df['OCCUPANCY FUTURE TIME'] = df["TIME"].shift(-288 * 7)
    df['OCCUPANCY_FUTURE'] = df['AVAILABLE BIKES'].shift(-288 * 7)
    df.dropna(inplace=True)
    df.to_csv('my_csv.csv')
    return df


def train_test_model(original_df, df):
    # df = df[df['DAY_OF_WEEK'] < 5]
    # X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    # X = df[['k-3-q', 'k-6-q', 'k-9-q']]
    X = df[['k-3-q', 'k-6-q', 'k-9-q']]
    # X = df[['k-1d', 'k-2d', 'k-3d', 'k-4d']]
    y = df['OCCUPANCY_FUTURE']
    X_train, X_test, yTrain, yTest = (train_test_split(X, y, random_state=0))
    linear_reg_model = LinearRegression().fit(X_train, yTrain)
    model = Ridge(fit_intercept=False).fit(X_train, yTrain)
    y_pred = model.predict(X)
    print(str.format("RMSE : {}", math.sqrt(mean_squared_error(y, y_pred))))
    return X, y_pred


def plot_predictions(original_df, X, y_pred, step, station_label):
    # original_df = original_df[original_df['DAY_OF_WEEK'] < 5]
    plt.scatter(original_df.index, original_df['AVAILABLE BIKES'], color='blue')
    plt.scatter(X.index, y_pred, color='yellow')
    plt.xlabel("time(month-day hour)")
    plt.ylabel("Bike Occupancy")
    plt.title(station_label + " Predictions (" + str(step * 5) + " minutes ahead) - Weekly Pattern")
    plt.legend(["Training Data", "Predictions"], loc='lower right')
    plt.xlim([pd.to_datetime('2020-02-26', format='%Y-%m-%d'),
              pd.to_datetime('2020-02-28', format='%Y-%m-%d')])
    plt.xticks(rotation=15)
    plt.show()


def report_parameters(trained_model, step_q):
    feature_weights_table = PrettyTable(
        # ['Step(q)',
        #  'k-3-q', 'k-6-q', 'k-9-q',
        #  'k-1d', 'k-2d', 'k-3d',
        #  'k-1w', 'k-2w', 'k-3w'])
        ['Step(q)',
         'k-3-q', 'k-6-q', 'k-9-q'])
        # ['Step(q)',
        #  'k-1d', 'k-2d', 'k-3d', 'k-4d'])
    feature_weights_table.add_row(
        [step_q,
         trained_model.coef_[0],
         trained_model.coef_[1],
         trained_model.coef_[2],
         # trained_model.coef_[3]
         ])
    print(feature_weights_table)


def engineer_features(df, step_size):
    X = df[['k-3-q', 'k-6-q', 'k-9-q']]
    # X = df[['k-1w', 'k-2w', 'k-3w', 'k-4w']]
    # X = df[['k-1d', 'k-2d', 'k-3d', 'k-4d']]
    y = df['OCCUPANCY_FUTURE']
    linear_reg_model = LinearRegression().fit(X, y)
    report_parameters(linear_reg_model, step_size)
    # importance = linear_reg_model.coef_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()


def main():
    # load main dataset
    dublin_bikes_df = pd.read_csv('D:\AAATrinity\Machine Learning\dublinbikes_20200101_20200401.csv',
                                  usecols=['NAME', 'TIME', 'AVAILABLE BIKES'], parse_dates=['TIME'], index_col="TIME")

    # station_df = extract_station(dublin_bikes_df, 'CITY QUAY')
    station_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

    original_df = select_datetime_range(station_df, '2020-02-04', '2020-03-14')
    original_df = select_features(original_df)

    # # plot_date_range(df)
    # for step in [2, 6, 12]:
    #     df = extract_feature_set(original_df, step)
    #     engineer_features(df, step)
    #     X, y_pred = train_test_model(original_df, df)
    #     plot_predictions(df, X, y_pred, step, 'City Quay')
    #     # plot_predictions(df, X, y_pred, step, 'Brookfield Road')

    step = 12
    df = extract_feature_set(original_df, step)
    engineer_features(df, step)
    X, y_pred = train_test_model(original_df, df)
    # plot_predictions(df, X, y_pred, step, 'City Quay')
    plot_predictions(df, X, y_pred, step, 'Brookfield Road')


if __name__ == '__main__':
    main()
