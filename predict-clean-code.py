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
import statistics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    df = df[df['DAY_OF_WEEK'] < 5]

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
    for i in range(1, lag):
        col_name = 'k-' + str(lag - i) + 'd'
        df[col_name + '-TIME'] = df["TIME"].shift(-288 * (i + 1))
        df[col_name] = df['AVAILABLE BIKES'].shift(-288 * (i + 1))

    for i in range(1, lag):
        col_name = 'k-' + str(i) + 'w'
        df[col_name + '-TIME'] = df["TIME"].shift((288 * 5 * i) - 288 * 5)
        df[col_name] = df['AVAILABLE BIKES'].shift((288 * 5 * i) - 288 * 5)

    df['OCCUPANCY FUTURE TIME'] = df["TIME"].shift(-288 * 5)
    df['OCCUPANCY_FUTURE'] = df['AVAILABLE BIKES'].shift(-288 * 5)
    df.dropna(inplace=True)
    df.to_csv('my_csv.csv')
    return df


def cross_validate(df):
    # df = df[df['DAY_OF_WEEK'] < 5]
    original_df = df.copy()
    df.reset_index(inplace=True, drop=True)
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    # X = df[['k-1d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    # X = df[['k-3-q', 'k-6-q', 'k-9-q']]
    # X = df[['k-1d', 'k-2d', 'k-3d']]
    # X = df[['k-1w', 'k-2w', 'k-3w']]
    y = df['OCCUPANCY_FUTURE']

    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)

    mean_list_ridge = []
    stdev_list_ridge = []
    C_list_Ridge = [0.1, 0.25, 0.5, 0.75, 1]
    for C in C_list_Ridge:
        k_fold = KFold(n_splits=5)
        mean_sqaured_error_list = []
        for train_index, test_index in k_fold.split(X_poly):
            model = Lasso(alpha=(1 / C), max_iter=2000000)
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # X_train, X_test = X_poly.iloc[train_index], X_poly.iloc[test_index]
            # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            ypred = model.predict(X_test)
            mean_sqaured_error_list.append(math.sqrt(mean_squared_error(y_test, ypred)))
        mean_list_ridge.append(statistics.mean(mean_sqaured_error_list))
        stdev_list_ridge.append(statistics.stdev(mean_sqaured_error_list))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(C_list_Ridge, mean_list_ridge, color='blue', label='Mean')
    ax.errorbar(C_list_Ridge, stdev_list_ridge, color='red', label='Standard Deviation')
    ax.set_ylabel("mean & standard deviation")
    ax.set_xlabel("C")
    ax.set_title("5 Folds with Ridge Regression")
    ax.legend(loc='upper right')
    plt.show()


def train_test_model(df):
    # X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    original_df = df.copy()
    X = df[['k-1d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    y = df['OCCUPANCY_FUTURE']
    T = df['TIME']
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    X_train, X_test, yTrain, yTest, t_train, t_test = (train_test_split(X_poly, y, T, random_state=0))
    model = Lasso(alpha=(1 / 0.5), max_iter=2000000).fit(X_train, yTrain)
    y_pred = model.predict(X_poly)
    print(str.format("RMSE : {}", math.sqrt(mean_squared_error(y, y_pred))))

    plt.scatter(original_df.index, original_df['AVAILABLE BIKES'], color='blue')
    plt.scatter(T, y_pred, color='yellow')
    plt.xlabel("time(month-day hour)")
    plt.ylabel("Bike Occupancy")
    plt.legend(["Training Data", "Predictions"], loc='lower right')
    plt.xlim([pd.to_datetime('2020-02-26', format='%Y-%m-%d'),
              pd.to_datetime('2020-02-28', format='%Y-%m-%d')])
    plt.xticks(rotation=15)
    plt.show()


# def cross_validate_poly(df):
#     X = df[['k-1d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
#     y = df['OCCUPANCY_FUTURE']
#     T = df['TIME']
#     poly_mean_error = []
#     poly_std_error = []
#     polynomial_range = [1, 2, 3]
#     for q in polynomial_range:
#         Xtrain_poly = PolynomialFeatures(q).fit_transform(X)
#         model = Ridge(max_iter=2000000)
#         scores = cross_val_score(model, Xtrain_poly, y, cv=5, scoring='neg_mean_squared_error')
#         poly_mean_error.append(-1*np.array(scores).mean())
#         poly_std_error.append(-1*np.array(scores).std())
#
#     fig, axes = plt.subplots()
#     axes.errorbar(polynomial_range, poly_mean_error, yerr=poly_std_error, linewidth=3,
#                   label='logistic regression with L2')
#     axes.set_xlabel('Polynomial Range');
#     axes.set_ylabel("mean & standard deviation")
#     fig.legend()
#     plt.show()

def plot_predictions(original_df, X, y_pred, step, station_label):
    # original_df = original_df[original_df['DAY_OF_WEEK'] < 5]
    plt.scatter(original_df.index, original_df['AVAILABLE BIKES'], color='blue')
    plt.scatter(X.index, y_pred, color='yellow')
    plt.xlabel("time(month-day hour)")
    plt.ylabel("Bike Occupancy")
    plt.title(station_label + " Predictions (" + str(step * 5) + " minutes ahead) - Weekly Pattern excluding Weekends",
              fontsize=10)
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
        # ['Step(q)',
        #  'k-3-q', 'k-6-q', 'k-9-q'])
        # ['Step(q)',
        #  'k-1d', 'k-2d', 'k-3d'])
        # ['Step(q)',
        #  'k-1w', 'k-2w', 'k-3w'])
        ['Step(q)',
         'k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w'])
    feature_weights_table.add_row(
        [step_q,
         trained_model.coef_[0],
         trained_model.coef_[1],
         trained_model.coef_[2],
         trained_model.coef_[3],
         trained_model.coef_[4],
         trained_model.coef_[5],
         trained_model.coef_[6],
         trained_model.coef_[7],
         trained_model.coef_[8],
         ])
    print(feature_weights_table)


def engineer_features(df, step_size):
    # X = df[['k-3-q', 'k-6-q', 'k-9-q']]
    # X = df[['k-1d', 'k-2d', 'k-3d']]
    # X = df[['k-1w', 'k-2w', 'k-3w']]

    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    y = df['OCCUPANCY_FUTURE']
    # linear_reg_model = LinearRegression().fit(X, y)
    # report_parameters(linear_reg_model, step_size)


def main():
    # load main dataset
    dublin_bikes_df = pd.read_csv('D:\AAATrinity\Machine Learning\dublinbikes_20200101_20200401.csv',
                                  usecols=['NAME', 'TIME', 'AVAILABLE BIKES'], parse_dates=['TIME'], index_col="TIME")

    station_df = extract_station(dublin_bikes_df, 'CITY QUAY')
    # station_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

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
    df_copy = df.copy()
    cross_validate(df_copy)
    # cross_validate_poly(df_copy)
    train_test_model(df)
    # plot_predictions(df, X, y_pred, step, 'City Quay')
    # plot_predictions(df, X, y_pred, step, 'Brookfield Road')


if __name__ == '__main__':
    main()
