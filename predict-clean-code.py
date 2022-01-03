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
from sklearn.metrics import r2_score

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


def cross_validate_poly(df):
    # df = df[df['DAY_OF_WEEK'] < 5]
    original_df = df.copy()
    df.reset_index(inplace=True, drop=True)
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]

    y = df['OCCUPANCY_FUTURE']
    T = df['TIME']

    mean_list_ridge = []
    stdev_list_ridge = []
    r2 = []

    k_list = [1, 2, 3, 4]
    for degree in k_list:
        k_fold = KFold(n_splits=10)
        mean_sqaured_error_list = []
        r2_list = []
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        for train_index, test_index in k_fold.split(X_poly):
            model = Ridge(alpha=0, max_iter=2000000)
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # X_train, X_test = X_poly.iloc[train_index], X_poly.iloc[test_index]
            # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            ypred = model.predict(X_test)
            r2_list.append(model.score(X_test, y_test))
            mean_sqaured_error_list.append(math.sqrt(mean_squared_error(y_test, ypred)))
        print("RMSE :: " + str(degree) + " :: " + str(statistics.mean(mean_sqaured_error_list)))
        print("RMSE :: " + str(degree) + " :: " + str(statistics.stdev(mean_sqaured_error_list)))
        print("R2 :: " + str(degree) + " :: " + str(statistics.mean(r2_list)))
        r2.append(statistics.mean(r2_list))
        mean_list_ridge.append(statistics.mean(mean_sqaured_error_list))
        stdev_list_ridge.append(statistics.stdev(mean_sqaured_error_list))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(k_list, mean_list_ridge, color='blue', label='RMSE')
    ax.errorbar(k_list, r2, color='red', label='R2')
    ax.set_ylabel("Metric Scale")
    ax.set_xlabel("Polynomial Degree")
    ax.set_title("5 Folds with Ridge Regression")
    ax.legend(loc='upper right')
    plt.show()


def cross_validate_ridge_kfold(df):
    df.reset_index(inplace=True, drop=True)
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]

    y = df['OCCUPANCY_FUTURE']
    T = df['TIME']
    mean_list_ridge = []
    stdev_list_ridge = []
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    C_list_Ridge = [0.0001, 0.001, 0.01, 0.1, 1]
    for C in C_list_Ridge:
        k_fold = KFold(n_splits=5)
        mean_sqaured_error_list = []
        for train_index, test_index in k_fold.split(X_poly):
            # alpha=1/2*C
            model = Ridge(alpha=(1 / (2 * C)))
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            ypred = model.predict(X_test)
            mean_sqaured_error_list.append(mean_squared_error(y_test, ypred))
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


def no_cross_validate(df):
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    y = df['OCCUPANCY_FUTURE']
    X_train, X_test, yTrain, yTest = (train_test_split(X, y, random_state=12))
    linear_reg_model = LinearRegression().fit(X_train, yTrain)
    model = Ridge(fit_intercept=False).fit(X_train, yTrain)
    y_pred = model.predict(X)
    print(str.format("RMSE : {}", math.sqrt(mean_squared_error(y, y_pred))))
    print(str.format("R2 : {}", model.score(X_test, yTest)))
    return X, y_pred


def cross_validate_ridge(df):
    df.reset_index(inplace=True, drop=True)
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]

    y = df['OCCUPANCY_FUTURE']
    T = df['TIME']
    mean_list_ridge = []
    stdev_list_ridge = []
    poly = PolynomialFeatures(1)
    X_poly = poly.fit_transform(X)
    C_list_Ridge = [0.001, 0.01, 0.1, 1, 10]
    for C in C_list_Ridge:
        X_train, X_test, yTrain, yTest = (train_test_split(X_poly, y, random_state=0))
        model = Ridge(alpha=(1 / (2 * C)))
        model.fit(X_train, yTrain)
        ypred = model.predict(X_poly)
        mean_list_ridge.append(math.sqrt(mean_squared_error(y, ypred)))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(C_list_Ridge, mean_list_ridge, color='blue', label='RMSE')
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("C")
    ax.set_title("Ridge Cross Validation against Hyperparameter (C)")
    ax.legend(loc='upper right')
    plt.show()


def report_parameters(trained_model, step_q):
    feature_weights_table = PrettyTable(
        # ['Step(q)',
        # 'k-3-q', 'k-6-q', 'k-9-q',
        # 'k-1d', 'k-2d', 'k-3d',
        #  'k-1w', 'k-2w', 'k-3w'])
        # ['Step(q)',
        #  'k-1d', 'k-2d', 'k-3d'])
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


def predict_ridge(df, step, station_label):
    X = df[['k-3-q', 'k-6-q', 'k-9-q', 'k-1d', 'k-2d', 'k-3d', 'k-1w', 'k-2w', 'k-3w']]
    y = df['OCCUPANCY_FUTURE']
    T = df['TIME']
    poly = PolynomialFeatures(4)
    X_poly = poly.fit_transform(X)

    X_train, X_test, yTrain, yTest, t_train, t_test = (train_test_split(X_poly, y, T, random_state=0))
    model = Ridge().fit(X_train, yTrain)

    y_pred = model.predict(X_poly)

    plt.scatter(df.index, df['AVAILABLE BIKES'], color='blue')
    plt.scatter(T, y_pred, color='yellow')
    plt.xlabel("time(month-day hour)")
    plt.ylabel("Bike Occupancy")
    plt.title(
        station_label + " - " + str(
            step * 5) + " minutes ahead Predictions - With Polynomial Features (degree=4)", fontsize=10)
    plt.legend(["Training Data", "Predictions"], loc='lower right')
    plt.xlim([pd.to_datetime('2020-02-26', format='%Y-%m-%d'),
              pd.to_datetime('2020-02-28', format='%Y-%m-%d')])
    plt.xticks(rotation=15)
    plt.show()
    print(str.format("RMSE : {}", math.sqrt(mean_squared_error(y, y_pred))))
    print(str.format("R2 : {}", r2_score(y, y_pred)))
    report_parameters(model, step)


def predict_baseline(df, station_name):
    df['y_Pred'] = df["AVAILABLE BIKES"].shift(288 * 7)
    data = [df["AVAILABLE BIKES"], df["y_Pred"]]
    headers = ['y', 'y_Pred']
    df3 = pd.concat(data, axis=1, keys=headers)
    df3.dropna(inplace=True)
    df3.to_csv('baseline.csv')

    y = df3['y']
    y_pred = df3['y_Pred']
    plt.scatter(df.index, df['AVAILABLE BIKES'], color='blue')
    plt.scatter(df3.index, df3['y_Pred'], color='yellow')
    plt.xlabel("time(month-day hour)")
    plt.ylabel("Bike Occupancy")
    plt.title(
        station_name + " - " + str(
            12 * 5) + " minutes ahead Predictions - Without Polynomial Features", fontsize=10)
    plt.legend(["Training Data", "Predictions"], loc='lower right')
    plt.xlim([pd.to_datetime('2020-02-26', format='%Y-%m-%d'),
              pd.to_datetime('2020-02-28', format='%Y-%m-%d')])
    plt.xticks(rotation=15)
    plt.show()
    print(str.format("RMSE : {}", math.sqrt(mean_squared_error(y, y_pred))))
    print(str.format("R2 : {}", r2_score(y, y_pred)))


def main():
    # load main dataset
    dublin_bikes_df = pd.read_csv('D:\AAATrinity\Machine Learning\dublinbikes_20200101_20200401.csv',
                                  usecols=['NAME', 'TIME', 'AVAILABLE BIKES'], parse_dates=['TIME'], index_col="TIME")

    # station_df = extract_station(dublin_bikes_df, 'CITY QUAY')
    station_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

    range_df = select_datetime_range(station_df, '2020-02-04', '2020-03-14')
    original_df = select_features(range_df)

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
    # cross_validate_poly(df)
    # cross_validate_ridge_kfold(df)
    cross_validate_ridge(df)
    # predict_ridge(df, step, 'City Quay')
    # predict_ridge(df, step, 'Brookfield Road')

    # predict_baseline(range_df.copy(), 'City Quay')

    # no_cross_validate(df)

    # plot_predictions(df, X, y_pred, step, 'City Quay')
    # plot_predictions(df, X, y_pred, step, 'Brookfield Road')


if __name__ == '__main__':
    main()
