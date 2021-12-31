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


def extract_station(dublin_bikes_df, station_name):
    return dublin_bikes_df[dublin_bikes_df['NAME'] == station_name]


def select_features(station_df):
    # drop NAME as is it not required
    df = station_df.drop(columns=['NAME'])
    # df.reset_index(drop=True, inplace=True)

    #to perform shift operations on this column
    df['TIME'] = df.index

    # add a unix timestamp column
    # df['TIMESTAMP'] = pd.array(df.index.astype(np.int64)) / 1000000000

    # time in days
    # df['TIME_IN_DAYS'] = (df['TIMESTAMP'] - df['TIMESTAMP'][0]) / 60 / 60 / 24

    # add a day of week column
    df['DAY_OF_WEEK'] = df.index.dayofweek

    # filtering out weekends
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

def extract_feature_set(df, step_size):
    return

def main():
    # load main dataset
    dublin_bikes_df = pd.read_csv('./station_dataset/dublinbikes_20200101_20200401.csv',
                                  usecols=['NAME', 'TIME', 'AVAILABLE BIKES'], parse_dates=['TIME'], index_col="TIME")

    station_df = extract_station(dublin_bikes_df, 'GRAND CANAL DOCK')
    # station_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

    df = select_datetime_range(station_df, '2020-02-04', '2020-03-14')
    df = select_features(df)

    # plot_date_range(df)

    extract_feature_set(df, step_size=2)


if __name__ == '__main__':
    main()
