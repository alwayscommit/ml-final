import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True

dublin_bikes_df = pd.read_csv('D:\AAATrinity\Machine Learning\dublinbikes_20200101_20200401.csv',
                              usecols=['NAME', 'TIME', 'AVAILABLE BIKES', 'BIKE STANDS'], parse_dates=['TIME'],
                              index_col="TIME")


def extract_station(dublin_bikes_df, station_name):
    return dublin_bikes_df[dublin_bikes_df['NAME'] == station_name]


main_df = extract_station(dublin_bikes_df, 'CITY QUAY')
suburb_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

# feature engineering

# specify start date and end date between the date range present in the dataset
start_date = '2020-02-03'
end_date = '2020-02-21'

main_df = main_df[(main_df.index >= start_date) & (main_df.index <= end_date)]
suburb_df = suburb_df[(suburb_df.index >= start_date) & (suburb_df.index <= end_date)]

# TODO remove year from X label
plt.scatter(main_df.index, main_df['AVAILABLE BIKES'], color='blue', marker='.', label='City Quay')
plt.scatter(suburb_df.index, suburb_df['AVAILABLE BIKES'], color='yellow', marker='.', label='Brookfield Road')
plt.title('Three Week Data indicating Weekly Pattern')
plt.xlabel('Date')
plt.ylabel('Number of Available Bikes')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.xticks(rotation=15)
plt.show()


