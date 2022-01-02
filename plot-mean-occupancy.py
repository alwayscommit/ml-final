import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True

def extract_station(dublin_bikes_df, station_name):
    return dublin_bikes_df[dublin_bikes_df['NAME'] == station_name]

dublin_bikes_df = pd.read_csv('D:\AAATrinity\Machine Learning\dublinbikes_20200101_20200401.csv',
                                  usecols=['NAME', 'BIKE STANDS', 'TIME', 'AVAILABLE BIKES'], parse_dates=['TIME'], index_col="TIME")

# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
main_df = extract_station(dublin_bikes_df, 'CITY QUAY')
suburb_df = extract_station(dublin_bikes_df, 'BROOKFIELD ROAD')

main_df['OCCUPANCY'] = main_df['AVAILABLE BIKES'] / main_df['BIKE STANDS']
main_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1, inplace=True)
main_df['DAY OF WEEK'] = main_df.index.dayofweek
day_of_week_main_df = main_df.groupby(['DAY OF WEEK']).mean()

suburb_df['OCCUPANCY'] = suburb_df['AVAILABLE BIKES'] / suburb_df['BIKE STANDS']
suburb_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1, inplace=True)
suburb_df['DAY OF WEEK'] = suburb_df.index.dayofweek
day_of_week_suburb_df = suburb_df.groupby(['DAY OF WEEK']).mean()

plt.figure(1)


plt.plot(day_of_week_main_df.index, day_of_week_main_df['OCCUPANCY'], color='yellow', marker='.', label='City Quay')
plt.plot(day_of_week_suburb_df.index, day_of_week_suburb_df['OCCUPANCY'], color='blue', marker='.', label='Brookfield Road')
plt.title('Week-wise Occupancy(mean) from 1st Jan to 1st April')
plt.xlabel('Day of Week')
plt.ylabel('Occupancy (%)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))


main_df['HOUR'] = main_df.index.hour
hour_main_df = main_df.groupby(['HOUR']).mean()

suburb_df['HOUR'] = suburb_df.index.hour
hour_suburb_df = suburb_df.groupby(['HOUR']).mean()

plt.figure(2)


plt.plot(hour_main_df.index, hour_main_df['OCCUPANCY'], color='yellow', marker='.', label='City Quay')
plt.plot(hour_suburb_df.index, hour_suburb_df['OCCUPANCY'], color='blue', marker='.', label='Brookfield Road')
plt.title('Hour-wise Occupancy(mean) from 1st Jan to 1st April')
plt.xlabel('Hour')
plt.ylabel('Occupancy (%)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.show()
