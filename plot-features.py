import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True

# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
# feature selection
main_df = pd.read_csv('./station_dataset/grand_canal_dock.csv',
                      usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
                      parse_dates=['TIME'], index_col="TIME")
suburb_df = pd.read_csv('./station_dataset/brookfield-road.csv',
                        usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
                        parse_dates=['TIME'], index_col="TIME")

# feature engineering
main_df['OCCUPANCY'] = main_df['AVAILABLE BIKES'] / main_df['BIKE STANDS']
main_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1, inplace=True)

# specify start date and end date between the date range present in the dataset
start_date = '2020-01-01'
end_date = '2020-04-01'

df = main_df[(main_df.index >= start_date) & (main_df.index <= end_date)]

X = df.index
y = df['OCCUPANCY']

# TODO remove year from X label
plt.scatter(X, y, color='red', marker='.',  label='Bike Occupancy')
plt.title('Dublin Bikes data from January to March 2020')
plt.xlabel('Date')
plt.ylabel('Occupancy (%)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.xticks(rotation=10)
plt.show()
