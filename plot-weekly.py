import pandas as pd
import matplotlib.pyplot as plt
import datetime

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

title = ['Jan 25 (Saturday) to Jan 31 (Friday)', 'Feb 01 (Saturday) to Feb 07 (Friday)',
         'March 01 (Sunday) to March 07 (Saturday)']
start_date = ['2020-01-25', '2020-02-01', '2020-03-01']
end_date = ['2020-01-31', '2020-02-07', '2020-03-07']

for i in range(0, 3):
    df = main_df[(main_df.index >= start_date[i]) & (main_df.index <= end_date[i])]
    X = df.index
    y = df['OCCUPANCY']
    plt.figure(i)
    # TODO remove year from X label
    plt.plot(X, y, color='blue', label='Bike Occupancy')
    plt.title('Dublin Bikes data for ' + title[i])
    plt.xlabel('Time (Days)')
    plt.ylabel('Occupancy (%)')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.xlim([pd.to_datetime(start_date[i], format='%Y-%m-%d'),
              pd.to_datetime(end_date[i], format='%Y-%m-%d')])
    plt.xticks(rotation=15)

plt.show()
