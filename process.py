import pandas as pd
import os

# figure out the amount of activity based on updated time for each station
# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
dublin_bikes = pd.read_csv('station_dataset/dublinbikes_20200101_20200401.csv')
dublin_bikes.groupby('STATION ID').count()
dublin_bikes_dropped = dublin_bikes.drop(['TIME'], axis=1)
dublin_bikes_dropped = dublin_bikes_dropped.drop_duplicates(keep='first').groupby(
    ['STATION ID', 'NAME']).size().reset_index(
    name='Activity Count').sort_values(by=['Activity Count'], ascending=False)

station = dublin_bikes[dublin_bikes["NAME"] == "KILMAINHAM GAOL"]

outdir = './station_dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)
station.to_csv("./station_dataset/kilmainham-gaol.csv")
print("Station datasets saved.")
