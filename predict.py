import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
import math, sys
from prettytable import PrettyTable

feature_weights_table = PrettyTable(['Step(q)', 'Intercept', 'k-3-q', 'k-2-q', 'k-1-q'])

def report_parameters(trained_model, step_q):
    feature_weights_table.add_row(
        [step_q, trained_model.intercept_, trained_model.coef_[0], trained_model.coef_[1],
         trained_model.coef_[2]])

plt.rc('font', size=10)
plt.rcParams['figure.constrained_layout.use'] = True

# dublin_bikes = pd.read_csv('/content/drive/MyDrive/ML Final/dublinbikes_20200101_20200401.csv')
# feature selection
main_df = pd.read_csv('./station_dataset/grand_canal_dock.csv',
                      usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
                      parse_dates=['TIME'], index_col="TIME")
# main_df = pd.read_csv('./station_dataset/brookfield-road.csv',
#                         usecols=['TIME', 'AVAILABLE BIKES', 'BIKE STANDS'],
#                         parse_dates=['TIME'], index_col="TIME")

# feature engineering
main_df['OCCUPANCY'] = main_df['AVAILABLE BIKES'] / main_df['BIKE STANDS']
main_df.drop(['AVAILABLE BIKES', 'BIKE STANDS'], axis=1, inplace=True)

# main_df = main_df[(main_df.index >= '2020-02-04') & (main_df.index <= '2020-03-14')]

t_original = main_df.index
y_original = main_df['OCCUPANCY']

# 10 minutes in the future
# step = 2
# 30 minutes in the future
# step = 6
# 60 minutes in the future
step = 2

# number of points to pick
lag = 3

# per 5 minutes
dd = 1

# dataset
time_series = pd.Series(main_df.index.format())

# to output csv with features for manual verification
feature_df = time_series[0:y_original.size - step - (lag * dd)]

# X is the Feature Set
X = None
for i in range(0, lag):
    feature = y_original[i * dd:y_original.size - step - ((lag - i) * dd)]
    feature_df = np.column_stack((feature_df, feature))
    if X is None:
        X = feature
    else:
        X = np.column_stack((X, feature))

y = y_original[lag * dd + step::1]
t = time_series[lag * dd + step::1]

feature_df = np.column_stack((feature_df, y))
feature_df = np.column_stack((feature_df, t))

feature_df = pd.DataFrame(feature_df,
                          columns=['Original Time', 'k-3-q', 'k-2-q', 'k-1-q', 'Occupancy (y)', 'Occupancy Time'])
feature_df.to_csv('grand-canal-dock-dataset-2-step.csv')

from sklearn.model_selection import train_test_split

train, test = train_test_split(np.arange(0, y.size), test_size=0.2)

from sklearn.linear_model import Ridge

model = Ridge(fit_intercept=False).fit(X[train], y[train])

report_parameters(model, step)
print(feature_weights_table)

y_pred = model.predict(X[test])
print(math.sqrt(mean_squared_error(y[test], y_pred)))

plt.scatter(t_original, y_original, color='blue')
plt.scatter(t.to_numpy()[test], y_pred, color='yellow')
plt.xlabel("time(days)")
plt.ylabel("Bike Occupancy")
plt.legend(["training data", "predictions"], loc='upper right')
plt.xlim([pd.to_datetime('2020-03-23', format='%Y-%m-%d'),
              pd.to_datetime('2020-03-29', format='%Y-%m-%d')])
plt.show()
