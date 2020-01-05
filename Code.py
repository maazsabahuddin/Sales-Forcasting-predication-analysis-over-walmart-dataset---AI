import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from datetime import datetime

# Simple project for AI - Machine Learning using Regression with KFold cross validation.

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 18)
sns.set(style="ticks", color_codes=True)

# Reading the data from the csv files
dataset = pd.read_csv("train.csv", names=['Store', 'Dept', 'Date', 'weeklySales', 'isHoliday'], sep=',', header=0)

features = pd.read_csv("features.csv",
                       names=['Store', 'Date', 'Temperature', 'Fuel_Price', 'Markdown1', 'Markdown2', 'Markdown4',
                              'Markdown5', 'CPI', 'Unemployment', 'IsHoliday'], sep=',', header=0).drop(
    columns=['IsHoliday'])

stores = pd.read_csv("stores.csrv", names=['Store', 'Type', 'Size'], sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')


# Helper Functions
def show_scatter_plot(data, column):
    plt.figure()
    plt.scatter(data[column], dataset['weeklySales'])
    plt.ylabel('weeklySales')
    plt.xlabel(column)
    plt.savefig(column + '.png')
    plt.show()


def show_simple_plot(data, title):
    data.plot(x='ID', y='Weekly_Sales', title=title)
    plt.savefig(title + '.png')
    plt.show()


def neural_networks_regressor():
    reg = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', verbose=3)
    return reg


def linear_regression():
    reg = LinearRegression()
    return reg


def extra_tree_regressor():
    reg = ExtraTreesRegressor(n_estimators=80, max_features='auto', verbose=1, n_jobs=1)
    return reg


def model_():
    return extra_tree_regressor()
    # return neural_networks_regressor()
    # return linear_regression()


def train_(x_train, y_train):
    m = model_()
    m.fit(x_train, y_train.values.ravel())
    return m


def predict_(m, test_x):
    return pd.Series(m.predict(test_x))


def train_and_predict(train_x, train_y, test_x):
    m = train_(train_x, train_y)
    return predict_(m, test_x), m


def calculate_error(test_y, predicted, weights):
    return mean_absolute_error(test_y, predicted, sample_weight=weights)


def append_in_file(filename, data):
    file = open(filename, "a")
    file.write("-" * 100 + "\n")
    file.write("Date & Time : {}".format(datetime.today()) + "\n")
    file.write("-" * 100 + "\n")
    file.write(data)
    file.write("-" * 100 + "\n")
    file.close()


# Exploring the data using the matplotlib
# show_scatter_plot(dataset, 'Size')
# show_scatter_plot(dataset, 'isHoliday')
# show_scatter_plot(dataset, 'Fuel_Price')
# show_scatter_plot(dataset, 'CPI')
# show_scatter_plot(dataset, 'Temperature')
# show_scatter_plot(dataset, 'Type')
# show_scatter_plot(dataset, 'Unemployment')
# show_scatter_plot(dataset, 'Dept')
# show_scatter_plot(dataset, 'Store')

# dataset = dataset.loc[dataset.index <= 20000]
# sns_plot = sns.pairplot(dataset.fillna(0), vars=['weeklySales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'])
# plt.show()


dataset = pd.get_dummies(dataset, columns=["Type"])
# Setting 0.0 to MarkDown(1-5) NaN values
dataset = dataset.fillna(0)
# Creating a new Column 'Month' from Date
dataset['Month'] = pd.to_datetime(dataset['Date']).dt.month
dataset = dataset.drop(columns=['Date', 'CPI', 'Fuel_Price', 'Unemployment'])

# Splitting the data into test and train
x_train = dataset[['Store', 'Dept', 'isHoliday', 'Size', 'Temperature', 'Markdown1', 'Markdown2', 'Markdown4',
                   'Markdown5', 'Type_A', 'Type_B', 'Type_C', 'Month']]
y_train = dataset[['weeklySales']]

start_time = datetime.now()
# Using the KFold Cross Validation to get the best model for Predictions
kf = KFold(n_splits=3)
splitted = []
for name, group in dataset.groupby(["Store", "Dept"]):
    group = group.reset_index(drop=True)
    trains_x = []
    trains_y = []
    tests_x = []
    tests_y = []
    if group.shape[0] <= 3:
        f = np.array(range(3))
        np.random.shuffle(f)
        group['fold'] = f[:group.shape[0]]
        continue
    fold = 0
    for train_index, test_index in kf.split(group):
        group.loc[test_index, 'fold'] = fold
        fold += 1
    splitted.append(group)

splitted = pd.concat(splitted).reset_index(drop=True)
print(splitted)

best_model = None
error_cv = 0
best_error = np.iinfo(np.int32).max
for fold in range(3):
    dataset_train = splitted.loc[splitted['fold'] != fold]
    dataset_test = splitted.loc[splitted['fold'] == fold]
    train_y = dataset_train['weeklySales']
    train_x = dataset_train.drop(columns=['weeklySales', 'fold'])
    test_y = dataset_test['weeklySales']
    test_x = dataset_test.drop(columns=['weeklySales', 'fold'])
    print(dataset_train.shape, dataset_test.shape)
    predicted, model = train_and_predict(train_x, train_y, test_x)
    weights = test_x['isHoliday'].replace(True, 3).replace(False, 1)
    error = calculate_error(test_y, predicted, weights)
    error_cv += error
    print(fold, error)
    if error < best_error:
        print('Find best Model')
        best_error = error
        best_model = model
error_cv /= 3
print("Error cv:", error_cv)
print("Best Error:", best_error)

print("--Model Trained Successfully--")

model_name = type(best_model).__name__

# Loading Testing Dataset
dataset_test = pd.read_csv("test.csv", names=['Store', 'Dept', 'Date', 'isHoliday'], sep=',', header=0)
features = pd.read_csv("features.csv", sep=',', header=0,
                       names=['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown4',
                              'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']).drop(columns=['IsHoliday'])
stores = pd.read_csv("stores.csv", names=['Store', 'Type', 'Size'], sep=',', header=0)
dataset_test = dataset_test.merge(stores, how='left').merge(features, how='left')

dataset_test = pd.get_dummies(dataset_test, columns=["Type"])
dataset_test = dataset_test.fillna(0)
column_date = dataset_test['Date']
dataset_test['Month'] = pd.to_datetime(dataset_test['Date']).dt.month
dataset_test = dataset_test.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment'])

# Splitting the data into x and y
x_test = dataset_test
# Predicting the Weekly Sales using the Test Data
y_predict = best_model.predict(x_test)
dataset_test['weeklySales'] = y_predict
dataset_test['Date'] = column_date
data_ids = dataset_test['id'] = dataset_test['Store'].astype(str) + '_' + dataset_test['Dept'].astype(str) + '_' + \
                                dataset_test['Date'].astype(str)
dataset_test = dataset_test[['id', 'weeklySales']]
dataset_test = dataset_test.rename(columns={'id': 'ID', 'weeklySales': 'Weekly_Sales'})
stop_time = datetime.now()

# Showing the simple plot
show_simple_plot(dataset_test, model_name)
# plt.scatter(dataset_test['ID'], dataset_test['Weekly_Sales'])
# plt.show()

# Writing predictions to csv
dataset_test.to_csv('predictions_using_' + model_name + '.csv', index=False)
total_time = stop_time - start_time
print("Total processing time: " + str(total_time))
data = "Error using {} = {}".format(model_name, best_error) + "\nTotal processing time: {}".format(total_time) + "\n"
append_in_file("Information.txt", data)
