import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, precision_score, \
    recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import joblib
import os

dataset_train = pd.read_csv('Data/train.txt', sep=' ', header=None).drop([26, 27], axis=1) # cols 26 and 27 are useless
# print(dataset_train.head())
print(dataset_train.describe())
col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19','s20','s21']
dataset_train.columns = col_names

# importing the test dataset
# 13096 observations with 26 features
dataset_test = pd.read_csv('Data/test.txt', sep=' ', header=None).drop([26, 27], axis=1)
dataset_test.columns = col_names

# importing the truth dataset
# 100 rows 1 column: it represents the remaining useful life (RUL) per each motors in the test dataset
truth = pd.read_csv('Data/truth.txt', sep=' ', header=None).drop([1], axis=1)
truth.columns = ['more']
truth = truth.set_index(truth.index + 1)

### CALCULATE RUL TRAIN ###
dataset_train['RUL']=dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']

### CALCULATE RUL TEST ###
truth['max'] = dataset_test.groupby('id')['cycle'].max() + truth['more']
dataset_test['RUL'] = [truth['max'][i] for i in dataset_test.id] - dataset_test['cycle']

df_train = dataset_train.copy()
df_test = dataset_test.copy()

features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                   's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name = 'RUL'

### SCALING DATA ###
sc = MinMaxScaler(feature_range=(0, 1))  # preferable btw 0 and 1
df_train[features_col_name] = sc.fit_transform(df_train[features_col_name])
df_test[features_col_name] = sc.transform(df_test[features_col_name])
df_train = df_train.dropna(axis=1)

cols_sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11',
                   's12', 's13', 's14', 's15', 's17', 's20', 's21']
target = ['RUL']

# explore = sns.PairGrid(data=train_df.query('id < 2') ,
#                  x_vars='RUL',
#                  y_vars=cols_sensors,
#                  hue="id", height=2, aspect=3)
# explore = explore.map(plt.scatter, alpha=0.5)
# explore = explore.set(xlim=(400,0))
# explore = explore.add_legend()
# plt.show()

X_train = df_train[cols_sensors]
y_train = df_train[target]
y_train = np.squeeze(np.asarray(y_train))
X_test = np.concatenate(([df_test[df_test['id']==id][cols_sensors].values[-1:] for id in df_test['id'].unique()]))
X_test = np.asarray(X_test)

y_mask = [len(df_test[df_test['id']==id]) >= 0 for id in df_test['id'].unique()]
y_test = df_test.groupby('id')['RUL'].nth(-1)[y_mask].values
y_test = y_test.reshape(y_test.shape[0],1).astype(np.float32)
y_test = np.squeeze(np.asarray(y_test))


## LINEAR REGRESSION ###
cv = 5
SEED = 42
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lr = LinearRegression()

params_lr = {
    'fit_intercept': [True, False]  # True: the y-intercept will be found otherwise it will be set to 0 (best: true)
}
optimized_lr = GridSearchCV(estimator=lr,
                            cv=cv,
                            param_grid=params_lr,
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            n_jobs=-1,
                            refit=True
                           )
optimized_lr.fit(X_train, y_train)
y_pred = optimized_lr.predict(X_test)
print("Linear Regression Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Linear Regression Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Linear Regression r-squared: ", r2_score(y_test, y_pred))



## RANDOMFOREST REGRESSION ###
rf = ensemble.RandomForestRegressor(n_estimators = 100, max_features=5, bootstrap=True)
params_rf = [
{'n_estimators': [3, 10, 30, 50, 100], 'max_features': [5, 10, 13, 14]},
{'bootstrap': [False], 'n_estimators': [3, 10, 50, 100], 'max_features': [6, 8, 12]},
] # n_estimator = number of trees
{'max_features': 5, 'n_estimators': 100}
optimized_rf = GridSearchCV(estimator = rf,
                            cv=5,
                            param_grid=params_rf,
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            n_jobs=-1,
                            refit=True
                           )
# optimized_rf.fit(X_train, y_train)
# y_pred = optimized_rf.predict(X_test)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Regression Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Random Forest Regression Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Random Forest Regression r-squared: ", r2_score(y_test, y_pred))


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 1, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual RUL')
ax.set_ylabel('Predicted RUL')
ax.set_title('Remaining Useful Life Actual vs. Predicted')
plt.show()