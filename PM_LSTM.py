import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
plt.style.use('ggplot')

# to see al the cols and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# importing the train dataset
# 20631 observations with 26 features
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
dataset_train['RUL'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']

### ADD NEW LABEL TRAIN ###
w1 = 50
w0 = 10
dataset_train['label1'] = np.where(dataset_train['RUL'] <= w1, 1, 0 )
dataset_train['label2'] = dataset_train['label1']
dataset_train.loc[dataset_train['RUL'] <= w0, 'label2'] = 2

### CALCULATE RUL TEST ###
truth['max'] = dataset_test.groupby('id')['cycle'].max() + truth['more']
dataset_test['RUL'] = [truth['max'][i] for i in dataset_test.id] - dataset_test['cycle']

### ADD NEW LABEL TEST ###
dataset_test['label1'] = np.where(dataset_test['RUL'] <= w1, 1, 0)
dataset_test['label2'] = dataset_test['label1']
dataset_test.loc[dataset_test['RUL'] <= w0, 'label2'] = 2

df_train = dataset_train.copy()
df_test = dataset_test.copy()

features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                   's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name = 'label'

# plotting
# plot_cols = ['s1', 's2', 's3', 's4', 's5']
# ax1 = df_train[plot_cols].plot(subplots=True, sharex=True, figsize=(20, 20))
# plt.show();

# feature scaling
###################### TRY OTHERS #####################
sc = MinMaxScaler(feature_range=(0, 1))  # preferable btw 0 and 1
df_train[features_col_name] = sc.fit_transform(df_train[features_col_name])
df_test[features_col_name] = sc.transform(df_test[features_col_name])


#window size
seq_length = 30  # if it was 1 we have X_train.shape = (20531, 1, 24)

seq_cols = features_col_name

# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]
# generator for the sequences
seq_gen = (list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols))
           for id in df_train['id'].unique())

# generate sequences and convert to numpy array
X_train = np.concatenate(list(seq_gen)).astype(np.float32)

X_test = [df_test[df_test['id']==id][seq_cols].values[-seq_length:]
                       for id in df_test['id'].unique() if len(df_test[df_test['id']==id]) >= seq_length]
X_test = np.asarray(X_test).astype(np.float32)
# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

y_train = []
for engine_id in df_train.id.unique():
    for label in gen_labels(df_train[df_train.id == engine_id], seq_length, ['label2']):
        y_train.append(label)
y_train = np.asarray(y_train).reshape(-1, 1)

y_mask = [len(df_test[df_test['id']==id]) >= seq_length for id in df_test['id'].unique()]
y_test = df_test.groupby('id')['label2'].nth(-1)[y_mask].values
y_test = y_test.reshape(y_test.shape[0],1).astype(np.float32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

### LSTM MODEL ###
""" we build a deep network. The first layer is an LSTM layer with 100 units followed by another
LSTM layer with 50 units. Dropout is also applied after each LSTM layer to control overfitting. Final layer is a
Dense output layer with single unit and sigmoid activation since this is a binary classification problem. """
nb_features = X_train.shape[2]
timestamp = seq_length
model_loaded = joblib.load("LSTM_model.pkl")
model = Sequential()

################### TRY TO MODIFY ##################
model.add(LSTM(
         input_shape=(timestamp, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))  # in order to reduce the overfitting and ustable gradient problem

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=3, activation='sigmoid'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# fit the network
# model.fit(X_train, y_train, epochs=25, batch_size=200, validation_split=0.05, verbose=1,
#           callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')])

# training metrics
# scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
# print('Train Accurracy: {}'.format(scores[1]))

y_pred=model_loaded.predict_classes(X_test)
print(classification_report(np.where(y_test != 0)[1], y_pred))
print(confusion_matrix(np.where(y_test != 0)[1], y_pred))
# joblib.dump(model, "LSTM_model.pkl")
# test example
def prob_failure(machine_id, nb_features, timestamp):
    machines_df = df_test[df_test.id == machine_id]
    if (len(machines_df)< timestamp):
        print("try with a different engine")
        failure_prob = -1
    else:
        machines_test = list(gen_sequence(machines_df, seq_length, seq_cols))
        machines_test = np.array(machines_test).reshape(len(machines_test), timestamp, nb_features)
        m_pred1= model_loaded.predict_proba(machines_test)[:,1]
        failure_prob = m_pred1[-1]*100 # because I only care about the last time window which is representative of the actual situation
    return failure_prob

machine_id = 3
print('Probability that engine', machine_id, 'will fail within w1: ', prob_failure(machine_id, nb_features, timestamp), "%")