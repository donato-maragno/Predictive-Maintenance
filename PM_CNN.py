import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import joblib

train_df = pd.read_csv('Data/train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])

### LOAD TEST ###
test_df = pd.read_csv('Data/test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

### LOAD GROUND TRUTH ###
truth_df = pd.read_csv('Data/truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
truth_df.columns = ['more']
truth_df = truth_df.set_index(truth_df.index + 1)

### CALCULATE RUL TRAIN ###
train_df['RUL']=train_df.groupby(['id'])['cycle'].transform(max)-train_df['cycle']

### ADD NEW LABEL TRAIN ###
w1 = 45
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2


### SCALE TRAIN DATA ###
def scale(df):
    return (df - df.min()) / (df.max() - df.min())

for col in train_df.columns:
    if col[0] == 's':
        train_df[col] = scale(train_df[col])

train_df = train_df.dropna(axis=1)

### CALCULATE RUL TEST ###
truth_df['max'] = test_df.groupby('id')['cycle'].max() + truth_df['more']
test_df['RUL'] = [truth_df['max'][i] for i in test_df.id] - test_df['cycle']

### ADD NEW LABEL TEST ###
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

### SCALE TEST DATA ###
for col in test_df.columns:
    if col[0] == 's':
        test_df[col] = scale(test_df[col])

test_df = test_df.dropna(axis=1)

sequence_length = 30

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

### SEQUENCE COL: COLUMNS TO CONSIDER ###
sequence_cols = []
for col in train_df.columns:
    if col[0] == 's':
        sequence_cols.append(col)

### GENERATE X TRAIN TEST ###
X_test0 = []
for engine_id in train_df.id.unique():
    for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
        X_test0.append(sequence)

seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

# generate sequences and convert to numpy array
X_train = np.concatenate(list(seq_gen)).astype(np.float32)
X_test = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:]
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
X_test = np.asarray(X_test)
X_test0 = np.asarray(X_test0)

### GENERATE Y TRAIN TEST ###
y_train, y_test0 = [], []
for engine_id in train_df.id.unique():
    for label in gen_labels(train_df[train_df.id == engine_id], sequence_length, ['label2']):
        y_train.append(label)
    for label in gen_labels(test_df[test_df.id == engine_id], sequence_length, ['label2']):
        y_test0.append(label)

y_train = np.asarray(y_train).reshape(-1, 1)

y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
y_test = test_df.groupby('id')['label2'].nth(-1)[y_mask].values
y_test = y_test.reshape(y_test.shape[0],1).astype(np.float32)
y_test0 = np.asarray(y_test0).reshape(-1, 1)

### ENCODE LABEL ###
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_test0 = to_categorical(y_test0)

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def plotting ():
    plt.figure(figsize=(10, 10))
    for i in range(0, 17):
        plt.subplot(3, 6, i + 1)
        rec = rec_plot(X_train[0, :, i])
        plt.imshow(rec)
        plt.title(sequence_cols[i])
    plt.show()
# plotting()

### TRANSFORM X TRAIN TEST IN IMAGES ###
X_train_img = np.apply_along_axis(rec_plot, 1, X_train).astype('float16')
X_test_img = np.apply_along_axis(rec_plot, 1, X_test).astype('float16')
X_test_img0 = np.apply_along_axis(rec_plot, 1, X_test0).astype('float16')

### MODEL ###
# model_loaded = joblib.load("PM_CNN.pkl")
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 17))) #tanh and Sigmoid soffer of vanishing gradient
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=6)

model.fit(X_train_img, y_train, batch_size=512, epochs=25, callbacks=[es],
          validation_split=0.2, verbose=1)
joblib.dump(model,"PM_CNN.pkl")
scores = model.evaluate(X_test_img0, y_test0, verbose=1)
print('Accurracy: {}'.format(scores[1]))

y_pred0 = model.predict_classes(X_test_img0)
print(classification_report(np.where(y_test0 != 0)[1], y_pred0))
print(confusion_matrix(np.where(y_test0 != 0)[1], y_pred0))

y_pred = model.predict_classes(X_test_img)
print(classification_report(np.where(y_test != 0)[1], y_pred))
print(confusion_matrix(np.where(y_test != 0)[1], y_pred))
