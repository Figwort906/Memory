import uwb_dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, CSVLogger

# import raw data
data = uwb_dataset.import_from_files()

# divide CIR by RX preable count (get CIR of single preamble pulse)
# item[2] represents number of acquired preamble symbols
for item in data:
    item[15:] = item[15:]/float(item[2])

# test data
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0]
x_test = data[30000:, 15:]
y_test = data[30000:, 0]

# feed data
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

Nnew=[]
for item in x_train:
    item = item[max([0,item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_train = np.asarray(Nnew)

Nnew=[]
for item in x_test:
    item = item[max([0,item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_test = np.asarray(Nnew)

Nnew=[]
for item in x_val:
    item = item[max([0,item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_val = np.asarray(Nnew)

# total 1016 data for CIR
# define model
model = Sequential()
model.add(Embedding(2700, 128, input_length=100))
model.add(Conv1D(filters=10, kernel_size=4, padding='valid', activation='relu'))
model.add(Conv1D(20, 5, padding='valid', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
model.add(Conv1D(20, 4, padding='valid', activation='relu'))
model.add(Conv1D(40, 4, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# 定义 CSVLogger 来保存日志
csv_logger = CSVLogger('training_log.csv', append=True)

# compile，早期停止
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 训练模型并保存日志
hist = model.fit(x_train, y_train, epochs=9, batch_size=64, validation_data=(x_val, y_val))

# evaluation
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)
