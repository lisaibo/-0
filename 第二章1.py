import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("train.csv")
train_x = np.array(data["heartbeat_signals"].str.split(",", expand=True)).astype("float32").reshape(-1,205,1)
train_y = np.array(data["label"].astype("int8"))
data1 = pd.read_csv("testA.csv")
test_x = np.array(data1['heartbeat_signals'].str.split(',', expand=True)).astype("float32").reshape(-1,205,1)
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(train_x.reshape(-1,205), train_y)
y_preknn = tf.one_hot(knn.predict(test_x.reshape(-1,205)), depth=4)
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same',
                 input_shape=(205, 1),  activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4,activation='softmax')
])
model1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc']
                )
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=9, padding='same',
                 input_shape=(205, 1),  activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=6, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4,activation='softmax')
])
model2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc']
                )
model1.fit(train_x, train_y, epochs=30, batch_size=200,validation_split=0.1)
model2.fit(train_x, train_y, epochs=30, batch_size=200,validation_split=0.1)
y_pre1 = model1.predict(test_x)
y_pre2 = model2.predict(test_x)

y_pre = np.array(y_pre1 + y_pre2 + y_preknn)
a = tf.one_hot(y_pre.argmax(axis=1), depth=4).numpy()
t = pd.DataFrame()
t["id"] = data1["id"]
t["label_0"] = a[:, 0]
t["label_1"] = a[:, 1]
t["label_2"] = a[:, 2]
t["label_3"] = a[:, 3]
t.to_csv("sample_submit.csv", index=None)
