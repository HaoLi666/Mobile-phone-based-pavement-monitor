import numpy as np
import pandas as pd
# from sklearn.neural_network import MLPRegressor
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acces = []

    def on_batch_end(self, batch, logs={}):
        self.acces.append(logs.get('acc'))


data = pd.read_csv('data/ARI/ARI_PCI.csv')
data_z = np.array(data['z'])
data_PCI = np.array(data['PCI'])

train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)
X_train = np.array(train_data['z'])
X_test = np.array(test_data['z'])
y_train = np.array(train_data['PCI'])
y_test = np.array(test_data['PCI'])


def transfer(x, y):
    new_x = []
    for item in x:
        new_item = item.split(',')
        the_item = []
        for it in new_item:
            the_item.append(float(it))
        new_x.append(the_item)
    new_y = []
    for item in y:
        new_item = float(item)
        new_y.append(new_item/10)

    return np.array(new_x), np.array(new_y)


X_train, y_train = transfer(X_train, y_train)
X_test, y_test = transfer(X_test, y_test)
# data_X = []
# for item in data_z:
#     new_item = item.split(',')
#     the_item = []
#     for it in new_item:
#         the_item.append(float(it))
#     data_X.append(the_item)
# data_X = np.array(data_X)
# # print(data_X)
#
# data_y = []
# for item in data_PCI:
#     new_item = float(item)
#     data_y.append(new_item/10)
# data_y = np.array(data_y)
# print(data_y)

# one hot encoding
# df_y = pd.DataFrame(data_y)
# df_y.columns = ['PCI']
# df_y = pd.get_dummies(df_y)
# print(df_y)

# X_train, X_test = data_X[:200], data_X[200:]
# y_train, y_test = data_y[:200], data_y[200:]

# print(len(y_train))

# regression
# ================= build model =================
def build_model(m):
    model = keras.Sequential([
        layers.Dense(m+1, activation=tf.nn.sigmoid, input_shape=(40, )),
        # layers.Dense(64, activation=tf.nn.sigmoid),
        layers.Dense(1, use_bias=True)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.SGD(0.01)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model(len(X_train))

# ================= inspect model =================
print(model.summary())

example_batch = X_train[:10]
example_result = model.predict(example_batch)
print(example_result)


# ================= train model =================
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 10

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ')  # configuration
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')  # configuration
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# plot_history(history)

model = build_model(len(X_test))

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_test, y_test, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))

# ================= evaluating =================
test_predictions = model.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()


# ANN tf
# ======================== trainning test ========================
model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=(40, )),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = LossHistory()
model.fit(X_train, y_train, epochs=5, callbacks=[history])

print(history.acces)
plt.plot(range(35), history.acces)
plt.xlabel("BATCH & EPOCH")
plt.ylabel("accurate")
plt.show()

# evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# ================= evaluating =================
test_predictions = model.predict_classes(X_test)

error = test_predictions - y_test

plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()

test_err = 1 - test_acc
plt.barh(range(2), [test_acc, test_err], height=0.7, color='steelblue', alpha=0.8)
plt.yticks(range(2), ['predict_right', 'predict_error'])
plt.xlim(0, 1)
plt.xlabel("Predict Rate")
plt.show()

# # ANN skilearn
# mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=1200, alpha=1e-5, solver='lbfgs')
# mlp.fit(X_train, y_train)
# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))

# KNN
# neigh = KNeighborsClassifier(n_neighbors=2)
# neigh = neigh.fit(X_train, y_train)
# print("Training set score: %f" % neigh.score(X_train, y_train))
# print("Test set score: %f" % neigh.score(X_test, y_test))
