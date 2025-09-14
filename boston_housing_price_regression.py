from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
mean = x_train.mean(axis = 0)

std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
mse, mae = model.evaluate(x_test, y_test, verbose=2)
print("Средняя абсолютная ошибка (тысяч долларов):", mae)
pred = model.predict(x_test)
print("Предсказанная стоимость", pred[24][0], ", правильная стоимость:", y_test[24])
for i in range(0, 101):
  pred = model.predict(x_test)
  print("Предсказанная стоимость", pred[i][0], ", правильная стоимость:", y_test[i])
  i+1
