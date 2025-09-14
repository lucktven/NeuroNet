!git clone https://github.com/keras-team/keras-tuner.git
%cd keras-tuner
!pip install .
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from google.colab import files
from kerastuner import RandomSearch, Hyperband, BayesianOptimization
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape (60000, 784)
x_test = x_test. reshape (10000, 784)
x_train = x_train / 255
x_test = x_test / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical (y_test, 10)
def build_model (hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add (Dense (units=hp. Int ( 'units_input',
                                        min_value=512,
                                        max_value=1024,
                                        step=32),
                      input_dim=784,
                      activation=activation_choice))
    model.add(Dense(units=hp.Int('units hidden' ,
                                        min_value=128,
                                        max_value=600,
                                        step=32),
                      activation=activation_choice))
    model.add (Dense(10, activation='softmax'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy' ])
    return model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=80,
    directory='test_directory'
    )
tuner.search_space_summary()
tuner.search (x_train,
              y_train,
              batch_size=256,
              epochs=40,
              validation_split=0.1
              )
tuner. results_summary()
models=tuner.get_best_models (num_models=3)
for model in models:
  model. summary ()
  model.evaluate (x_test, y_test)
  print()
  
model =tuner.get_best_models(num_models=3)[0]
history = model.fit(x_train, y_train,
                    batch_size=200,
                   epochs=100,
                   validation_split=0.2,
                   verbose=1)
scores = model.evaluate(x_test, y_test, verbose=1)
print ("Доля верных ответов на тестовых данных, в процентах: ", round (scores[1] * 100, 4))
