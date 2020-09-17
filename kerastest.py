import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Блокирует предупреждения что используется CPU а не GPU

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Обучающая выборка
c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4])

model = keras.Sequential() # Вызываем класс Sequential(). Последовательная нейронная сеть
model.add(Dense(units=1, input_shape=(1,), activation='linear')) #Dense - создает слой нейронов. units=1-ск-ко нейронов, input_shape=(1,)-ск-ко входов, activation='linear'-активационная функция
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1)) #loss - критерий качества, optimizer-оптимизатор

history = model.fit(c, f, epochs=100000, verbose=0)#fit- запуск обучения(c-вход.знач, f-вых.знач, epochs=500-количество итераций, verbose=0-выводить или нет инфу в процессе обучения)
print("Обучение завершено")

print(model.predict([100])) # predict-подаем на вход свои данные
print(model.get_weights()) #get_weights- отобразить весовые коэфиценты

plt.plot(history.history['loss']) #обращаемся к переменной history и выводим со словаря history значение loss
plt.grid(True)
plt.show()