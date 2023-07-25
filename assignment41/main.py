import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

#data
data = pd.read_csv('dataset/weatherHistory.csv')
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)
data['Year'] = data['Formatted Date'].dt.year
data['DayOfYear'] = data['Formatted Date'].dt.dayofyear

avgDayTemperature = data.groupby(['DayOfYear', 'Year'])['Temperature (C)'].mean().reset_index()
avgDayTemperature


#normalizing
avgDayTemperature['Year'] = avgDayTemperature['Year'] - 2006

scaler = StandardScaler()
scaler.fit(avgDayTemperature['Temperature (C)'].values.reshape(-1,1))
avgDayTemperature['Temperature (C)']=scaler.transform(avgDayTemperature['Temperature (C)'].values.reshape(-1,1))

#train
X = avgDayTemperature[['DayOfYear', 'Year']].values
Y = avgDayTemperature[['Temperature (C)']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=24)

#model
model = tf.keras.models.Sequential([
    Dense(128, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear') 
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
loss=tf.keras.losses.MeanSquaredError()
)

model.fit(X_train, Y_train, epochs=500)

#evaluate
model.evaluate(X_test, Y_test)
     

plt.figure(figsize=(12, 6))
plt.scatter(avgDayTemperature['DayOfYear'], avgDayTemperature['Temperature (C)'], marker='.')
plt.xlabel('Day Of Year')
plt.ylabel('Temperature (C)')
plt.title('Average Day Temperature from 2006 to 2016')
plt.show()
