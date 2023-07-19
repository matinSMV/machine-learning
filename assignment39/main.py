import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#data
data = pd.read_csv('dataset/weatherHistory.csv')
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)
data['Year'] = data['Formatted Date'].dt.year
data['DayOfYear'] = data['Formatted Date'].dt.dayofyear

avgDayTemperature = data.groupby(['DayOfYear', 'Year'])['Temperature (C)'].mean().reset_index()
avgDayTemperature


#class
class Perceptron:
    def __init__(self, epochs=1, lr=0.000001):
        # set hyperParameters
        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.lr = lr
        self.epochs = epochs

        self.W = []
        self.B = []
        self.loss = []
        self.R2 = []
    def fit(self, x_train, y_train):
        for epoch in range(self.epochs):
            for i in range(len(x_train)):    
                y_pred = np.matmul(x_train[i], self.w) + self.b
                e = y_train[i] - y_pred
                self.w = self.w + e * self.lr * x_train[i]
                self.b = self.b + e * self.lr
                
                Y_pred = np.matmul(x_train, self.w)
                error = np.mean(np.abs(y_train - Y_pred))
                
                self.loss.append(error)
                self.R2.append(r2_score(y_train, Y_pred))
                self.W.append(self.w)
                self.B.append(self.b)

        np.save('HP_WandB.npy', self.W + self.B)

    def predict(self, x):
        y_pred = np.matmul(x, self.w) + self.b
        return y_pred

    def evaluate(self, X, Y):
        y_pred = np.matmul(X, self.w) + self.b
        MSE = mean_squared_error(Y, y_pred)
        R2 = r2_score(Y, y_pred)
        return R2, MSE

    def get_loss(self):
        return self.loss

    def get_R2(self):
        return self.R2

X = avgDayTemperature['DayOfYear'].values.reshape(-1, 1)
Y = avgDayTemperature['Temperature (C)'].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=24)

perceptron = Perceptron(epochs=1, lr=0.000001)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X)
r2, mse = perceptron.evaluate(X_test, Y_test)
print(f'MSE: {mse}')

plt.figure(figsize=(12, 6))
plt.scatter(X, Y, marker='.', label='Data')
plt.plot(X, Y_pred, color='red', lw=4, label='Fitted line')
plt.xlabel('Day Of Year')
plt.ylabel('Temperature (C)')
plt.title('Average Day Temperature from 2006 to 2016')
plt.legend(loc='best')
plt.show()

#Loss
loss = perceptron.get_loss()
plt.plot(loss[20:])
plt.xlabel('Iteration #')
plt.ylabel('Error')
plt.title('Loss')
plt.show()

#iteration
r2 = perceptron.get_R2()
plt.plot(list(map(abs, r2[20:])), lw=0.7) 
plt.xlabel('Iteration #')
plt.ylabel('Score')
plt.title('R2 Score')
plt.show()