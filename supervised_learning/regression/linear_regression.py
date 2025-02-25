# linear regression
# linearly prediction
# dependent variable and independent variable
# y = mx + c

# dependent variable - y and x
# m - slope
# c - constant intercept
# independent variable - m and c

# loss and error
# i want to draw a link betwene those datapoints to minimalize the error(loss)

# mse # mean squared error
# virtual environment
# pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error

np.random.seed(23)

# label data
x = np.array([5, 15, 25, 35, 45, 55,23,32])  # age
y = np.array([10000, 12000, 15000, 20000, 30000, 40000,16000,18000])  # salary

# 5 ,     15 ,   25   , 23 ,   32     # to chnage the m and c
# 10000, 12000, 15000, 16000, 18000

# plt.scatter(x, y)
# plt.show()

# model train  # train 80% / testing => 20 %
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(y_test)

model = LinearRegression()
model.fit(x_train.reshape(-1,1), y_train)
y_pred = model.predict(x.reshape(-1,1))

mse = mean_squared_error(y,y_pred)
rmse = root_mean_squared_error(y,y_pred)
print(mse)
print(rmse)

# plot the data
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.show()
