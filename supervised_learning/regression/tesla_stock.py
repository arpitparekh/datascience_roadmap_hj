path= "/home/arpit-parekh/Downloads/archive(5) (2)/TSLA.csv"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(path)

"""
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   Date       2956 non-null   object  // remove this
 1   Open       2956 non-null   float64
 2   High       2956 non-null   float64
 3   Low        2956 non-null   float64
 4   Close      2956 non-null   float64
 5   Adj Close  2956 non-null   float64
 6   Volume     2956 non-null   int64   // convert this into float
"""
df['Volume'] = df['Volume'].astype(float)
print(df.info())

x = df.drop(['Date','Close'],axis=1)
y = df['Close']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# plot the data
plt.scatter(x_test['Low'],y_test,color='blue')
plt.scatter(x_test['Low'],y_pred,color='red')
# plt.show()

def userInput():
    open = float(input("Enter Open "))
    high = float(input("Enter High "))
    low = float(input("Enter Low "))
    volume = float(input("Enter Volume "))
    adj_close = float(input("Enter Adj Close "))

    x = pd.DataFrame({
        'Open':[open],
        'High':[high],
        'Low':[low],
        'Adj Close':[adj_close],
        'Volume':[volume]
    })

    y_pred = model.predict(x)
    print("Predicted Close Price is",y_pred)



userInput()
