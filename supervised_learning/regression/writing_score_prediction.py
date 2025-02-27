dataset_path = "/home/arpit-parekh/Downloads/archive(3)/exams.csv"
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv(dataset_path)
print(df.head())
print(df.info())

"""
Data columns (total 8 columns):
#   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
0   gender                       1000 non-null   object
1   race/ethnicity               1000 non-null   object
2   parental level of education  1000 non-null   object
3   lunch                        1000 non-null   object
4   test preparation course      1000 non-null   object
5   math score                   1000 non-null   int64
6   reading score                1000 non-null   int64
7   writing score                1000 non-null   int64
"""

# convert all the string columns to numeric
# label encoding
encoder  = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['race/ethnicity'] = encoder.fit_transform(df['race/ethnicity'])
df['parental level of education'] = encoder.fit_transform(df['parental level of education'])
df['lunch'] = encoder.fit_transform(df['lunch'])
df['test preparation course'] = encoder.fit_transform(df['test preparation course'])

print(df.head())
print(df.info())

x = df.drop(['writing score'],axis=1)
y = df['writing score']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# plot the data
plt.scatter(x_test['reading score'],y_test,color='blue')
plt.scatter(x_test['reading score'],y_pred,color='red')
plt.show()
