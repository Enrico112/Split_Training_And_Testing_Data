import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('carprices.csv')
plt.figure()
plt.scatter(df['Mileage'], df['Sell Price($)'])
plt.figure()
plt.scatter(df['Age(yrs)'], df['Sell Price($)'])

# to call mileage x sell price figure
# plt.figure(1)

X = df[['Mileage', 'Age(yrs)']].values
y = df[['Sell Price($)']].values

# split the df into training and testing dfs
# test size selects ratio of test size / total size
# selection is random, i.e. it does not follow index
# to select the same random sub dfs specify the same random_state
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
# check lengths
len(X_train)
len(X_test)
len(y_train)
len(y_test)

# create lin reg obj
clf = LinearRegression()

# TRAINING
# fit model on training df
clf.fit(X_train, y_train)

# TESTING
# predict X_test using model
clf.predict(X_test)

# compares predicted values to actual values and computes accuracy
clf.score(X_test, y_test)
