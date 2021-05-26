import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

filename = "dunedata.csv"
df = pd.read_csv(filename)

#define x and y
x=df.drop(['punt1', 'punt2', 'punt3', 'jaar'], axis=1).values
y=df['punt1'].values

#split the dataset in training & test set
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.15, random_state=42)


#train the model on the training set
model=LinearRegression()
model.fit(x_train, y_train)

#predict the test set results
y_pred=model.predict(x_test)
y_pred2=model.predict(x_train)

#evaluate the model
print("R2 score training: ", r2_score(y_train, y_pred2))
print("R2 score test: ", r2_score(y_test, y_pred))

#plot results
plt.figure(figsize=(10,5))

joinedPredicted = np.append(y_pred2, y_pred)
joinedActual = np.append(y_train, y_test)

plt.plot(range(0,len(y_train) + len(y_test)), joinedPredicted, label="Predicted values (joined)")
plt.plot(range(0,len(y_train) + len(y_test)), joinedActual, label="Actual values (joined)")
plt.plot(range(0,len(y_train)), y_pred2, label="Predicted train values")
plt.plot(range(0,len(y_train)), y_train, label="Actual train values")  
plt.plot(range(len(y_train),len(y_train) + len(y_test)), y_pred, label="Predicted test values")
plt.plot(range(len(y_train),len(y_train) + len(y_test)), y_test, label="Actual test values")
plt.xlabel('Year')
plt.ylabel('Duneheight')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()

pred_y_df=pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred, 'Difference': y_test-y_pred, 'Error Rate': (y_pred-y_test)/y_test * 100})
print(pred_y_df)
