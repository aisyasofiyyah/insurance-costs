import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')

X1= pd.DataFrame(df['bmi'])
y1= df['charges']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

model= LinearRegression()
model.fit(X_train,y_train)
y1_pred= model.predict(X_test)

print("Root mean squared error: {} ".format(mean_squared_error(y_test, y_pred)**0.5))
print('Variance score: {} '.format(r2_score(y_test,y_pred)))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y1_pred, color='blue', linewidth=1)
plt.xlabel("BMI")
plt.ylabel("Charges")

plt.title('Charges vs BMI')
plt.show()

#finding correlation between age and costs
X2= pd.DataFrame(df['age'])
y2= df['charges']
X_train,X_test,y_train,y_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

model = LinearRegression(fit_intercept=True)

model.fit(X_train, y_train) 
y2_pred = model.predict(X_test)

print(f"RMSE: {(mean_squared_error(y_test, y2_pred))**0.5}.")
print(f"R^2: {r2_score(y_test, y2_pred):.4f}")

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y2_pred, color='blue', linewidth=1)
plt.xlabel("Age")
plt.ylabel("Charges")

plt.title('Charges vs Age')
plt.show()
