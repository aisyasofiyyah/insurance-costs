import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Medical Costs Prediction App</h1>", unsafe_allow_html=True)

costs = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')

option = st.sidebar.selectbox(
    'Select Variables',
     ['age','bmi','children','sweetviz'])

if option=='age':    
    X2= pd.DataFrame(costs['age'])
    y2= costs['charges']
    X_train,X_test,y_train,y_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train) 
    y2_pred = model.predict(X_test)
    st.write(f"RMSE: {(mean_squared_error(y_test, y2_pred))**0.5}.")
    st.write(f"R^2: {r2_score(y_test, y2_pred):.4f}")
 
    fig = px.scatter(
        x=costs["charges"],
        y=costs["age"],
    )
    fig.update_layout(
        xaxis_title="Charges",
        yaxis_title="Age",
    )
    st.write(fig)
    
    chart_data = pd.DataFrame(X2,y2,y2_predict)
    columns=['charges', 'age', 'y2_predict'])

    st.line_chart(chart_data)

elif option=='bmi':
  
  model= LinearRegression()
  X1= pd.DataFrame(costs['bmi'])
  y1= costs['charges']
  X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

  model.fit(X_train,y_train)
  y1_pred= model.predict(X_test)
  st.write("Root mean squared error: {} ".format(mean_squared_error(y_test, y1_pred)**0.5))
  st.write('Variance score: {} '.format(r2_score(y_test,y1_pred)))
  
  #columns=['BMI','Charges'] 
  fig = px.scatter(
        x=costs["charges"],
        y=costs["bmi"],
    )
  fig.update_layout(
        xaxis_title="Charges",
        yaxis_title="BMI",
    )
  st.write(fig)
  
#elif option=='children'
#  figure3=
#  columns=[

#else
#    sweetv
    



