import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

costs = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Medical Costs Prediction App</h1>", unsafe_allow_html=True)
st.markdown("""
        The data set contains information about medical costs depending on someone's age, bmi, number of children, is a smoker or not.
        
        Objectives of this app are:
        1) What is the relationship between age and medical costs?
        2) What is the relationship between bmi and medical costs?
        3) What variables impacts medical costs.
""")

option = st.sidebar.selectbox(
    'Select Variables',
     ['age','bmi','children','sweetviz'])

if option=='age':
    st.subheader("Correlation between Age and Medical costs")
    X2= pd.DataFrame(costs['age'])
    y2= costs.charges
    X_train, X_test, y_train, y_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train) 
    y2_pred = model.predict(X_test)
    
    st.table({
            'RMSE':[{(mean_squared_error(y_test, y2_pred))**0.5}],
            'Variance':[{r2_score(y_test, y2_pred)}]
             })
    
    f=go.figure(figsize=(15,10))
    px.scatter(X_test, y_test, color='blue', label='Age')
    px.plot(X_test, y2_pred, color='red', label='Predicted Medical Costs', linewidth=1)
    px.xlabel("Charges")
    px.ylabel("Age")
    px.title('Age vs Charges')    
    px.legend()
    st.plotly_chart(f)

elif option=='bmi':
  
  model= LinearRegression()
  X1= pd.DataFrame(costs['bmi'])
  y1= costs['charges']
  X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

  model.fit(X_train,y_train)
  y1_pred= model.predict(X_test)
  st.table({
            'RMSE':{(mean_squared_error(y_test, y1_pred))**0.5},
            'Variance':{r2_score(y_test, y1_pred)}
             })
  
  f1=plt.figure()
  plt.scatter(X_test, y_test, color='black')
  plt.plot(X_test, y1_pred, color='blue', linewidth=1)
  plt.xlabel("Charges")
  plt.ylabel("BMI")
  plt.title('BMI vs Charges')
    
  st.write(f1)
  #columns=['BMI','Charges'] 
  #fig = px.scatter(
  #      x=costs["charges"],
  #      y=costs["bmi"],
  #  )
  #fig.update_layout(
  #      xaxis_title="Charges",
  #      yaxis_title="BMI",
  #  )
  #st.write(fig)
  
#elif option=='children'
#  figure3=
#  columns=[

#else
#    sweetv
    



