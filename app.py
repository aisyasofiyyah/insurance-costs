import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

costs = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Medical Costs Prediction App</h1>", unsafe_allow_html=True)

st.sidebar.markdown("""The data set contains information about medical costs depending on someone's age, bmi, number of children, is a smoker or not.
        \nObjectives of this app are:
        \n1) What is the relationship between age and medical costs?
        \n2) What is the relationship between bmi and medical costs?
        \n3) What variables impacts medical costs.
  """)
        
option = st.sidebar.selectbox(
    'Select Variables',
     ['Age','BMI','No. of Children','sweetviz'])

if option=='Age':
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

    f=px.scatter(data_frame=costs, x="age",y="charges", trendline="ols")
    #plt.scatter(X_test, y_test, color='blue', label='Age')
    #plt.plot(X_test, y2_pred, color='red', label='Predicted Medical Costs', linewidth=1)
    #plt.xlabel("Charges")
    #plt.ylabel("Age")
    #plt.title('Age vs Charges')    
    #plt.legend()
    st.write(f)

elif option=='BMI':
  
    model= LinearRegression()
    X1= pd.DataFrame(costs['bmi'])
    y1= costs['charges']
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

    model.fit(X_train,y_train)
    y1_pred= model.predict(X_test)
    st.table({
            'RMSE':[{(mean_squared_error(y_test, y1_pred))**0.5}],
            'Variance':[{r2_score(y_test, y1_pred)}]
             })
  
    f1=plt.figure(figsize=(15,10))
    plt.scatter(X_test, y_test, color='blue', label='BMI')
    plt.plot(X_test, y1_pred, color='red', label='Predicted Medical Costs', linewidth=1)
    plt.xlabel("Charges")
    plt.ylabel("BMI")
    plt.legend
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
  
elif option=='No. of Children':
  
  st.write("Under construction")
  model= LinearRegression()
  X3= pd.DataFrame(costs['children'])
  y3= costs['charges']
  X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.25, random_state = 0)

  model.fit(X_train,y_train)
  y3_pred= model.predict(X_test)
  st.table({
            'RMSE':[{(mean_squared_error(y_test, y3_pred))**0.5}],
            'Variance':[{r2_score(y_test, y3_pred)}]
             })
  
  f3=plt.figure(figsize=(15,10))
  plt.scatter(X_test, y_test, color='blue', label='No. of Children')
  plt.plot(X_test, y3_pred, color='red', label='Predicted Medical Costs', linewidth=1)
  plt.xlabel("Charges")
  plt.ylabel("No. of Children")
  plt.legend
  plt.title('No. of Children vs Charges')
    
  st.write(f3)

#else
#    sweetv
    



