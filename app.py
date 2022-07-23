import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

costs = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Medical Costs Prediction App</h1>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    'Select Variables',
     ['age','bmi','children','sweetviz'])

if option=='age':
    st.header("Modeling")
    X2= pd.DataFrame(costs['age'])
    y2= costs.charges
    X_train, X_test, y_train, y_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train) 
    y2_pred = model.predict(X_test)
       
    st.markdown(f"""Linear Regerssion model trained:
            -RMSE:{(mean_squared_error(y_test, y2_pred))**0.5}
            -Variance, r^2:{r2_score(y_test, y2_pred):.4f}
    """)
    
    f=plt.figure()
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y2_pred, color='blue', linewidth=1)
    plt.xlabel("Charges")
    plt.ylabel("Age")
    plt.title('Age vs Charges')    
    st.write(f)

elif option=='bmi':
  
  model= LinearRegression()
  X1= pd.DataFrame(costs['bmi'])
  y1= costs['charges']
  X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

  model.fit(X_train,y_train)
  y1_pred= model.predict(X_test)
  st.markdown(f"""Linear Regression model trained:
            -RMSE:{(mean_squared_error(y_test, y1_pred))**0.5}
            -Variance, r^2:{r2_score(y_test, y1_pred):.4f}
    """)
  
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
    



