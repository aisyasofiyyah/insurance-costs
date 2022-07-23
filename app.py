import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Medical Costs Prediction App</h1>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    'Select Variables',
     ['age','bmi','children','sweetviz'])
     
def read_data():
    return pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/insurance-costs/main/insurance.csv')[['age','bmi','children','charges']]
costs=read_data()

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
    st.plotly_chart(f)

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
    



