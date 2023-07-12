import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import plotly.express as px



st.header('Model Analysis')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)


Algo = st.selectbox(
    'Select Algorithm',
    ('xgBoost','RandomForest',' Linear Regression'))
    
Explore = st.button('Explore')

#1.xgboost
if Explore:
    if Algo == 'xgBoost':
        TargetVariable='mainEngFO'

        #Predictors=['Draft','RelDirection','WindForce','disLOG','speedLOG']
        Predictors=['Draft','speedLOG','rpm','disLOG']
        X= df[Predictors].values
        y= df[TargetVariable].values

        train_X, test_X, train_y, test_y = train_test_split(X, y,
                              test_size = 0.25, random_state = 222)

        xgb_r = xgb.XGBRegressor(objective ='reg:linear',
                          n_estimators = 2000, seed = 120)


        xgb_r.fit(train_X, train_y)

        pred = xgb_r.predict(test_X)

        rmse = np.sqrt(MSE(test_y, pred))
        print("RMSE : % f" %(rmse))
        r2 = np.round(xgb_r.score(test_X,test_y),2)
        r2 = r2*100
        print(r2)
        
        mse = np.round(MSE(test_y,pred),2)
        
        mae = np.round(MAE(test_y,pred),2)
        
        
        st.subheader('Model Evaluation')
        st.write("R Squared : " + str(r2) + '%')
        st.write('Mean Square Error: ' + str(mse))
        st.write('Mean Absolute Error: ' + str(mae))
        
        
       
        #plots
        
        df1 = pd.DataFrame({"Predicted FC":pred,        
                     "Actual FC":test_y,
                     })
        
        fig1 = px.line(df1,title="Predicted Fuel Con vs Actual Fuel Con ")
        fig1.update_layout(yaxis_title='Fuel Con')
        fig1.update_layout(xaxis_title= 'Count')    
        st.plotly_chart(fig1)
        
        
        fig2 = px.scatter(df, x= test_y, y= pred,  trendline="ols", trendline_color_override="red")
        fig2.update_layout(yaxis_title='Predicted FC')
        fig2.update_layout(xaxis_title= 'Actual FC')
        
        st.plotly_chart(fig2)
      
        
    
        
#2.randomforest        
    elif Algo == 'RandomForest':
        
        TargetVariable='mainEngFO'
        #Predictors=['Draft','RelDirection','WindForce','disLOG','speedLOG']
        Predictors=['Draft','speedLOG','rpm','disLOG']
        X= df[Predictors].values
        y= df[TargetVariable].values

        train_X, test_X, train_y, test_y = train_test_split(X, y,
                              test_size = 0.25, random_state = 222)
        
        rfregressor = RandomForestRegressor(n_estimators= 100, random_state=(0))
        rfregressor.fit(train_X, train_y)
        
        pred = rfregressor.predict(test_X)
        r2 = np.round(rfregressor.score(test_X,test_y),2)
        r2 = r2*100
        
        mse = np.round(MSE(test_y,pred),2)
        
        mae = np.round(MAE(test_y,pred),2)
        
        
        st.subheader('Model Evaluation')
        st.write("R Squared : " + str(r2) + '%')
        st.write('Mean Square Error: ' + str(mse))
        st.write('Mean Absolute Error: ' + str(mae))
        


   
       
        df1 = pd.DataFrame({"Predicted FC":pred,        
                     "Actual FC":test_y,
                     })
        
        fig1 = px.line(df1,title="Predicted Fuel Con vs Actual Fuel Con")
        fig1.update_layout(yaxis_title=None)
        fig1.update_layout(xaxis_title=None)
        st.plotly_chart(fig1)
        
        fig2 = px.scatter(df, x= test_y, y= pred,  trendline="ols", trendline_color_override="red")
        fig2.update_layout(yaxis_title='Predicted FC')
        fig2.update_layout(xaxis_title= 'Actual FC')
        st.plotly_chart(fig2)
        
        
        
    
    
#3. linear regression   
    else:
        TargetVariable='mainEngFO'
        #Predictors=['Draft','RelDirection','WindForce','disLOG','speedLOG']
        Predictors=['Draft','speedLOG','rpm','disLOG']

        X= df[Predictors].values
        y= df[TargetVariable].values
        
        
        train_X, test_X, train_y, test_y = train_test_split(X, y,
                              test_size = 0.25, random_state = 555)
        
        lr = LinearRegression()
        lr.fit(train_X, train_y)
        pred = lr.predict(test_X)
        r2 = np.round(lr.score(test_X,test_y),2)
        r2 = r2*100

        mse = np.round(MSE(test_y,pred),2)
         
        mae = np.round(MAE(test_y,pred),2)
         
         
        st.subheader('Model Evaluation')
        st.write("R Squared : " + str(r2) + '%')
        st.write('Mean Square Error: ' + str(mse))
        st.write('Mean Absolute Error: ' + str(mae))
         
        
        df1 = pd.DataFrame({"Predicted FC":pred,        
                     "Actual FC":test_y,
                     })
        
        fig1 = px.line(df1,title="Predicted Fuel Con vs Actual Fuel Con")
        fig1.update_layout(yaxis_title=None)
        fig1.update_layout(xaxis_title=None)
        st.plotly_chart(fig1)
        
        fig2 = px.scatter(df, x= test_y, y= pred,  trendline="ols", trendline_color_override="red")
        fig2.update_layout(yaxis_title='Predicted FC')
        fig2.update_layout(xaxis_title= 'Actual FC')
        st.plotly_chart(fig2)
        
       
        
                
        
    


    



    
    

    

    
    



