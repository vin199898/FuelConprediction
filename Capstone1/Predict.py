import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import pickle



st.header('Predict Vessel Carbon Emission')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)



Algo = st.selectbox(
    'Select Algorithm',
    ('xgBoost','RandomForest',' Linear Regression'))

Accuracy = st.button('Accuracy')
st.write("Input Parameters:")

#Accuracy

#1.xgboost
if Accuracy:
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

        st.markdown('Accuracy: '+str(r2) + "%")
        pickle.dump(xgb_r, open('./model1.sav', 'wb'))
        
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

        st.markdown('Accuracy: '+str(r2) + "%")
        
        pickle.dump(rfregressor, open('./model2.sav', 'wb'))
    
    
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

        st.markdown('Accuracy: '+str(r2) + "%")
        
        pickle.dump(lr, open('./model3.sav', 'wb'))
        
     
################################################################################Prediction

#########1.xgboost
if Algo == "xgBoost":
    model = pickle.load(open('model1.sav', 'rb'))
    def user_report():
        
        
      global speedLOG
    
       

      draft = st.number_input('Draft(m)', min_value = 2.0, max_value = 20.0, value = 2.0, step = 0.1)
      speedLOG = st.number_input('Speed(Knots)', min_value = 0.0, max_value = 30.0, value = 0.0, step = 0.1)
      rpm = st.number_input('Engine Speed(RPM)', min_value = 0.0, max_value = 150.0, value = 0.0, step = 0.1)
      Distance = speedLOG*24
      
      
      
      
      
      user_report_data = {
          'draft': draft,
          'speed':speedLOG,
          'RPM':rpm,
          'disLOG':Distance 
      }
      report_data = pd.DataFrame(user_report_data, index=[0])
      return report_data
     

    user_data = user_report()
    
    
    VoyageDistance = st.number_input('Voyage Distance', min_value = 0.0, max_value = 20000.0, value = 0.0, step = 0.1)
    
    
    
    prediction = st.button('Prediction')
    
    
    
    if prediction:
        FC = np.round(model.predict(user_data),2)
        st.subheader("Predicted  Fuel Consumption: " + str(FC[0]) +' Tonnes Per day')
        carbon = np.round((FC*3.206),2)
        st.subheader("Predicted  C02 Emission:  " + str(carbon[0]) + ' Tonnes Per day')
        
        Time = (VoyageDistance/speedLOG)/24
        Total_FC = np.round((Time*FC),2)
        Total_carbon = np.round((Total_FC*3.206),2)
        
        
        st.subheader("Predicted Voyage Fuel Consumption: " + str(Total_FC[0]) +' Tonnes ')
        st.subheader("Predicted  Voyage C02 Emission:  " + str(Total_carbon[0]) + ' Tonnes ')
        
        
##############2.randomforest        
elif Algo == 'RandomForest':
    model2 = pickle.load(open('model2.sav', 'rb'))
    def user_report2():
        
        global speedLOG
        
        draft = st.number_input('Draft', min_value = 2.0, max_value = 20.0, value = 2.0, step = 0.1)
        speedLOG = st.number_input('Speed', min_value = 0.0, max_value = 30.0, value = 0.0, step = 0.1)
        rpm = st.number_input('RPM', min_value = 0.0, max_value = 150.0, value = 0.0, step = 0.1)
        Distance = speedLOG*24
        
        
        
        user_report_data2 = {
            'draft': draft,
            
            'speed':speedLOG,
            'RPM':rpm,
            'disLOG':Distance
          
        }
        report_data2 = pd.DataFrame(user_report_data2, index=[0])
        return report_data2
       
        
    user_data2 = user_report2()
    
    VoyageDistance = st.number_input('Voyage Distance', min_value = 0.0, max_value = 20000.0, value = 0.0, step = 0.1)
    
    prediction2 = st.button('Prediction')
    
    if prediction2:
       FC = np.round(model2.predict(user_data2),2)
       st.subheader("Predicted  Fuel Consumption: " + str(FC[0]) +' Tonnes Per day')
       carbon = np.round((FC*3.206),2)
       st.subheader("Predicted  C02 Emission:  " + str(carbon[0]) + ' Tonnes Per day')
       
       Time = (VoyageDistance/speedLOG)/24
       Total_FC = np.round((Time*FC),2)
       Total_carbon = np.round((Total_FC*3.206),2)
       
       
       st.subheader("Predicted Voyage Fuel Consumption: " + str(Total_FC[0]) +' Tonnes ')
       st.subheader("Predicted  Voyage C02 Emission:  " + str(Total_carbon[0]) + ' Tonnes ')
      
###############3.linear regression    
else:
    model3 = pickle.load(open('model3.sav', 'rb'))
    def user_report3():
        
        global speedLOG
        
        draft = st.number_input('Draft', min_value = 2.0, max_value = 20.0, value = 2.0, step = 0.1)
        speedLOG = st.number_input('Speed', min_value = 0.0, max_value = 30.0, value = 0.0, step = 0.1)
        rpm = st.number_input('RPM', min_value = 0.0, max_value = 150.0, value = 0.0, step = 0.1)
        Distance = speedLOG*24
        
        
        
        
        user_report_data3 = {
            'draft': draft,
            'speed':speedLOG,
            'RPM':rpm,
            'disLOG':Distance
          
        }
        report_data3 = pd.DataFrame(user_report_data3, index=[0])
        return report_data3
       
        
    user_data3 = user_report3()
    
    VoyageDistance = st.number_input('Voyage Distance', min_value = 0.0, max_value = 20000.0, value = 0.0, step = 0.1)
    
    prediction3 = st.button('Prediction')
    
    if prediction3:
        FC = np.round(model3.predict(user_data3),2)
        st.subheader("Predicted  Fuel Consumption: " + str(FC[0]) +' Tonnes Per day')
        carbon = np.round((FC*3.206),2)
        st.subheader("Predicted  C02 Emission:  " + str(carbon[0]) + ' Tonnes Per day')
        
        Time = (VoyageDistance/speedLOG)/24
        Total_FC = np.round((Time*FC),2)
        Total_carbon = np.round((Total_FC*3.206),2)
        
        
        st.subheader("Predicted Voyage Fuel Consumption: " + str(Total_FC[0]) +' Tonnes ')
        st.subheader("Predicted  Voyage C02 Emission:  " + str(Total_carbon[0]) + ' Tonnes ')
       
        
        
    
            
        
        
        
        
   
    
    
    

        
        
        
        

        

       

    

 
        
    
         
     

    

   
   
       
    
    

  






    

        
        
    


    



    
    

    

    
    



