import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import  load_model
import streamlit as st

start='2000-01-01'
end=date.today()

st.title('STOCK TREND PREDICTION')
user_input = st.text_input('Enter Stock Code','SBIN.NS')
df=data.DataReader(user_input, 'yahoo', start, end)

#Describing_Data
st.subheader('Data')
st.write(df.describe())

#visualisation new
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# #visualisation
# st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# fig = plt.figure(figsize = (12,6))
# plt.plot(ma100)
# plt.plot(ma200)
# plt.plot(df.Close)
# st.pyplot(fig)


#Split data into training set and test set new

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1

data_training_array = scaler.fit_transform(data_training)



# #Split data into training set and test set
# dataset_train=df.iloc[0:int(0.8*len(df)),:]
# dataset_test=df.iloc[int(0.8*len(df)):,:]


# #Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
# sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1


# training_set_scaled=sc.fit_transform(dataset_train)

#load model
model = load_model('complete.h5')

#Testing

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

y_predicted[y_predicted.shape[0]-1][0]





# # testing
# real_stock_price=dataset_test.iloc[:,4:5].values

# X_test=[]
# for i in range(100,len(training_set_scaled)):
#     X_test.append(training_set_scaled[i-100:i,0])
#             #Convert list to numpy arrays
# X_test=np.array(X_test)
        

# X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        

# predicted_stock_price=model.predict(X_test)

# predicted_stock_price=sc.inverse_transform(predicted_stock_price)
# fig = plt.figure(figsize=(25,15),dpi=65)
# plt.plot(real_stock_price,label='Actual Price')  
# plt.plot(predicted_stock_price,label='Predicted Price')
          
# plt.legend(loc=4)
# plt.show()

