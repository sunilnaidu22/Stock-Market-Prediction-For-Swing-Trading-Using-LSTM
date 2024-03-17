#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px


# In[2]:


df=pd.read_csv('C:/Users/91638/Downloads/final data.csv')


# In[3]:


columns_to_round = ['Open ', 'High ', 'Low ', 'Close ']
df[columns_to_round] = df[columns_to_round].round()
print(df)


# In[4]:




figure = go.Figure(data=[go.Candlestick(x=df["Date "],
                                        open=df["Open "], high=df["High "],
                                        low=df["Low "], close=df["Close "])])
figure.update_layout(title="Nifty Stock Price Analysis", xaxis_rangeslider_visible=False, height=400, width=1000)
figure.show()


# In[5]:


df = df.dropna()
df = df.reset_index(drop=True)


# In[6]:


print(df.isnull().sum())


# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df1=df.reset_index()['Close ']


# In[11]:


df1


# In[12]:


plt.plot(df1)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[14]:


print(df1)


# In[15]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[16]:


training_size,test_size


# In[17]:


train_data


# In[18]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# In[19]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[20]:


print(X_train.shape)
print(X_test.shape)


# In[21]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[22]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[23]:


model.summary()


# In[24]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=10,verbose=1)


# In[25]:


import tensorflow as tf


# In[26]:


tf.__version__


# In[27]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[28]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[29]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[30]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

# Create a figure with a specific size
plt.figure(figsize=(12, 6))

# Shift train predictions for plotting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

# Show the plot
plt.show()


# In[32]:


len(test_data)


# In[33]:


x_input=test_data[340:].reshape(1,-1)
x_input.shape


# In[34]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[35]:


temp_input


# In[36]:


print("Input shape:", x_input.shape)


# In[37]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=281
i=0
while(i<50):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input =x_input.reshape(1,-1)
        
        #print(x_input)
        yhat = model.predict(x_input, verbose=1)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    


# In[38]:


print(lst_output)


# In[39]:


day_new=np.arange(1,101)
day_pred=np.arange(101,151)


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


len(df1)


# In[42]:


plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(df1[-100:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))


# In[43]:


plt.figure(figsize=(12, 6))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[120:])


# In[44]:


import plotly.graph_objects as go

# Assuming 'df' is your DataFrame with columns "Date", "Open", "High", "Low", "Close"
# Calculate the moving averages
df['SMA50'] = df['Close '].rolling(window=50).mean()

# Create a Plotly figure with both the candlestick chart and the moving averages
figure = go.Figure()

# Add the candlestick chart
figure.add_trace(go.Candlestick(x=df["Date "],open=df["Open "],high=df["High "],low=df["Low "],close=df["Close "],name="Candlesticks"))

# Add the 50-period and 200-period moving averages as line graphs
figure.add_trace(go.Scatter(x=df["Date "], y=df['SMA50'], mode='lines', line=dict(color='blue'), name='SMA50'))


# Update layout
figure.update_layout(
    title="Stock Price Analysis with Candlestick Chart and Moving Averages",
    xaxis_rangeslider_visible=False
)

# Show the figure
figure.show()



# In[45]:


import plotly.graph_objects as go

# Assuming 'df' is your DataFrame with columns "Date", "Open", "High", "Low", "Close"
# Calculate the moving averages
df['SMA200'] = df['Close '].rolling(window=200).mean()


# Create a Plotly figure with both the candlestick chart and the moving averages
figure = go.Figure()

# Add the candlestick chart
figure.add_trace(go.Candlestick(x=df["Date "],open=df["Open "],high=df["High "],low=df["Low "],close=df["Close "],name="Candlesticks"))

#  200-period moving averages 
figure.add_trace(go.Scatter(x=df["Date "], y=df['SMA200'], mode='lines', line=dict(color='blue'), name='SMA200'))


# Update layout
figure.update_layout(
    title="Stock Price Analysis with Candlestick Chart and Moving Averages",
    xaxis_rangeslider_visible=False
)

# Show the figure
figure.show()


# In[ ]:




