import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import os
# import datetime
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler

st.header('stock market prediction')

Time_windows_for_training=st.slider("select time window for training",0,200)


stock_name=st.text_input("Enter the Stock Symbol")
start_date=st.date_input("Enter the start date for the model to")
end_date=st.date_input("Enter the end date of the model to train")

def prepare_data(data):
    # st.write(int(len(data)))

    # st.write(int(len(data)*0.80))
    # st.write(int(len(data)*0.80))
    data_train=pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
    test_data=pd.DataFrame(data['Close'][int(len(data)*0.80):])
    # st.write(data_train.shape)
    # st.write(test_data.shape)
    scaler=MinMaxScaler(feature_range=(0,1))
    data_train_scaler=scaler.fit_transform(data_train)


    x_train,y_train=[],[]
    for i in range(100,len(data_train_scaler)):
        x_train.append(data_train_scaler[i-100:i])
        y_train.append(data_train_scaler[i])
    x_train,y_train=np.array(x_train),np.array(y_train)

    return x_train,y_train,test_data,data_train


def training_model(x_train, y_train):
  
    st.info("Training the model...")
    model = Sequential()
 
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(units=80, activation='relu'))  # Removed return_sequences=True
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))  # Output layer
    st.info("Model architecture created.") # Output layer
 


    model.compile(optimizer='adam',loss='mean_squared_error')

    st.info("model is training...")
    progress_bar=st.progress(0)
    for epoch in range(1,51):
        model.fit(x_train,y_train,epochs=1,batch_size=32,verbose=0)
        progress_bar.progress(epoch/50)

    model.save(model_file)
    st.success(f"model trained and saved as {model_file}")
    return model    

    

    
    

try:
    data=yf.download(stock_name,start=start_date,end=end_date)
    if data.empty:
        st.error("data is not downloaded.please try it again")
    else:

        st.subheader("stock data")
        st.write(data)
        x_train,y_train,test_data,data_train=prepare_data(data)
        # print(test_data,data_train)

        model_file=f"model/{stock_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.h5"
        st.write(model_file)
     

        if os.path.exists(model_file):
            st.info("model already trained for this stock and date range.loading the model...")
            model=load_model(model_file)
        else:
            model=training_model(x_train,y_train)
        # model=training_model(x_train,y_train)
        

        scaler=MinMaxScaler(feature_range=(0,1))


        past_100_days=data_train.tail(100)
        final_test_data=pd.concat([past_100_days,test_data],ignore_index=False)
        final_test_scaled=scaler.fit_transform(final_test_data)
        # st.write(final_test_scaled)


        x_test,y_test=[],[]
        for i in range(100,len(final_test_scaled)):
            x_test.append(final_test_scaled[i-100:i])
            y_test.append(final_test_scaled[i])
        x_test,y_test=np.array(x_test),np.array(y_test)

        predictions=model.predict(x_test)
 
        predictions=scaler.inverse_transform(predictions)
        st.subheader("predicted Values ") 
        st.write(pd.DataFrame(predictions,columns=['predicted price']))
        # st.write(predictions)

        st.subheader('original price vs predicted price')
        fig1=plt.figure(figsize=(10,6))
        plt.plot(test_data.values,label='original price',color='blue')
        plt.plot(predictions,label='predicted price',color='red')
        plt.xlabel('time date')
        plt.ylabel('price')
        plt.legend()
        st.pyplot(fig1)

        st.subheader('price vs MA50')
        ma_50=data['Close'].rolling(50).mean()
        fig2=plt.figure(figsize=(10,6))
        plt.plot(data['Close'],label=' price',color='blue')
        plt.plot(ma_50,label='MA50',color='red')
        # plt.plot(ma_100,labe4l='MA50',color='green')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        st.pyplot(fig2)


        st.subheader('price vs MA50 vs MA100')
        ma_100=data['Close'].rolling(100).mean()
        fig3=plt.figure(figsize=(10,6))
        plt.plot(data['Close'],label=' price',color='blue')
        plt.plot(ma_50,label='MA50',color='red')
        plt.plot(ma_100,label='MA100',color='green')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        st.pyplot(fig3)


        st.subheader('price vs MA100 vs MA200')
        ma_200=data['Close'].rolling(200).mean()
        fig4=plt.figure(figsize=(10,6))
        plt.plot(data['Close'],label='price',color='blue')
        plt.plot(ma_100,label='MA100',color='green')
        plt.plot(ma_200,label='MA200',color='orange')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        st.pyplot(fig4)

except:
    st.error("invaild entry is made .please check the response again")



