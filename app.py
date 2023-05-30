import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from plotly.subplots import make_subplots
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.tools as tls
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,LSTM
import datetime as dt 


st.title("Stock Dashboard")
ticker=st.sidebar.text_input("Ticker","AAPL",key="tname")
start_date=st.sidebar.date_input("Start Date",date(2022,3,31),key="sdate")
end_date=st.sidebar.date_input("End Date",key="edate")
data=yf.download(ticker,start=start_date,end=end_date)
ticker1=""
Line_chart,Cnadlestick=st.tabs(["Line Chart","Candlestick Chart"])
with Line_chart:
  fig=px.line(data,x=data.index,y=data['Adj Close'],title=ticker)
  st.plotly_chart(fig)
with Cnadlestick:
  df=data
# removing all empty dates
# build complete timeline from start date to end date
  dt_all = pd.date_range(start=df.index[0],end=df.index[-1])
  # retrieve the dates that ARE in the original datset
  dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
  # define dates with missing values
  dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
  fig.update_layout(xaxis_rangebreaks=[dict(values=dt_breaks)])
  # add moving averages to df
  df['MA20'] = df['Close'].rolling(window=20).mean()
  df['MA5'] = df['Close'].rolling(window=5).mean()
  # first declare an empty figure
  fig = go.Figure()
  # add OHLC trace
  macd = MACD(close=df['Close'], 
              window_slow=26,
              window_fast=12, 
              window_sign=9)
  # stochastics
  stoch = StochasticOscillator(high=df['High'],
                              close=df['Close'],
                              low=df['Low'],
                              window=14, 
                              smooth_window=3)

  fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                      vertical_spacing=0.01, 
                      row_heights=[0.5,0.1,0.2,0.2])
  # Plot OHLC on 1st subplot (using the codes from before)
  fig.add_trace(go.Candlestick(x=df.index,
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'], 
                              showlegend=False))
  # add moving average traces
  fig.add_trace(go.Scatter(x=df.index, 
                          y=df['MA5'], 
                          line=dict(color='blue', width=2), 
                          name='MA 5'))
  fig.add_trace(go.Scatter(x=df.index, 
                          y=df['MA20'], 
                          line=dict(color='orange', width=2), 
                          name='MA 20'))

  # Plot volume trace on 2nd row 
  colors = ['green' if row['Open'] - row['Close'] >= 0 
            else 'red' for index, row in df.iterrows()]
  fig.add_trace(go.Bar(x=df.index, 
                      y=df['Volume'],
                      marker_color=colors
                      ), row=2, col=1)

  # Plot MACD trace on 3rd row
  colors = ['green' if val >= 0 
            else 'red' for val in macd.macd_diff()]
  fig.add_trace(go.Bar(x=df.index, 
                      y=macd.macd_diff(),
                      marker_color=colors
                      ), row=3, col=1)
  fig.add_trace(go.Scatter(x=df.index,
                          y=macd.macd(),
                          line=dict(color='red', width=2)
                          ), row=3, col=1)
  fig.add_trace(go.Scatter(x=df.index,
                          y=macd.macd_signal(),
                          line=dict(color='blue', width=1)
                          ), row=3, col=1)

  # Plot stochastics trace on 4th row 
  fig.add_trace(go.Scatter(x=df.index,
                          y=stoch.stoch(),
                          line=dict(color='orange', width=2)
                          ), row=4, col=1)
  fig.add_trace(go.Scatter(x=df.index,
                          y=stoch.stoch_signal(),
                          line=dict(color='blue', width=1)
                          ), row=4, col=1)

  # update layout by changing the plot size, hiding legends & rangeslider, and removing gaps between dates
  fig.update_layout(height=900, 
                    showlegend=False, 
                    xaxis_rangeslider_visible=False,
                    xaxis_rangebreaks=[dict(values=dt_breaks)])
  fig.update_layout(title=ticker)

  # update y-axis label
  fig.update_yaxes(title_text="Price", row=1, col=1)
  fig.update_yaxes(title_text="Volume", row=2, col=1)
  fig.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
  fig.update_yaxes(title_text="Stoch", row=4, col=1)
  st.plotly_chart(fig)
  
  #############################################################################################################33
  
pricing_data,fundamental_data,news,tech_indicator,price_pred=st.tabs(["Pricing Data","Fundamental Data","Top 10 News","Technical Analysis Dashboard","Price Prediction"])

with pricing_data:
  st.header("Price Movement")
  data2=data
  data2["% Change"]=data['Adj Close']/data['Adj Close'].shift(1) -1
  data2.dropna(inplace = True)
  selected_columns=['Open','High','Low','Close','Volume']
  
  button_clicked1 = st.button("Price Movement Table", key='checkbox1')
  if 'checkbox1_state' not in st.session_state:
    st.session_state.checkbox1_state = False
  # Check if the button is clicked
  if button_clicked1:
      st.session_state.checkbox1_state = not st.session_state.checkbox1_state
  # Display the dataframe table based on checkbox state
  if st.session_state.checkbox1_state:
      st.write(data2[selected_columns])
      
  annual_return=data2['% Change'].mean()*252*100
  stdev=np.std(data2['% Change'])*np.sqrt(252)
  st.write("Annual Return is ",annual_return,'%')
  st.write("Standard Deviation is ",stdev*100,'%')
  st.write("Risk Adj. Return is",annual_return/(stdev*100))
  
  ################################################################################################
  
from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
  st.header("Fundamental Details")
  key='ZMVF7QF0GIQVB691'
  fd=FundamentalData(key,output_format='pandas')
  
  option = st.radio("Select a Fundamental Detail option", ("Price Data Table","Balance Sheet Analysis", "Income Statement Analysis", "Cash Flow Analysis"))
  
  
  if(option=="Price Data Table"):
    st.subheader("Price Movement")
    st.write(data2)
  if(option=="Balance Sheet Analysis"):
   st.subheader('Balance Sheet')
   balance_sheet=fd.get_balance_sheet_annual(ticker)[0]
   bs=balance_sheet.T[2:]
   bs.columns=list(balance_sheet.T.iloc[0])
   st.write(bs)
  if(option=="Income Statement Analysis"):
   st.subheader('Income Statement')
   income_statement=fd.get_income_statement_annual(ticker)[0]
   is1=income_statement.T[2:]
   is1.columns=list(income_statement.T.iloc[0])
   st.write(is1)
  if(option=="Cash Flow Analysis"):
   st.subheader('Cash Flow Statement')
   cash_flow=fd.get_cash_flow_annual(ticker)[0]
   cf=cash_flow.T[2:]
   cf.columns=list(cash_flow.T.iloc[0])
   st.write(cf)
   ###########################################################################################
from stocknews import StockNews
with news:
      st.header(f'News of {ticker}')
      sn=StockNews(ticker, save_news=False)
      df_news=sn.read_rss()
      for i in range(0,10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment=df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
#############################################################################################
with tech_indicator:
  st.subheader("Technical Analysis Dashboard")
  df=pd.DataFrame()
  ind_list=df.ta.indicators(as_list=True)
  technical_indicator=st.selectbox('Tech Indicator',options=ind_list)
  method=technical_indicator
  indicator=pd.DataFrame(getattr(df.ta,method)(time=data.index,low=data['Low'],close=data['Close'],high=data['High'],open=data['Open'],volume=data['Volume']))
  indicator['Close']=data['Close']
  figW_ind_new=px.line(indicator)
  st.plotly_chart(figW_ind_new)
  st.write(indicator)
###############################################################################################
with price_pred:
  st.subheader("Price Prediction")
  prediction_days=st.number_input("Enter the Days",60)
  
  sdate=dt.datetime(2022,1,1)
  data3=yf.download(ticker,start=sdate,end=start_date)
  scaler=MinMaxScaler(feature_range=(0,1))
  scaled_data=scaler.fit_transform(data3['Close'].values.reshape(-1,1))
    
  x_train=[]
  y_train=[]
    
  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

  x_train,y_train=np.array(x_train),np.array(y_train)
  x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    #Building Model
  model = Sequential()
    
  model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units=1)) #Prediction of the next price
    
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train,y_train, epochs=25,batch_size=32)
    
    ######Test The Model Accuracy on Existing Data######
  edate=dt.datetime.now()
  test_data=yf.download(ticker,start=start_date,end=edate)
    
  actual_prices = test_data['Close'].values
    
  total_dataset=pd.concat((data3['Close'],test_data['Close']), axis=0)
    
  model_inputs=total_dataset[len(total_dataset)- len(test_data) - prediction_days:].values
  model_inputs=model_inputs.reshape(-1,1)
  model_inputs=scaler.transform(model_inputs)
    
    #####Make Predictions on Test Data########
    
  x_test=[]
    
  for x in range(prediction_days, len(model_inputs)):
   x_test.append(model_inputs[x-prediction_days:x, 0])
      
  x_test=np.array(x_test)
  x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
  predicted_prices=model.predict(x_test)
  predicted_prices=scaler.inverse_transform(predicted_prices)
  
    ### Plot The Predictions ###
  plt.plot(actual_prices, color="blue", label=f"Actual {ticker} Price")
  plt.plot(predicted_prices, color="Green",label=f"Predicted {ticker} Price")

  fig = tls.mpl_to_plotly(plt.gcf())
  fig.update_layout(title=f"{ticker} Share Price", xaxis_title='Time', yaxis_title='Price',showlegend=True,width=800)

  st.plotly_chart(fig)

  ### Predict Next Day ###########

  real_data=[model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
  real_data=np.array(real_data)
  real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

  prediction=model.predict(real_data)
  prediction=scaler.inverse_transform(prediction)
  formatted_number = f"{prediction[0][0]:.2f}"

  st.write("Prediction is:  ",formatted_number)
    
    
