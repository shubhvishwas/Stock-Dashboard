import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from plotly.subplots import make_subplots


st.title("Stock Dashboard")
ticker=st.sidebar.text_input("Ticker")
start_date=st.sidebar.date_input("Start Date")
end_date=st.sidebar.date_input("End Date")
ticker1="AAPL"
start_date1=date(2022,3,31)
end_date1=date(2023,4,1)
data=yf.download(ticker,start=start_date,end=end_date)
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
  
pricing_data,fundamental_data,news,tech_indicator=st.tabs(["Pricing Data","Fundamental Data","Top 10 News","Technical Analysis Dashboard"])

with pricing_data:
  st.header("Price Movement")
  data2=data
  data2["% Change"]=data['Adj Close']/data['Adj Close'].shift(1) -1
  data2.dropna(inplace = True)
  st.write(data2)
  annual_return=data2['% Change'].mean()*252*100
  stdev=np.std(data2['% Change'])*np.sqrt(252)
  st.write("Annual Return is ",annual_return,'%')
  st.write("Standard Deviation is ",stdev*100,'%')
  st.write("Risk Adj. Return is",annual_return/(stdev*100))
  
from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
  st.write("Fundamental")
  key='ZMVF7QF0GIQVB691'
  fd=FundamentalData(key,output_format='pandas')
  st.subheader('Balance Sheet')
  balance_sheet=fd.get_balance_sheet_annual(ticker)[0]
  bs=balance_sheet.T[2:]
  bs.columns=list(balance_sheet.T.iloc[0])
  st.write(bs)
  st.subheader('Income Statement')
  income_statement=fd.get_income_statement_annual(ticker)[0]
  is1=income_statement.T[2:]
  is1.columns=list(income_statement.T.iloc[0])
  st.write(is1)
  st.subheader('Cash Flow Statement')
  cash_flow=fd.get_cash_flow_annual(ticker)[0]
  cf=cash_flow.T[2:]
  cf.columns=list(cash_flow.T.iloc[0])
  st.write(cf)
from stocknews import StockNews
with news:
      st.header(f'News of {ticker}')
      sn=StockNews(ticker, save_news=False)
      df_news=sn.read_rss()
      for i in range(0,10):
        st.subheader(f'News{i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment=df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
        
# import openai
# import os
# openai.api_key = "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"
# model_engine = "text-davinci-002"
# temperature = 0.7
# max_tokens = 100

# def generate_response(prompt):
#     response = openai.api_request(
#     "completions/create",
#     data={
#         "engine": model_engine,
#         "prompt": prompt,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#     },
# )
#     return response.choices[0].text.strip()

# buy=f"3 Resaons to buy {ticker} stock"
# sell=f"3 Resaons to sell {ticker} stock"
# swot=f"SWOT analysis of {ticker} stock"

# with openai:
#   buy_reason,sell_reason,swot_analysis=st.tabs(['3 Resaons to buy','3 Resaons to sell','SWOT analysis'])
#   with buy_reason:
#     st.subheader(f'3 Resaons to BUY {ticker} stock')
#     st.write(generate_response(buy))
#   with sell_reason:
#     st.subheader(f'3 Resaons to SELL {ticker} stock')
#     st.write(generate_response(sell))
#   with swot_analysis:
#     st.subheader(f'SWOT analysis of {ticker} stock')
#     st.write(generate_response(swot))
import pandas_ta as ta
with tech_indicator:
  st.subheader("Technical Analysis Dashboard")
  df=pd.DataFrame()
  ind_list=df.ta.indicators(as_list=True)
  technical_indicator=st.selectbox('Tech Indicator',options=ind_list)
  method=technical_indicator
  indicator=pd.DataFrame(getattr(ta,method)(low=data['Low'],close=data['Close'],high=data['High'],open=data['Open'],volume=data['Volume']))
  indicator['Close']=data['Close']
  figW_ind_new=px.line(indicator)
  st.plotly_chart(figW_ind_new)
  st.write(indicator)
  
  
  
