import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
import plotly.express as px
import requests


st.title("Stock Dashboard")
ticker=st.sidebar.text_input("Ticker")
start_date=st.sidebar.date_input("Start Date")
end_date=st.sidebar.date_input("End Date")
ticker1="AAPL"
start_date1=date(2022,3,31)
end_date1=date(2023,4,1)
data=yf.download(ticker,start=start_date,end=end_date)
fig=px.line(data,x=data.index,y=data['Adj Close'],title=ticker)
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
  
  
  
