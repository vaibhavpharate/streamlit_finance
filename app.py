from json import load
from os import close
#from re import S
import numpy as np
import pandas as pd
from streamlit import beta_util
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import math
from sklearn.metrics import mean_squared_error,r2_score

st.set_page_config(layout="wide")
company = st.sidebar.selectbox("Select Company", ['TATACOFFEE.NS',"GMBREW.NS",'UBL.NS',"VBL.NS","SDBL.NS"])


year = '7y'

if company == 'TATACOFFEE.NS':
    year = '15y'
else:
    year = '7y'


name_model = {"TATACOFFEE.NS":'lstm_1.h5',"GMBREW.NS":'akshat.h5','UBL.NS':'ubl_1.h5',"VBL.NS":'varun_1.h5',"SDBL.NS":'sdbl_1.h5'}
name_company = {"TATACOFFEE.NS":'Tata Coffee NSE',"GMBREW.NS":'G.M.Breweries','UBL.NS':'United Breweries Limited',"VBL.NS":"Varun Beverages Ltd","SDBL.NS":"Som Distilleries And Breweries Ltd"}


#tata_coffee = yf.Ticker('TATACOFFEE.NS')
tata_coffee = yf.Ticker(company)
stock_price = tata_coffee.history(period=year)

stock_price_daily = tata_coffee.history(period=year,step='1d')
stock_price  = stock_price.style.format({"Daily": lambda t: t.strftime("%m/%d/%Y")}).data
stock_price['Daily'] = stock_price.index.values
st.title(name_company[company])
st.write(stock_price.iloc[:-1].sort_index(ascending=False))

close_price = stock_price['Close']
m50 = close_price.rolling(50).mean()
m100 = close_price.rolling(100).mean()
m250 = close_price.rolling(250).mean()


st.header("Moving Averages")
st.line_chart(data=pd.DataFrame(data={"Closing Price":close_price,"50 Days":m50,"100 Days":m100}),)
stock_price['Daily Returns'] = stock_price['Close'].pct_change()

candle_layout = go.Layout(autosize=True,
    width=1440,height=800)

# st.write(stock_price)

stock_price_2  = stock_price[stock_price['Daily']>'2021-01-01']

st.header("Candlestick plot for year 2021")
candle_stick = go.Figure(data=[go.Candlestick(x=stock_price_2['Daily'],
                open=stock_price_2['Open'],
                high=stock_price_2['High'],
                low=stock_price_2['Low'],
                close=stock_price_2['Close'])],layout=candle_layout)

st.plotly_chart(candle_stick)



def tell_status(daily_ret):
    val = daily_ret
    if val > 0:
        return "P"
    elif val < 0:
        return "N"
    elif val == 0:
        return "Z"
    else:
        return np.NaN

stock_price['ret_status'] = stock_price['Daily Returns'].apply(tell_status)
fig_returns = px.line(stock_price[1:],y='Daily Returns',color='ret_status', width=1440)
st.plotly_chart(fig_returns)

col1, col2 = st.beta_columns(2)
col1.header("Daily Returns Distribution")
fig_returns_pie = px.pie(stock_price[2:],names='ret_status')
col1.plotly_chart(fig_returns_pie)

col2.header('Daily Returns Table')

col2.write(pd.DataFrame({'Index':stock_price['ret_status'].value_counts().index.values,"Values":stock_price['ret_status'].value_counts().values}))


## Load my model
#model = load_model('lstm_1.h5')
model = load_model(name_model[company])

## Preprocessor
scaler = MinMaxScaler(feature_range=(0,1))
df_transform = scaler.fit_transform(np.array(close_price).reshape(-1,1)) 

# split data
training_size = int(len(df_transform)*0.65)
test_size = int(len(df_transform) - training_size)
train_set,test_set = df_transform[:training_size,:],df_transform[training_size:len(df_transform),:1] 

## Considering last 100 days and create a dataset
def create_datasets(data,step):
    data_X,data_Y = [],[]
    for i in range(len(data)-step-1):
        a = data[i:(i+step),0]
        data_X.append(a)
        data_Y.append(data[i+step,0])
    return np.array(data_X),np.array(data_Y)
# splitting the data
X_train,y_train = create_datasets(train_set,100)
X_test,y_test = create_datasets(test_set,100)

## reshape for input
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


## Predict train and test set
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

## Inverse Transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

## Calculate mean squared error
st.header(f"Mean squared Error = {math.sqrt(mean_squared_error(y_test,test_predict))}")

## collecting data predictions and actual closing price to plot graph
look_back = 100
train_predict_plot = np.empty_like(df_transform)
train_predict_plot[:,:] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
#train_predict = scaler.inverse_transform(train_predict_plot)
#
test_predict_plot = np.empty_like(df_transform)
test_predict_plot[:,:] = np.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(df_transform)-1, :] = test_predict
#test_predict = scaler.inverse_transform(test_predict_plot)

dummy_data = pd.DataFrame({'Actual Closing':close_price,"Train Prediction":train_predict_plot.flatten(),"Test Prediction":test_predict_plot.flatten()})
dummy_data = dummy_data.stack().reset_index(level=[0,1])
dummy_data.columns = ['Date',"Status","Closing Value"]



plot_ex_main = px.line(data_frame=dummy_data,x='Date',y='Closing Value',color='Status',width=1440)
st.plotly_chart(plot_ex_main)

data = pd.DataFrame(scaler.inverse_transform(y_train.reshape(1,-1))[0],columns=['Original'])
data['Predicted'] = train_predict

if company == "TATACOFFEE.NS":
    data = pd.DataFrame(scaler.inverse_transform(y_test.reshape(1,-1))[0],columns=['Original'])
    data['Predicted'] = test_predict

st.header(f"Accuracy of model {r2_score(data['Original'],data['Predicted'])*100}")

st.header("Balance Sheet")
st.write(tata_coffee.balance_sheet)

st.header("Financials")
st.write(tata_coffee.financials)

col_la1,col_la2= st.beta_columns([6,6])

#col_la1.header("Liabilities Vs Assets")

balance_sheet =  tata_coffee.balance_sheet
liab_assets = balance_sheet.loc[['Total Liab','Total Assets']].transpose()
liab_assets = liab_assets.reset_index()
liab_assets.columns = ['Year',"Total Liab","Total Assets"]

financials = tata_coffee.financials.transpose()
profit_margin =  pd.DataFrame(financials['Gross Profit'] / financials['Total Revenue'] * 100,columns=['Profit Margin']).reset_index()
profit_margin.columns = ['Year',"Profit Margin"]
debt_to_asset_ratio = (liab_assets['Total Liab'] / liab_assets['Total Assets']) *100
debt_to_asset_ratio = pd.DataFrame(debt_to_asset_ratio)
debt_to_asset_ratio['Year'] = range(2018,2022)
fig2 = go.Figure(
    data=[
        go.Bar(
            name="Liabilities",
            x = ['2018','2019','2020','2021'],
            y=liab_assets["Total Liab"],
            offsetgroup=0,
        ),
        go.Bar(
            name="Assets",
           x = ['2018','2019','2020','2021'],
            y=liab_assets["Total Assets"],
            offsetgroup=1,
        ),
        
    ],
    layout=go.Layout(
        title="Liabilities Vs Assets",
    )
    
)
col_la1.plotly_chart(fig2)
fig_dta = px.line(x=debt_to_asset_ratio['Year'],y=debt_to_asset_ratio[0],text=debt_to_asset_ratio[0])
fig_dta.update_layout(title='Debt to Asset %',
                   xaxis_title='Year',
                   yaxis_title='Debt to Asset %')
col_la1.plotly_chart(fig_dta)

# go.Line(
#         name="Profit Margins",
#             x = ['2018','2019','2020','2021'], 
#             y = debt_to_asset_ratio[0],
#             legendgroup = 2
#         )


### Revenue and net Income
revenue_income = financials.loc[:,["Total Revenue","Net Income"]].reset_index()
revenue_income.columns = ['Year',"Total Revenue","Net Income"]
fig3 = go.Figure(
    data=[
        go.Bar(
            name="Total Revenue",
            x = ['2018','2019','2020','2021'],
            y=revenue_income["Total Revenue"],
            offsetgroup=0,
        ),
        go.Bar(
            name="Net Income",
             x = ['2018','2019','2020','2021'],  
            y=revenue_income["Net Income"],
            offsetgroup=1,
        ),
    
    ],
     layout=go.Layout(
        title="Total Revenue Vs Net Income",
    )
    
)
profit_margin_chart = px.line(x=profit_margin['Year'],y=profit_margin['Profit Margin'])
profit_margin_chart.update_layout(title='Profit Margin %',
                   xaxis_title='Year',
                   yaxis_title='Profit Margin %')
col_la2.plotly_chart(fig3)
col_la2.plotly_chart(profit_margin_chart)

### Cash Flow
demo_cash_flow =  pd.DataFrame(tata_coffee.cashflow.loc['Investments'])
demo_cash_flow['Operation'] = financials['Operating Income']
demo_cash_flow['Financing']=tata_coffee.cashflow.loc['Total Cash From Financing Activities']
#demo_cash_flow['Capital Expenditures'] = tata_coffee.cashflow.loc['Capital Expenditures']
fig_cf = go.Figure(
    data=[
        go.Line(
        name="Investments",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Investments'],
            legendgroup = 0
        ),
        go.Line(
        name="Operation",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Operation'],
            legendgroup = 1
        ),
        go.Line(
        name="Financing",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Financing'],
            legendgroup = 2
        ),
        
    ],
    layout=go.Layout(
        title="Cash Flow",
        xaxis_title="Year",
        width=1440
    )
)

st.plotly_chart(fig_cf)


## Forecasting
x_input = test_set[len(test_set)-100:].reshape(1,-1)
#x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

st.write(f"Next Day Prediction {scaler.inverse_transform(lst_output)[0]}")

for_week , for_month = st.beta_columns([6,6])
forecast_week = px.line(scaler.inverse_transform(lst_output[:7]))
for_week.markdown("7 Days Forecast")
for_week.plotly_chart(forecast_week)

forecast = px.line(scaler.inverse_transform(lst_output))
for_month.markdown("30 Days Forecast")
for_month.plotly_chart(forecast)