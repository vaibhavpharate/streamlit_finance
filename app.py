from json import load
from os import close
#from re import S
import numpy as np
import pandas as pd
from streamlit import beta_util
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components

from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import math
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error,r2_score

st.set_page_config(layout="wide")
company = st.sidebar.selectbox("Select Company", ['TATACOFFEE.NS',"GMBREW.NS",'UBL.NS',"VBL.NS","SDBL.NS"])

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)
year = '7y'

if company == 'TATACOFFEE.NS':
    year = '15y'
else:
    year = '7y'


name_model = {"TATACOFFEE.NS":'lstm_1.h5',"GMBREW.NS":'akshat.h5','UBL.NS':'ubl_1.h5',"VBL.NS":'varun_1.h5',"SDBL.NS":'sdbl_1.h5'}
name_company = {"TATACOFFEE.NS":'Tata Coffee NSE',"GMBREW.NS":'G.M.Breweries','UBL.NS':'United Breweries Limited',"VBL.NS":"Varun Beverages Ltd","SDBL.NS":"Som Distilleries And Breweries Ltd"}
company_beta = {"TATACOFFEE.NS":'0.67',"GMBREW.NS":'0.43','UBL.NS':'0.962',"VBL.NS":"-0.78","SDBL.NS":"1.21"}
m_caps = {"TATACOFFEE.NS":38017032192,"GMBREW.NS":11186743296,'UBL.NS':415512461312,"VBL.NS":388690411520,"SDBL.NS":2635254272}
pb_ratio = {"TATACOFFEE.NS":2.09,"GMBREW.NS":2.22,'UBL.NS':11.59,"VBL.NS":10.83,"SDBL.NS":0.93}
errors_closing = {"TATACOFFEE.NS":[-1.08585169,  1.06522336, -0.11981459, -1.39245885, -0.21628   ,
       -0.79973401, -0.91046924, -0.83623734, -0.74393422,  0.0133395 ,
        1.81528767,  0.69001974,  1.51590479, -0.96286325,  1.2212872 ,
       -0.20944978, -1.85364728, -0.56737389,  1.92722747,  1.69605294,
       -1.61498578,  0.73187959,  0.5045477 , -0.68556861,  1.69966733,
        0.74900103,  1.26790833, -1.45961194,  0.5527758 , -1.22711711,
       -0.63769319],"GMBREW.NS":[ 1.2798287 ,  1.58891295, -0.12484145, -1.52990055,  1.89060682,
       -1.59801256, -0.54411964, -0.77097147, -0.0303971 , -0.57614794,
        0.71756143,  1.02706645, -1.50962238,  0.87916545, -1.80836209,
       -1.32000456, -0.81846669, -1.40000596,  0.71412343, -1.80525926,
        1.68387378, -1.24955119, -0.12978343, -1.26521101,  0.10797086,
       -0.94478904, -0.26956952,  0.30660229, -1.13332952,  1.82794822,
        1.55276495],'UBL.NS':[-0.41562925,  0.40986875, -1.21473154, -1.29208584,  0.66820892,
        0.5266936 ,  1.77877092,  0.83692102,  0.3634152 , -0.31435161,
       -0.4742284 , -1.33672559,  0.79952574,  0.0588116 ,  0.93280172,
        1.12648665,  0.42614735, -1.15239627,  1.91037756,  1.91192434,
       -1.39683105, -1.05630575, -1.65488723, -0.06797449,  1.9462345 ,
        0.10285305, -1.83446881,  1.2906857 , -1.59660968,  0.32291098,
       -1.152749  ],"VBL.NS":[-0.95104616, -0.4177579 , -0.40432542,  1.68061768,  0.79307644,
        0.44538621, -0.96996978,  0.30295693, -0.19689268, -0.92683842,
       -0.87089419,  1.18352916,  1.66089555,  1.73104902,  0.35031735,
        1.5475097 , -0.72278816,  1.79025112,  1.4351023 ,  1.72859693,
       -0.19160078,  0.88926828,  0.71450367, -1.83381444, -1.68858831,
        0.2932397 ,  0.61093374, -1.54506743,  0.41542716, -1.97296582,
        0.92631389],"SDBL.NS":[ 1.39329268,  0.09666991, -0.105865  ,  0.67511885, -0.47184181,
       -1.14387292, -1.33847317,  0.23110492,  1.5758312 ,  0.35780715,
        1.61724912, -0.80293484,  1.36742093,  0.00234662,  0.38222281,
        0.15108937,  0.11367238, -0.30080174,  0.41736979,  1.28308901,
       -1.88785703, -1.91811684,  1.35451094, -0.42054111,  1.03638162,
       -0.21540528,  0.1295123 , -0.5069668 ,  1.3057917 ,  0.666916  ,
       -1.17070356]}

today = datetime.today().day
error_now = errors_closing[company][today-1]
#tata_coffee = yf.Ticker('TATACOFFEE.NS')
tata_coffee = yf.Ticker(company)
stock_price = tata_coffee.history(period=year)
info_all = tata_coffee.info


news_data = {
    'TATACOFFEE.NS':{
        'date':'Jul 27',
        "heading":"Investor sentiment improved over the past week",
        "content":"""<div><p>After last week's 28% share price gain to ₹235, the stock trades at a trailing P/E ratio of 32.9x.</p>
            <ul>
                <li>Average trailing P/E is 20x in the Food industry in India.</li>
                <li>Total returns to shareholders of 115% over the past three years.</li>
            </ul>
        </div>"""
    },
    "VBL.NS":{
        'date':'Aug 09',
        "heading":"Insider recently sold ₹13m worth of stock",
        "content":"""<div><p>On the 5th of August, Ravindra Dhariwal sold around 17k shares on-market at roughly </p>
        ₹781 per share. In the last 3 months, there was an even bigger sale from another insider worth ₹1.7b. 
        Insiders have been net sellers, collectively disposing of ₹1.7b more than they bought in the last 12 months.
        </div>"""
    },
    "UBL.NS":{
        'date':'Aug 04',
        'heading':'CII penalized a penalty of ₹752 cr to ubl and carlsberg',
        'content':"""<div><p> beer fixing prize in 7 states of India</p>
        </div>"""
    },
    "SDBL.NS":{
        'date':'Mar 01',
        'heading':'New 90-day high: ₹35.05',
        'content':"""<div><p>The company is up 35% from its price of ₹26.00 on 01 December 2020. The Indian market is up 16% over the last 90 days,
         indicating the company outperformed over that time. It also outperformed the Beverage industry, which is up 9.0% over the same period.
        </p></div> """
    },
     "GMBREW.NS":{
        'date':'Jun 05',
        'heading':'Investor sentiment improved over the past week',
        'content':"""
        <div>
            <p>After last week's 22% share price gain to ₹524, the stock trades at a trailing P/E ratio of 12x.</p>
                <ul>
                    <li>Average trailing P/E is 30x in the Beverage industry in India.</li>
                    <li>Total loss to shareholders of 34% over the past three years.</li>
                </ul>
            </div>"""
    },

}



stock_price_daily = tata_coffee.history(period=year,step='1d')
stock_price  = stock_price.style.format({"Daily": lambda t: t.strftime("%m/%d/%Y")}).data
stock_price['Daily'] = stock_price.index.values
stock_price['Daily Returns'] = stock_price['Close'].pct_change()

last_pct = stock_price['Daily Returns'][-1]
sheet_balance= tata_coffee.balance_sheet
color_close = "no"
if last_pct > 0:
    color_close = 'success'
elif last_pct < 0:
    color_close = 'danger'

year_change = info_all['52WeekChange']
color_year_change = "no"
if year_change > 0:
    color_year_change = 'success'
elif year_change < 0:
    color_year_change = 'danger'

#st.write(stock_price['Daily Returns'])
now = stock_price['Daily'].max()
year_back =stock_price['Daily'].max() - timedelta(weeks=52)
df_52 = stock_price[(stock_price['Daily']>year_back) &  (stock_price['Daily']<now)]
high52 = df_52['High'].max()
low52 = df_52['Low'].min()

st.title(name_company[company])
close_price = stock_price['Close']
#col_d1,col_d2,col_d3,col_d4 = st.columns([1,1,1,1]) 
st.markdown(
    f"""<div class='container-fluid mb-4 px-0'>
    <div class='row mx-0'>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>Closing Price</h3></div>
                <div class='card-body h5 text-{color_close}' >
                {close_price[-1]:.2f}
                </div>
            </div>
        </div>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>52 Week H/L</h3></div>
                <div class='card-body h5'>
                {high52:.2f} / {low52:.2f}
                </div>
            </div>
        </div>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>52W Change    </h3></div>
                <div class='card-body h5 text-{color_year_change}'>
                {year_change*100:.2f}%
                </div>
            </div>
        </div>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>Beta 5Y Monthly</h3></div>
                <div class='card-body h5'>
                {company_beta[company]}
                </div>
            </div>
        </div>
    </div>
    </div>""", unsafe_allow_html=True
)
col_head1,col_head2 = st.columns([1,1])
col_head1.subheader("Stock Prices")
col_head1.write(stock_price.iloc[:,:-2].sort_index(ascending=False))
dougnnut_m_cap = px.pie(names=m_caps.keys(),values=m_caps.values(),hole=.5)
col_head2.subheader("Market Caps")
col_head2.plotly_chart(dougnnut_m_cap)

m50 = close_price.rolling(50).mean()
m100 = close_price.rolling(100).mean()
m250 = close_price.rolling(250).mean()


st.header("Moving Averages")
st.line_chart(data=pd.DataFrame(data={"Closing Price":close_price,"50 Days":m50,"100 Days":m100}),)

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

st.subheader("Significant News")
st.markdown(f"""
    <div class='card bg-transparent  border-secondary'>
        <div class='card-header bg-dark h3'>
            {news_data[company]['heading']}
        </div>
        <div class='card-body'>{news_data[company]['content']}</div>
        <div class='card-footer muted text-right'>
            {news_data[company]['date']}
        </div>
    </div>
""", unsafe_allow_html=True)


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

col1, col2 = st.columns(2)
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

#st.header(f"Accuracy of model {r2_score(data['Original'],data['Predicted'])*100}")

st.header("Balance Sheet")
st.write(sheet_balance)

st.header("Financials")
st.write(tata_coffee.financials)

col_la1,col_la2= st.columns([6,6])

#col_la1.header("Liabilities Vs Assets")

balance_sheet =  sheet_balance
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
            y=liab_assets["Total Liab"][::-1],
            offsetgroup=0,
        ),
        go.Bar(
            name="Assets",
           x = ['2018','2019','2020','2021'],
            y=liab_assets["Total Assets"][::-1],
            offsetgroup=1,
        ),
        
    ],
    layout=go.Layout(
        title="Total Liabilities Vs Total Assets",
    
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
            y=revenue_income["Total Revenue"][::-1],
            offsetgroup=0,
        ),
        go.Bar(
            name="Net Income",
             x = ['2018','2019','2020','2021'][::-1],  
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

## Current Assets and liabilities
current_a_l = balance_sheet.loc[['Total Current Assets','Total Current Liabilities']].transpose()
fig_ca_cl = go.Figure(
    data=[
        go.Bar(
            name="Current Liablities",
            x = ['2018','2019','2020','2021'],
            y=current_a_l["Total Current Liabilities"][::-1].values,
            offsetgroup=0,
        ),
        go.Bar(
            name="Current Assets",
             x = ['2018','2019','2020','2021'],  
            y=current_a_l["Total Current Assets"][::-1].values,
            offsetgroup=1,
        ),
    ],
    layout=go.Layout(
        title="Current Assets vs Current Liabilities",
        yaxis_title="Cost"
    )
)



col_cal1,col_cal_2 = st.columns([1,1])
col_cal1.plotly_chart(fig_ca_cl)
liquidity_ratio = pd.DataFrame(current_a_l['Total Current Assets']/ current_a_l['Total Current Liabilities'],columns=['lq'])
ca_cl = px.line(y=liquidity_ratio['lq'],x=liquidity_ratio.index.values,markers='O')
ca_cl.update_layout(xaxis_title="Time",
    yaxis_title="CA/CL Ratio",title="Liquidity Ratio")
col_cal_2.plotly_chart(ca_cl)


### Cash Flow
demo_cash_flow =  pd.DataFrame(tata_coffee.cashflow.loc['Investments',:],columns=['Investments'])
demo_cash_flow['Operation'] = financials['Operating Income']
demo_cash_flow['Financing']=tata_coffee.cashflow.loc['Total Cash From Financing Activities']
#demo_cash_flow['Capital Expenditures'] = tata_coffee.cashflow.loc['Capital Expenditures']

st.subheader("Cash Flow")
st.write(demo_cash_flow)

fig_cf = go.Figure(
    data=[
        go.Line(
        name="Investments",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Investments'][::-1],
            legendgroup = 0
        ),
        go.Line(
        name="Operation",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Operation'][::-1],
            legendgroup = 1
        ),
        go.Line(
        name="Financing",
            x = ['2018','2019','2020','2021'], 
            y = demo_cash_flow['Financing'][::-1],
            legendgroup = 2
        ),
        
    ],
    layout=go.Layout(

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

#st.write(f"Next Day Prediction {scaler.inverse_transform(lst_output)[0]}")
st.markdown(
    f"""<div class='container-fluid mb-4 px-0'>
    <div class='row mx-0'>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>Closing Price</h3></div>
                <div class='card-body h5 ' >
                {close_price[-1]:.2f}
                </div>
            </div>
        </div>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>Predicted Price </h3></div>
                <div class='card-body h5'>
                {abs(close_price[-1]*error_now):.2f} 
                </div>
            </div>
        </div>
        <div class='col'>
            <div class='card  bg-transparent border-secondary' >
                <div class='card-header bg-dark'><h3 class='mb-0 p-0'>Error Percentage    </h3></div>
                <div class='card-body h5 text'>
                {error_now:.2f}%
                </div>
            </div>
        </div>
    </div>
    </div>""", unsafe_allow_html=True)
for_week , for_month = st.columns([6,6])
forecast_week = px.line(scaler.inverse_transform(lst_output[:7]))
for_week.markdown("7 Days Forecast")
for_week.plotly_chart(forecast_week)

forecast = px.line(scaler.inverse_transform(lst_output))
for_month.markdown("30 Days Forecast")
for_month.plotly_chart(forecast)