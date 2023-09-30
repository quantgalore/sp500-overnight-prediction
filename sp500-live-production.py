# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import binomial_option_price, bjerksund_stensland_greeks, Binarizer, return_proba
from self_email import send_message

import requests
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import sqlalchemy
import mysql.connector
import yfinance as yf
import pytz

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
from sklearn.ensemble import RandomForestClassifier

start_time = datetime.now(tz=pytz.timezone("US/Eastern"))

polygon_api_key = "polygon.io api key, use code QUANTGALORE for 10% off"

calendar = get_calendar("NYSE")

underlying_ticker = "SPY"

today = datetime.today()
end_date = today + timedelta(days = 5)
days_before = today - timedelta(days = 5)

# get today and the next option expiration
trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = today, end_date = end_date).index.strftime("%Y-%m-%d").values[:2]})
# last trade date for yfinance
prior_trade_date = pd.DataFrame([{"trade_dates": calendar.schedule(start_date = days_before, end_date = today).index.strftime("%Y-%m-%d").values[-2]}])["trade_dates"].iloc[0]

feature_price_data_list = []

# today
date = trade_dates["trade_dates"].iloc[0]
# get the 1-dte expiration date
expiration_date = trade_dates["trade_dates"].iloc[-1]

# get the closing (4:00) price and -5/+5 strikes of that day
Underlying = yf.download("SPY", start = date, end = expiration_date, interval = "1d").tail(1)
Underlying_Price = Underlying["Adj Close"].iloc[0]

# assign the price parameter for option model
S = Underlying_Price

# get the contract names of the -5/+5 strikes (p and c)
Call_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&expiration_date={expiration_date}&as_of={expiration_date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
Call_Contracts["distance_from_price"] = abs(Call_Contracts["strike_price"] - S)
Call_Symbols = Call_Contracts.nsmallest(n = 10,columns = "distance_from_price", keep ="all")["ticker"].values

Put_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=put&expiration_date={expiration_date}&as_of={expiration_date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
Put_Contracts["distance_from_price"] = abs(Put_Contracts["strike_price"] - S)
Put_Symbols = Put_Contracts.nsmallest(n = 10,columns = "distance_from_price", keep ="all")["ticker"].values

# get the pricing and greek data for all of the options in the strike range
call_data_list = []
put_data_list = []

for call in Call_Symbols:
    
    # refer to the original contract symbol, strike, etc.
    call_contract_info = Call_Contracts[Call_Contracts["ticker"] == call]

    # get the closing (4:00) prices and volume
    try:
        call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{call}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    except:
        break
    call_ohlcv.index = pd.to_datetime(call_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
    
    # option pricing parameters, K = strike price, T = time to expiration, r = interest rate, n = timesteps (binomial model)
    K = call_contract_info["strike_price"].iloc[0]
    T = 1/252
    r = 0.05
    n = 500
    
    # the function we use to solve for implied vol, where the value of volatility makes the theo price = to the real price
    def f_call(volatility):
        return binomial_option_price(S, K, T, r, volatility, n, option_type = "call") - call_ohlcv["c"].iloc[0]

    # sometimes this fails to converge when the data isn't great, e.g., price < intrinsic value for wide spread ITM    
    try:    
        # the Newton-Raphson method to iterate and find the vol where the theo price is no more than $0.05 the real price
        call_implied_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
    except:
        # just carry over the last IV for the last strike, this rarely fails on the first iteration, so there'll usually be an existing vol
            pass
            
    # back-solve the greeks from bjerksund_stensland_greeks
    call_delta, call_gamma, call_theta, call_vega = bjerksund_stensland_greeks(S, K, T, r, sigma=call_implied_vol, option_type="call")
    
    # store the data 
    call_dataframe = pd.DataFrame([{"timestamp": date,"strike_price": K,
                                    "call_symbol": call,
                                    "call_delta": call_delta, "call_gamma": call_gamma,
                                    "call_theta": call_theta, "call_vega": call_vega,
                                    "call_implied_vol": call_implied_vol,
                                    "call_open": call_ohlcv["o"].iloc[0],"call_high": call_ohlcv["h"].iloc[0],
                                    "call_low": call_ohlcv["l"].iloc[0],"call_close": call_ohlcv["c"].iloc[0],
                                    "call_vw": call_ohlcv["vw"].iloc[0],"call_volume": call_ohlcv["v"].iloc[0]}])

    call_data_list.append(call_dataframe)

for put in Put_Symbols:
    
    put_contract_info = Put_Contracts[Put_Contracts["ticker"] == put]
    
    try:
        put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{put}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    except:
        break
    put_ohlcv.index = pd.to_datetime(put_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
    
    K = put_contract_info["strike_price"].iloc[0]
    T = 1/252
    r = 0.05
    n = 500
    
    def f_put(volatility):       
        return binomial_option_price(S, K, T, r, volatility, n, option_type = "put") - put_ohlcv["c"].iloc[0]

    try:    
      # Use the Newton-Raphson method to find the root
      put_implied_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
    except:
        # just carry over the IV for the last strike
        pass
  
    if put_implied_vol < 0:
    
        last_vol = put_data_list[-1]["put_implied_vol"].iloc[0]
        put_implied_vol = last_vol
    
    put_delta, put_gamma, put_theta, put_vega = bjerksund_stensland_greeks(S, K, T, r, sigma=put_implied_vol, option_type="put")
    
    put_dataframe = pd.DataFrame([{"timestamp": date,"strike_price": K,
                                    "put_symbol": put,
                                    "put_delta": put_delta, "put_gamma": put_gamma,
                                    "put_theta": put_theta, "put_vega": put_vega,
                                    "put_implied_vol": put_implied_vol,
                                    "put_open": put_ohlcv["o"].iloc[0],"put_high": put_ohlcv["h"].iloc[0],
                                    "put_low": put_ohlcv["l"].iloc[0],"put_close": put_ohlcv["c"].iloc[0],
                                    "put_vw": put_ohlcv["vw"].iloc[0],"put_volume": put_ohlcv["v"].iloc[0]}])

    put_data_list.append(put_dataframe)
    

if len(put_data_list) != len(call_data_list):
    raise Exception
if (len(put_data_list) < 1) or  (len(call_data_list) < 1):
    raise Exception

# combine all call/put datasets
Call_Data = pd.concat(call_data_list).set_index("timestamp")
Put_Data = pd.concat(put_data_list)

# combine both sides to create an option chain
Option_Chain = pd.merge(Put_Data, Call_Data, on ="strike_price").set_index("timestamp")

Option_Chain["underlying_price"] = S
# this will be used for selecting the nearest ITM strike to trade
Option_Chain["distance_from_price"] = abs(Option_Chain["strike_price"] - S)

net_call_gamma = Option_Chain["call_gamma"].sum()
net_call_volume = Option_Chain["call_volume"].sum()
mean_call_vol = Option_Chain["call_implied_vol"].mean()

net_put_gamma = Option_Chain["put_gamma"].sum()
net_put_volume = Option_Chain["put_volume"].sum()
mean_put_vol = Option_Chain["put_implied_vol"].mean()

total_gamma = net_call_gamma + net_put_gamma
total_volume = net_call_volume + net_put_volume
total_mean_vol = (mean_call_vol + mean_put_vol) / 2

# features
call_put_gamma_imbalance = net_call_gamma - net_put_gamma
gamma_times_volume = total_gamma * total_volume
ivol_adjusted_gamma = total_mean_vol * total_gamma

feature_price_dataframe = pd.DataFrame([{"timestamp": pd.to_datetime(date),
                                         "open":Underlying_Price, "close":Underlying_Price,
                                           "net_call_volume": net_call_volume, "net_put_volume": net_put_volume,
                                           "net_call_gamma": net_call_gamma, "net_put_gamma": net_put_gamma,
                                           "mean_call_ivol": mean_call_vol, "mean_put_vol": mean_put_vol,
                                           "total_gamma": total_gamma, "total_volume": total_volume, "mean_ivol": total_mean_vol,
                                           "call_put_gamma_imbalance": call_put_gamma_imbalance, 
                                           "gamma_times_volume": gamma_times_volume, "ivol_adjusted_net_gamma": ivol_adjusted_gamma}]).set_index("timestamp")

feature_price_dataframe["month"] = feature_price_dataframe.index.month
feature_price_dataframe["day"] = feature_price_dataframe.index.day
                        
engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
Pre_Training_Dataset = pd.read_sql("SELECT * FROM sp500_option_production", con = engine).set_index("timestamp")

training_features = ['month', 'day','call_put_gamma_imbalance', 'mean_call_ivol', 'net_call_gamma','net_call_volume', 'total_volume', 'returns']
production_features = ['month', 'day','call_put_gamma_imbalance', 'mean_call_ivol', 'net_call_gamma','net_call_volume', 'total_volume']

Training_Dataset = Pre_Training_Dataset[training_features].copy().dropna()

Training_Dataset_Long = Training_Dataset[Training_Dataset.index < date].tail(100)
Training_Dataset_Short = Training_Dataset[Training_Dataset.index < date].tail(20)

X_Long = Training_Dataset_Long.drop("returns", axis = 1).values
Y_Long = Training_Dataset_Long["returns"].apply(Binarizer).values

X_Short = Training_Dataset_Short.drop("returns", axis = 1).values
Y_Short = Training_Dataset_Short["returns"].apply(Binarizer).values

RandomForest_Model_Long = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X=X_Long, y=Y_Long)
RandomForest_Model_Short = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X=X_Short, y=Y_Short)

Production_Feature_Data = feature_price_dataframe[production_features].copy().values

Long_Prediction = RandomForest_Model_Long.predict(Production_Feature_Data)
Short_Prediction = RandomForest_Model_Short.predict(Production_Feature_Data)

Long_Probability = RandomForest_Model_Long.predict_proba(Production_Feature_Data)
Short_Probability = RandomForest_Model_Short.predict_proba(Production_Feature_Data)

Long_Prediction_DataFrame = pd.DataFrame([{"prediction": Long_Prediction[0]}])
Long_Prediction_DataFrame["probability_0"] = Long_Probability[:,0]
Long_Prediction_DataFrame["probability_1"] = Long_Probability[:,1]
Long_Prediction_DataFrame["probability"] = return_proba(Long_Prediction_DataFrame)

Short_Prediction_DataFrame = pd.DataFrame([{"prediction": Short_Prediction[0]}])
Short_Prediction_DataFrame["probability_0"] = Short_Probability[:,0]
Short_Prediction_DataFrame["probability_1"] = Short_Probability[:,1]
Short_Prediction_DataFrame["probability"] = return_proba(Short_Prediction_DataFrame)

end_time = datetime.now(tz=pytz.timezone("US/Eastern"))
Elapsed_Time = end_time - start_time

# Nearest_ITM

if Long_Prediction[0] == Short_Prediction[0]:
    
    probability = round((Long_Prediction_DataFrame["probability"].iloc[0] + Short_Prediction_DataFrame["probability"].iloc[0]) / 2, 2)
    Random_Forest_Prediction = Long_Prediction
    
    prediction_string = f"Prediction on {date}: {Random_Forest_Prediction[0]}, probability of {probability*100}% Elapsed Time: {Elapsed_Time}"
    print(prediction_string)
    send_message(message = prediction_string, subject = f"Trade Output on {today.strftime('%A')}, {date}")
else:
    prediction_string = f"Prediction of long and short dataset do not match. Do not trade. Elapsed Time: {Elapsed_Time}"
    print(prediction_string)
    send_message(message = prediction_string, subject = f"Trade Output on {today.strftime('%A')}, {date}")