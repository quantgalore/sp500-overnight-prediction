# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import binomial_option_price, bjerksund_stensland_greeks, Binarizer, return_proba

import requests
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import sqlalchemy
import mysql.connector

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar

polygon_api_key = "polygon.io api key, use code QUANTGALORE for 10% off"

calendar = get_calendar("NYSE")

underlying_ticker = "SPY"

today = (datetime.today() - timedelta(days = 1))
start_date = "2022-11-16"

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = start_date, end_date = today).index.strftime("%Y-%m-%d").values})

##

feature_price_data_list = []
option_chain_list = []
times = []

for date in trade_dates["trade_dates"]:
    
    start_time = datetime.now()
    
    if date == trade_dates["trade_dates"].iloc[-1]:
        next_trading_date = calendar.schedule(start_date = date, end_date = pd.to_datetime(date)+timedelta(days=5)).index.strftime("%Y-%m-%d").values[1]
        previous_trading_date = trade_dates[trade_dates.index == (trade_dates[trade_dates["trade_dates"] == date].index - 1)[0]].iloc[0][0]
        # get the 1-dte expiration date
        expiration_date = next_trading_date
        Underlying_Today = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{previous_trading_date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        Underlying_Today["returns"] = ((Underlying_Today["o"] - Underlying_Today["c"].shift(1)) / Underlying_Today["c"].shift(1)).fillna(0)
        today_overnight_return = Underlying_Today["returns"].iloc[-1]
        
        today_feature_price_dataframe = pd.DataFrame([{"timestamp": pd.to_datetime(date),
                                                  "open":Underlying_Today["o"].iloc[-1], "close":Underlying_Today["c"].iloc[0],
                                                    "net_call_volume": np.nan, "net_put_volume": np.nan,
                                                    "net_call_gamma": np.nan, "net_put_gamma": np.nan,
                                                    "mean_call_ivol": np.nan, "mean_put_vol": np.nan,
                                                    "total_gamma": np.nan, "total_volume": np.nan, "mean_ivol": np.nan,
                                                    "call_put_gamma_imbalance": np.nan, 
                                                    "gamma_times_volume": np.nan, "ivol_adjusted_net_gamma": np.nan}])
        
        feature_price_data_list.append(today_feature_price_dataframe)
        break
        
    else:
        # get the 1-dte expiration date
        expiration_date = trade_dates[trade_dates.index == (trade_dates[trade_dates["trade_dates"] == date].index + 1)[0]].iloc[0][0]
        
    # get the closing (4:00) price and -5/+5 strikes of that day
    Underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    Underlying.index = pd.to_datetime(Underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")

    Underlying_Price = Underlying["c"].iloc[0]
    Underlying_Open = Underlying["o"].iloc[0]
       
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
        continue
    if (len(put_data_list) < 1) or  (len(call_data_list) < 1):
        continue
    
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
                                             "open":Underlying_Open, "close":Underlying_Price,
                                               "net_call_volume": net_call_volume, "net_put_volume": net_put_volume,
                                               "net_call_gamma": net_call_gamma, "net_put_gamma": net_put_gamma,
                                               "mean_call_ivol": mean_call_vol, "mean_put_vol": mean_put_vol,
                                               "total_gamma": total_gamma, "total_volume": total_volume, "mean_ivol": total_mean_vol,
                                               "call_put_gamma_imbalance": call_put_gamma_imbalance, 
                                               "gamma_times_volume": gamma_times_volume, "ivol_adjusted_net_gamma": ivol_adjusted_gamma}])
                                
    feature_price_data_list.append(feature_price_dataframe)
    option_chain_list.append(Option_Chain)
    
    end_time = datetime.now()
    seconds_to_complete = (end_time - start_time).total_seconds()
    times.append(seconds_to_complete)
    iteration = round((np.where(trade_dates["trade_dates"]==expiration_date)[0][0]/len(trade_dates))*100,2)
    iterations_remaining = len(trade_dates) - np.where(trade_dates["trade_dates"]==expiration_date)[0][0]
    average_time_to_complete = np.mean(times)
    estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
    time_remaining = estimated_completion_time - datetime.now()
    
    print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

####

full_feature_price_data = pd.concat(feature_price_data_list).set_index("timestamp")

production_features = ['month','day','call_put_gamma_imbalance', 'mean_call_ivol', 'net_call_gamma','net_call_volume', 'total_volume']

Pre_Training_Dataset = full_feature_price_data.copy()
Pre_Training_Dataset["returns"] = ((Pre_Training_Dataset["open"] - Pre_Training_Dataset["close"].shift(1)) / Pre_Training_Dataset["close"].shift(1)).fillna(0).shift(-1)
Pre_Training_Dataset["month"] = Pre_Training_Dataset.index.month
Pre_Training_Dataset["day"] = Pre_Training_Dataset.index.day

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

with engine.connect() as conn:
    result = conn.execute(sqlalchemy.text(f'DROP TABLE {underlying_ticker}_option_backtest'))

Pre_Training_Dataset.to_sql(f"{underlying_ticker}_option_backtest", con = engine)

Option_Chain_DataFrame = pd.concat(option_chain_list).reset_index()

with engine.connect() as conn:
    result = conn.execute(sqlalchemy.text(f'DROP TABLE {underlying_ticker}_option_chain'))

Option_Chain_DataFrame.to_sql(f"{underlying_ticker}_option_chain", con = engine)
