# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import binomial_option_price, bjerksund_stensland_greeks, Binarizer, return_proba, round_to_multiple

import requests
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import sqlalchemy
import mysql.connector
import yfinance as yf

from pandas_market_calendars import get_calendar

from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestClassifier

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
polygon_api_key = "polygon.io api key, use code QUANTGALORE for 10% off"

calendar = get_calendar("NYSE")

production_features = ['month', 'day', 'call_put_gamma_imbalance', 'mean_call_ivol', 'net_call_gamma','net_call_volume', 'total_volume', 'returns']
option_features = ['month', 'day','call_put_gamma_imbalance', 'mean_call_ivol', 'net_call_gamma','net_call_volume', 'total_volume']

underlying_ticker = "SPY"

dataset = pd.read_sql(f"SELECT * FROM {underlying_ticker}_option_backtest", con = engine).set_index("timestamp")
option_chain_dataset = pd.read_sql(f"SELECT * FROM {underlying_ticker}_option_chain", con = engine).set_index("timestamp")
option_chain_dataset.index = pd.to_datetime(option_chain_dataset.index)

featured_dataset = dataset[production_features].copy().dropna()
first_available_date = featured_dataset.index[100]
trade_dates = featured_dataset[featured_dataset.index > first_available_date].index

# if you want to backtest using simulated E-Mini Futures, un-comment the commented lines

# Futures = round_to_multiple(yf.download("^GSPC", start = "2023-01-01", end = "2023-12-31", interval = "1d"), .25)

###
prediction_actual_list = []

for date in trade_dates:
       
    if date == trade_dates[-1]:
        break
    
    start_time= datetime.now()
    
    date_index = (np.where(trade_dates == date)[0][0])
    
    Training_Dataset_Long = featured_dataset[featured_dataset.index < date].tail(100)
    Training_Dataset_Short = featured_dataset[featured_dataset.index < date].tail(20)
    
    X_Long = Training_Dataset_Long.drop("returns", axis = 1).values
    Y_Long = Training_Dataset_Long["returns"].apply(Binarizer).values
    
    X_Short = Training_Dataset_Short.drop("returns", axis = 1).values
    Y_Short = Training_Dataset_Short["returns"].apply(Binarizer).values
    
    RandomForest_Model_Long = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X=X_Long, y=Y_Long)
    RandomForest_Model_Short = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X=X_Short, y=Y_Short)
    
    next_day = trade_dates[date_index+1]
    
    Production_Feature_Data = featured_dataset[featured_dataset.index == date].drop("returns", axis =1).values
    
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
    
    if Long_Prediction[0] == Short_Prediction[0]:
        
        probability = round((Long_Prediction_DataFrame["probability"].iloc[0] + Short_Prediction_DataFrame["probability"].iloc[0]) / 2, 2)
        Random_Forest_Prediction = Long_Prediction
    else:
        continue 
    
    # buy option
    
    options = option_chain_dataset[option_chain_dataset.index == date]
    gamma_imbalance = options["call_gamma"].sum() - options["put_gamma"].sum()
    
    Nearest_ITM = options[options["distance_from_price"] == options["distance_from_price"].min()]
    
    if Random_Forest_Prediction[0] == 0:
        next_contract = Nearest_ITM["put_symbol"].iloc[0]
    elif Random_Forest_Prediction[0] == 1:
        next_contract = Nearest_ITM["call_symbol"].iloc[0]
    
    if next_contract == Nearest_ITM["put_symbol"].iloc[0]:
        Open_Trade_Price = Nearest_ITM["put_close"].iloc[0]
                
    elif next_contract == Nearest_ITM["call_symbol"].iloc[0]:
        Open_Trade_Price = Nearest_ITM["call_close"].iloc[0]
        
    # un comment if you are simulating futures
        
    # Open_Trade_Price = Futures[Futures.index == date]["Adj Close"].iloc[0]
    # Closing_Trade_Price = Futures[Futures.index == next_day]["Open"].iloc[0]
    
    # Ticks = (Closing_Trade_Price - Open_Trade_Price) / .25
    # dv_per_tick = 12.25
    
    # pnl = Ticks * dv_per_tick
    
    # if pnl <= 0 and Random_Forest_Prediction[0] == 0:
        
    #     net_pnl = abs(pnl)
        
    # elif pnl >= 0 and Random_Forest_Prediction[0] == 0:
        
    #     net_pnl = abs(pnl)*-1
    # else:
    #     net_pnl = pnl
    #     gross_pnl = np.nan
        
    
    # comment out the lines below until Actual_Return if you are simulating futures. 
    
    closing_options = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{next_contract}/range/1/day/{next_day.strftime('%Y-%m-%d')}/{next_day.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    closing_options.index = pd.to_datetime(closing_options.index, unit = "ms", utc = True).tz_convert("America/New_York")
    Closing_Trade_Price = closing_options["o"].iloc[0]
    
    gross_pnl = Closing_Trade_Price - Open_Trade_Price
    net_pnl = gross_pnl - 0.05
    
    # actual
    
    Actual_Return = featured_dataset[featured_dataset.index == date]["returns"].apply(Binarizer).iloc[0]
    
    #
    
    Pred_Actual = pd.DataFrame([{"timestamp": date, "prediction": Random_Forest_Prediction[0], "probability": probability,
                                   "actual": Actual_Return, "open_price": Open_Trade_Price, "close_price": Closing_Trade_Price, "gross_pnl": gross_pnl,
                                   "net_pnl": net_pnl, "implied_vol": ((Nearest_ITM["put_implied_vol"].iloc[0] + Nearest_ITM["call_implied_vol"].iloc[0])/2),
                                   "gamma_imbalance": gamma_imbalance}])
    
    prediction_actual_list.append(Pred_Actual)
    
    end_time = datetime.now()
    iteration = round((np.where(trade_dates==date)[0][0]/len(trade_dates))*100,2)
    iterations_remaining = len(trade_dates) - np.where(trade_dates==date)[0][0]
    average_time_to_complete = (end_time - start_time).total_seconds()
    estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
    time_remaining = estimated_completion_time - datetime.now()
    
    print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
    
###
    
Prediction_DataFrame = pd.concat(prediction_actual_list).set_index("timestamp")
Prediction_DataFrame["capital"] = 1000 + (Prediction_DataFrame["net_pnl"].cumsum()*100)


plt.figure(dpi=600)
plt.plot(Prediction_DataFrame["capital"])
plt.xticks(rotation=45)
plt.suptitle("Standard")
plt.title("Trades Every Day")
plt.legend(["net_pnl"])
plt.show()

#
confident_prob_threshold = .65
Confident = Prediction_DataFrame[Prediction_DataFrame["probability"] > confident_prob_threshold].copy()
Confident["capital"] = 1000 + (Confident["net_pnl"].cumsum()*100)

plt.figure(dpi=600)
plt.plot(Confident["capital"])
plt.xticks(rotation=45)
plt.suptitle("Confident")
plt.title(f"Trades Only Above Threshold: {confident_prob_threshold}")
plt.legend(["net_pnl"])
plt.show()

#
cheap_threshold = 1.25
Cheap_Option_Strategy = Prediction_DataFrame[Prediction_DataFrame["open_price"] < cheap_threshold].copy()
Cheap_Option_Strategy["net_pnl"] = Cheap_Option_Strategy.apply(lambda row: row['net_pnl'] * 5 if row['open_price'] < cheap_threshold else row['net_pnl'], axis=1)
Cheap_Option_Strategy["capital"] = 1000 + (Cheap_Option_Strategy["net_pnl"].cumsum() *100)

plt.figure(dpi=600)
plt.plot(Cheap_Option_Strategy["capital"])
plt.xticks(rotation=45)
plt.suptitle("Cheap")
plt.title(f"Only trade when price is < ${cheap_threshold}, then buy 5x")
plt.legend(["net_pnl"])
plt.show()

#

overall_win_rate = len(Prediction_DataFrame[Prediction_DataFrame["prediction"] == Prediction_DataFrame["actual"]]) / len(Prediction_DataFrame)
confident_win_rate = len(Confident[(Confident["prediction"] == Confident["actual"])]) / len(Confident)
cheap_win_rate = len(Cheap_Option_Strategy[Cheap_Option_Strategy["prediction"] == Cheap_Option_Strategy["actual"]]) / len(Cheap_Option_Strategy)

print(f"\nOverall Win Rate: {round(overall_win_rate*100, 2)}% {len(Prediction_DataFrame)} Trades PnL = ${Prediction_DataFrame['capital'].iloc[-1]}\nConfident ({confident_prob_threshold}) Win Rate: {round(confident_win_rate*100, 2)}% {len(Confident)} Trades PnL = ${Confident['capital'].iloc[-1]}\nCheap (<${cheap_threshold}) Win Rate: {round(cheap_win_rate*100, 2)}% {len(Cheap_Option_Strategy)} Trades PnL = ${Cheap_Option_Strategy['capital'].iloc[-1]}")
