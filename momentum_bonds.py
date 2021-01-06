import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from typing import Tuple, List, Dict

from labeling.labeling import *
from sample_weights.attribution import *
from util.helper import *

def get_bond_prices() -> pd.DataFrame:
    # read in historical prices
    prices_2021 = pd.read_csv(prefix + 'bonds/price_data_2021.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2019_2020 = pd.read_csv(prefix + 'bonds/price_data_2019_2020.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2017_2018 = pd.read_csv(prefix + 'bonds/price_data_2017_2018.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2015_2016 = pd.read_csv(prefix + 'bonds/price_data_2015_2016.csv', parse_dates=['Dates'], index_col='Dates')

    prices = pd.concat([prices_2015_2016, prices_2017_2018, prices_2019_2020, prices_2021], join="outer", verify_integrity=True).rename_axis("date")
    return prices.fillna(method="pad")

def get_bond_desc() -> pd.DataFrame:
    desc = pd.read_csv(prefix + 'bonds/bonds_desc.csv', parse_dates=['date_issued', 'maturity', ], index_col='id').rename_axis('ticker')
    desc = desc.rename(columns={'ticker': 'name'})
    # some basic data cleaning steps to make the data ready for the pipeline
    desc = desc[['name', 'cpn', 'date_issued', 'maturity', 'amt_out', 'convertible', 'industry_group',]] # 'industry_sector', 'industry_subgroup' ]]
    desc.convertible = desc.convertible.map(yes_or_no).astype(bool)
    desc.cpn = pd.to_numeric(desc.cpn, errors='coerce')
    desc.maturity = pd.to_datetime(desc.maturity, errors='coerce')
    desc.date_issued = pd.to_datetime(desc.date_issued, errors='coerce')
    return desc

def data_pipeline(prices: pd.DataFrame, desc: pd.DataFrame, ratings_df: pd.DataFrame, t_params: Dict[str, any]) -> pd.DataFrame:
    labels = getLabels(prices, trgtPrices=t_params['targetPrices'], priceRange=t_params['priceRange'], lookbackDays=t_params['lookbackDays'], bigMove=t_params['bigMove'])
    bins = pricesToBins(labels, prices, ptSl=t_params['ptSl'], minRet=t_params['minRet'], holdDays=t_params['holdDays'])
    clfW = getWeightColumn(bins, prices)
    bins = pd.concat([bins, clfW], axis=1).rename_axis(['ticker', 'date'])
    df = bins.join(desc)
    df = df.join(prices.rename_axis("ticker", axis='columns').unstack().rename('close'))


    df = df.reset_index(level='ticker')
    df['month'] = pd.PeriodIndex(df.index, freq='M') - 1
    df = df.set_index('ticker', append=True).swaplevel()
    df = df.join(ratings_df[['moody', 'avg_rating']], on=['month', 'ticker'])
    df.drop(df[df.convertible].index, inplace=True)
    df.drop(columns=['month'], inplace=True)
    return df


def train_and_backtest(df: pd.DataFrame, prices: pd.DataFrame, num_attribs: List[str], cat_attribs: List[str], bool_attribs: List[str]) -> None:
    print("Training bond model")
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test,avgU = trainModel(num_attribs, cat_attribs, bool_attribs, df.copy())
    printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
    printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
    signals, positions = backtest(y_test, y_score_test, y_pred_test, df, prices, stepSize=.05)


def train_production(df: pd.DataFrame, num_attribs: List[str], cat_attribs: List[str], bool_attribs: List[str]) -> None:
    rf = trainModel(num_attribs, cat_attribs, bool_attribs, df, test_size=0)
    target = df['trgt']
    current_events = df.swaplevel().sort_index().loc['2021-01-06'].drop(columns=['bin', 'ret', 'clfW', 't1', 'trgt'])

    pred_now = rf.predict(current_events)
    score_now = rf.predict_proba(current_events)[:,1]

    data = {"pred": pred_now, "side": current_events.side.values, "close": current_events.close.values, "trgt": target.swaplevel().loc[current_events.index].values}

    trades = pd.DataFrame(data, index=current_events.index)
    breakpoint()
    trades = trades.join(current_events[['name', 'cpn', 'maturity']], how='left')
    trades['t1'] = pd.NaT
    signals = getSignal(trades, stepSize=0.05, prob=pd.Series(score_now, index=trades.index), pred=pd.Series(pred_now, index=trades.index), numClasses=2)
    numTrades = signals.abs().sum(axis=1)
    positions = signals.divide(numTrades, axis=0)
    trades['position'] = positions.unstack().swaplevel()
    print(trades[trades.pred == 1].loc['2021-01-06'])

def main():
    t_params = {
            'holdDays':     10,
            'bigMove':      3,
            'lookbackDays': 10,
            'targetPrices': [None, 999],
            'priceRange':   5,
            'ptSl':         [0.5, 3],
            'minRet':       0.005,
    }

    prices = get_bond_prices()
    desc = get_bond_desc()
    ratings_df = get_ratings('bonds')

    df = data_pipeline(prices, desc, ratings_df, t_params)

    num_attribs = ['cpn', 'date_issued', 'maturity', 'amt_out', 'close', 'avg_rating']
    cat_attribs = ['side', 'moody', 'industry_group' ]
    bool_attribs = ['convertible']



    if 'prod' in sys.argv:
        train_production(df, num_attribs, cat_attribs, bool_attribs)
    elif 'test' in sys.argv:
        train_and_backtest(df, prices, num_attribs, cat_attribs, bool_attribs)
    else:
        print("prod: trains the model on all data")
        print("test: runs a backtest")

    #try:
    #    train_production(df, num_attribs, cat_attribs, bool_attribs)
    #except Exception as e:
    #    import pdb
    #    pdb.set_trace()
    #    raise(e)
    #

if __name__ == "__main__":
    main()
