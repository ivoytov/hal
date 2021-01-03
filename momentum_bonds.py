import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple, List

from util.multiprocess import mp_pandas_obj
from labeling.labeling import *
from sample_weights.attribution import *
from util.helper import *

HOLD_DAYS = 12
BIG_MOVE = 2
LOOKBACK_DAYS = 10
TARGET_PRICES = [90, 999]  # Disable shorting of loans by setting target price to a number >100
PRICE_RANGE = 5
PT_SL = [1, 3]
MIN_RET = 0.005

# read in historical prices
prices_2019_2020 = pd.read_csv(prefix + 'bonds/price_data_2019_2020.csv', parse_dates=['Dates'], index_col='Dates')
prices_2017_2018 = pd.read_csv(prefix + 'bonds/price_data_2017_2018.csv', parse_dates=['Dates'], index_col='Dates')
prices_2015_2016 = pd.read_csv(prefix + 'bonds/price_data_2015_2016.csv', parse_dates=['Dates'], index_col='Dates')

prices = pd.concat([prices_2015_2016, prices_2017_2018, prices_2019_2020], join="outer", verify_integrity=True).rename_axis("date")
print("(# dates, # tickers)", prices.shape)
prices = prices.fillna(method="pad")

TARGET_PRICES = [95, 95]
PRICE_RANGE = 10

plotPricesAfterBigMove(prices, trgtPrice=TARGET_PRICES[0], priceRange=PRICE_RANGE, bigMove=BIG_MOVE, numDays=LOOKBACK_DAYS)

plotPricesAfterBigMove(prices, trgtPrice=TARGET_PRICES[1], priceRange=PRICE_RANGE, bigMove=-BIG_MOVE, numDays=LOOKBACK_DAYS)

labels = getLabels(prices, trgtPrices=TARGET_PRICES, priceRange=PRICE_RANGE, lookbackDays=HOLD_DAYS, bigMove=BIG_MOVE)

bins = pricesToBins(labels, prices, ptSl=PT_SL, minRet=MIN_RET, holdDays=HOLD_DAYS)
print(bins)
print(pd.pivot_table(bins, index=['side'], columns=['bin'], values='ret', aggfunc=[len]))

print(pd.pivot_table(bins, index=['side'], values='ret', aggfunc=[len, np.mean,  percentile(25), 
                                                            percentile(50), percentile(75), 
                                                            max, min, np.sum]))

bins[(bins.side == 1) & (bins.ret <.1) & (bins.ret >-.1)].ret.plot.hist(bins=20)
plt.show()
bins[(bins.side == -1) & (bins.ret <.1) & (bins.ret >-.1)].ret.plot.hist(bins=20)
plt.show()

clfW = getWeightColumn(bins, prices)
bins = pd.concat([bins, clfW], axis=1).rename_axis(['ticker', 'date'])
print(bins)

desc = pd.read_csv(prefix + 'bonds/bonds_desc.csv', parse_dates=['date_issued', 'maturity', ], index_col='id').rename_axis('ticker')
desc = desc.rename(columns={'ticker': 'name'})
# some basic data cleaning steps to make the data ready for the pipeline
desc = desc[['name', 'cpn', 'date_issued', 'maturity', 'amt_out', 'convertible', 'industry_group']]
desc.convertible = desc.convertible.map(yes_or_no).astype(bool)
desc.cpn = pd.to_numeric(desc.cpn, errors='coerce')
desc.maturity = pd.to_datetime(desc.maturity, errors='coerce')
desc.date_issued = pd.to_datetime(desc.date_issued, errors='coerce')
print(desc)

df = bins.join(desc).drop(columns='name')
df = df.join(prices.rename_axis("ticker", axis='columns').unstack().rename('close'))
print(df)

df.swaplevel().sort_index()

num_attribs = ['cpn', 'date_issued', 'maturity', 'amt_out', 'close', 'avg_rating']
cat_attribs = ['side', 'industry_group', 'moody']

moody = pd.read_csv(prefix + 'bonds/bonds_moody.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('moody')
snp = pd.read_csv(prefix + 'bonds/bonds_snp.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('snp')
moody.index.set_levels(pd.PeriodIndex(pd.to_datetime(moody.index.levels[0]), freq='M'), level=0, inplace=True)
snp.index.set_levels(pd.PeriodIndex(pd.to_datetime(snp.index.levels[0]), freq='M'), level=0, inplace=True)
print(moody.unique(), snp.unique(), sep='\n')

moody_num = moody.map(moody_scale).fillna(5)
snp_num = snp.map(snp_scale).fillna(5)
avg_rating = pd.concat([moody_num,snp_num],axis=1).mean(axis=1).rename('avg_rating')

df_rating = df.reset_index(level='ticker')
df_rating['month'] = pd.PeriodIndex(df_rating.index, freq='M') - 1
df_rating = df_rating.set_index('ticker', append=True).swaplevel()
df_rating = df_rating.join(avg_rating, on=['month', 'ticker'])
df_rating = df_rating.join(moody_num, on=['month', 'ticker'])
print(df_rating)

print("Training bond model")
X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test,avgU = trainModel(num_attribs, cat_attribs, ['convertible'], df_rating)
printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
signals, positions = backtest(y_test, y_score_test, y_pred_test, df, prices, stepSize=.05)

rf = trainModel(num_attribs, cat_attribs, ['convertible'], df_rating, test_size=0)
current_events = df_rating.swaplevel().sort_index().loc['2021-01-01'].drop(columns=['bin', 'ret', 'clfW', 't1'])
pred_now = rf.predict(current_events)
score_now = rf.predict_proba(current_events)[:,1]
data = {"pred": pred_now, "score": score_now, "side": current_events.side.values, "close": current_events.close.values}
trades = pd.DataFrame(data, index=current_events.index)
trades = trades.join(desc)
for date, ticker in trades.index:
  trgt = getDailyVol(prices[ticker])
  trades.loc[(date, ticker), 'trgt'] = trgt.loc[date]
trades['t1'] = pd.NaT
print(current_events)
print(trades)

signals = getSignal(trades, stepSize=0.05, prob=pd.Series(score_now, index=trades.index), pred=pd.Series(pred_now, index=trades.index), numClasses=2)
trades['signal'] = signals.unstack().swaplevel()
numTrades = signals.abs().sum(axis=1)
positions = signals.divide(numTrades, axis=0)
trades['position'] = positions.unstack().swaplevel()
print(trades)
