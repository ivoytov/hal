import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
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
import pyfolio as pf

from util.multiprocess import mp_pandas_obj
from labeling.labeling import *
from sample_weights.attribution import *
from sampling.bootstrapping import *

prefix = 'bloomberg/'
yes_or_no = pd.Series({'Y': True, 'N': False})

moody_scale = {
    'A1': 1,
    'A2': 1,
    'A3': 1,
    'Baa1': 2,
    'Baa2': 2,
    '(P)Baa2': 2,
    'Baa3': 2,
    '(P)Baa3': 2,
    'Ba1': 3,
    'Ba1u': 3,
    '(P)Ba1': 3,
    '(P)Ba2': 3,
    'Ba2': 3,
    'Ba2u': 3,
    '(P)Ba3': 3,
    'Ba3': 3,
    'B1': 4,
    '(P)B1': 4,
    'B2':  5,
    'B2u': 5,
    '(P)B2': 5,
    'B3': 6,
    'Caa1': 7,
    'Caa2' : 7,
    'Caa3': 7,
    'Ca': 7,
    'C': 7,
}

snp_scale = {
    'A+': 1,
    'A': 1,
    'A-': 1,
    'BBB+': 2,
    'BBB': 2,
    'BBB-': 2,
    'BB+': 3,
    'BB+u': 3,
    'BB': 3,
    'BB-': 3,
    '(P)BB-': 3,
    'B+': 4,
    '(P)B+': 4,
    'B': 5,
    'B-': 6,
    'CCC+': 7,
    'CCC': 7,
    'CCC-': 7,
    'CC': 7,
    'C': 7,
    'D': 7,
}

def get_ratings(name: str) -> pd.DataFrame:
    moody = pd.read_csv(prefix + f'{name}/{name}_moody.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('moody')
    snp = pd.read_csv(prefix + f'{name}/{name}_snp.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('snp')
    moody.index.set_levels(pd.PeriodIndex(pd.to_datetime(moody.index.levels[0]), freq='M'), level=0, inplace=True)
    snp.index.set_levels(pd.PeriodIndex(pd.to_datetime(snp.index.levels[0]), freq='M'), level=0, inplace=True)
    moody_num = moody.map(moody_scale).fillna(5)
    snp_num = snp.map(snp_scale).fillna(5)
    out_df = pd.concat([moody_num,snp_num],axis=1)
    out_df['avg_rating'] = out_df.mean(axis=1)
    return out_df


def plotPricesAfterBigMove(prices: pd.DataFrame, trgtPrice: float = None, priceRange: float = 2., bigMove: float = 3.,  numDays: int = 10) -> None:
  out = pd.DataFrame()
  for ticker in prices:
    close = prices[ticker]
    price_filter = (close > trgtPrice - priceRange) & (close < trgtPrice + priceRange) if trgtPrice else 1
    try:
      if bigMove > 0:
        t0 = close[(close.diff(periods=numDays) > bigMove) & price_filter].index[0]
      else:
        t0 = close[(close.diff(periods=numDays) < bigMove) & price_filter].index[0]
    except:
      # loan never met criteria, skip
      continue
    yx = close.iloc[close.index.get_loc(t0) - numDays:close.index.get_loc(t0) + numDays]
    yx.index = range(-numDays,len(yx.index)-numDays)
    out = pd.concat([out, yx], axis=1)

  if len(out):
    p = '{:2.2f}'.format(out.loc[0].median())
    out = out / out.loc[0] - 1
    print("Median price of event trigger", p)
    out = out.rename_axis("Trading Days before/after Event")
    out.plot(kind="line", legend=False, colormap="binary", linewidth=.3, ylabel=f"Price (% Gain from {p})", title="All Tickers")
    plt.show()
    plt.figure()
    out.T.median().plot(linewidth=3, ylabel=f"Price (% Gain from {p})", title="Median")
    plt.show()


def read_prices(security: str='loans') -> pd.DataFrame:
    # read in historical prices
    prices_2020 = pd.read_csv(prefix + security + '/price_data_2020_stub.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2019_2020 = pd.read_csv(prefix + security + '/price_data_2019-2020.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2016_2018 = pd.read_csv(prefix + security + '/price_data_2016-2018.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2013_2015 = pd.read_csv(prefix + security + '/price_data_2013-2015.csv', parse_dates=['Dates'], index_col='Dates')

    prices = pd.concat([prices_2013_2015, prices_2016_2018, prices_2019_2020, prices_2020], join="outer", verify_integrity=True).rename_axis("date")
    print("(# dates, # tickers)", prices.shape)
    return prices.fillna(method="pad")


def getLabels(prices: pd.DataFrame, trgtPrices: Tuple[float, float] = [None, None], priceRange: float = 2., lookbackDays: int = 10, bigMove: float = 3.) -> pd.DataFrame:
  prices_ma = prices.ewm(span=3).mean()
  price_chg = prices_ma.diff(periods=lookbackDays)
  buy = (price_chg.shift(1) > bigMove)
  sell = (price_chg.shift(1) < -bigMove)
  prices = prices.shift(0) # set to 0 to only buy a loan when the purchase price is in trgtPrice; set to 1 to buy the loan when the event trigger was within trgtPrice (but purchase price might be materially different)
  if trgtPrices[0]:
    buy  &= (prices > trgtPrices[0] - priceRange) & (prices < trgtPrices[0] + priceRange)
  if trgtPrices[1]:
    sell &= (prices > trgtPrices[1] - priceRange) & (prices < trgtPrices[1] + priceRange)

  return buy * 1.0 - sell * 1.0

def pricesToBins(labels: pd.DataFrame, prices: pd.DataFrame, ptSl: Tuple[float, float] = [1., 1.], minRet: float = 0.015, holdDays = 10) -> pd.DataFrame:
  out = pd.DataFrame()
  print("Getting bins")
  for ticker in tqdm(labels.columns):
    dates = labels[ticker][labels[ticker] != 0].index
    t1 = getVertBarrier(prices[ticker], dates, holdDays)
    trgt = getDailyVol(prices[ticker])
    events = get_events(prices[ticker], dates, pt_sl=ptSl, target=trgt, min_ret=minRet, num_threads=1, vertical_barrier_times=t1, side_prediction=labels[ticker])
    bins = get_bins(events, prices[ticker])
    bins['ticker'] = ticker   
    out = pd.concat([out, bins])

  return out.set_index('ticker', append=True).swaplevel().rename_axis(['ticker', 'date'])


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_



def avgActiveSignals(all_signals: pd.DataFrame, signalCol: str='signal') -> pd.Series:
  df1 = pd.DataFrame()
  for ticker, signals in all_signals.groupby(level='ticker'):
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.get_level_values('date'))
    tPnts = list(tPnts)
    tPnts.sort()
    out = pd.Series(dtype="float64")
    for loc in tPnts:
      df0 = (signals.index.get_level_values('date') <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
      act = signals[df0].index
      if len(act) > 0:
        out[loc]=signals.loc[act,signalCol].mean()
      else:
        out[loc] = 0 # no signals active at this time
    df1 = pd.concat([df1, out.to_frame(ticker)], axis=1)

  return df1


class DayCounterAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
      X.maturity = (X.maturity - X.index.get_level_values('date')).dt.days
      X.date_issued = (X.index.get_level_values('date') - X.date_issued).dt.days
      return X

def getIndMatrix(barIx, t1):
  # Get indicator matrix
  indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
  for i, (t0, t1) in enumerate(t1.iteritems()):
    indM.loc[t0:t1, i] = 1.
  return indM

def getAvgUniqueness(indM):
  # average uniqueness from indicator matrix
  c = indM.sum(axis=1) # concurrency
  u = indM.div(c, axis=0) # uniqueness
  return u[u>0].mean().mean() # average uniqueness

class MyPipeline(Pipeline):
  def fit(self, X, y, sample_weight=None, **fit_params):
    if sample_weight is not None:
      fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
    return super(MyPipeline, self).fit(X, y, **fit_params)


def trainModel(num_attribs: List[str], cat_attribs: List[str], bool_attribs: List[str], df: pd.DataFrame, test_size: float = 0.3) -> Tuple[any]:
  num_pipeline = Pipeline([
      ('day_counter', DayCounterAdder()),
      ('imputer', SimpleImputer(strategy='median')),
      ('std_scaler', StandardScaler()),
  ])

  bin_pipeline = ColumnTransformer([
          ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
          ("num", num_pipeline, num_attribs),
          ("covi_lite", "passthrough", bool_attribs),
      ])
  
  clf2=RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')

  # sort dataset by event date
  df = df.swaplevel().sort_index().dropna(subset=['bin', 't1', 'ret'])

  X = df.drop(columns=['bin', 'ret', 'clfW', 't1', 'trgt'])
  y = df[['bin', 'ret', 't1']]
  clfW = df.clfW
  print("Getting average uniqueness", len(X.index), y.t1.shape)
  avgU = []
  for ticker in X.index.get_level_values('ticker').unique():
    ind_matrix = get_ind_matrix(y.t1.swaplevel().loc[ticker], X.swaplevel().loc[ticker])
    avgU.append(get_ind_mat_average_uniqueness(ind_matrix))
  avgU = np.mean(avgU)
  rf = MyPipeline([
          ('bin', bin_pipeline),
          ('rf', BaggingClassifier(base_estimator=clf2, n_estimators=1000, max_samples=avgU, max_features=1.)),
  ])

  if test_size:
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, clfW, test_size=test_size, shuffle=False)
  else:
    X_train, y_train, W_train = X, y, clfW

  print(f"Training model with {X_train.shape} samples")
  rf.fit(X_train, y_train.bin, rf__sample_weight=W_train)
  
#  print("Getting feature importances")
#  cat_columns = [item for item in bin_pipeline.named_transformers_['cat'].get_feature_names(cat_attribs)]
#  columns = [*cat_columns, *num_attribs, *bool_attribs,]
#  feature_importances = np.mean([
#    tree.feature_importances_ for tree in rf['rf'].estimators_], axis=0)
#  pd.Series(feature_importances, index=columns).sort_values(ascending=True).plot(kind="barh")
#  plt.show()

  if test_size:
    print(f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, Test Score: {rf.score(X_test, y_test.bin):2.2f}")
    y_pred_train, y_pred_test = rf.predict(X_train), rf.predict(X_test)
    y_score_train, y_score_test = rf.predict_proba(X_train)[:,1], rf.predict_proba(X_test)[:,1]

#    print(cross_val_score(rf, X, y.bin, cv=5, scoring='f1'))
    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test, avgU
  else:
    print(f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, No Test Run")
    return rf


def printCurve(X, y, y_pred, y_score):
  print(f"Precision: {precision_score(y, y_pred):2.2f}, Recall: {recall_score(y, y_pred):2.2f}, Area under curve: {roc_auc_score(y, y_pred):2.2f}")
  print(confusion_matrix(y, y_pred))

  fpr, tpr, thresholds = roc_curve(y, y_score)

  def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

  plot_roc_curve(fpr, tpr)
  plt.show()


def discreteSignal(signal0: pd.DataFrame, stepSize: float) -> pd.DataFrame:
  signal1= ( signal0 / stepSize).round() * stepSize
  signal1[signal1 > 1] = 1 # cap
  signal1[signal1 < -1] = -1 #floor
  return signal1


def getSignal(events, stepSize, prob, pred, numClasses):
  # get signals from predictions
  if prob.shape[0] == 0:
    return pd.Series(dtype='float64')
  #1) generate signals from multinomial cassification
  signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** .5 #t-value of OvR
  signal0 = pred * (2* norm.cdf(signal0) - 1) # signal=side*size
  if 'side' in events:
    signal0 *= events.loc[signal0.index, 'side'] # meta labeling
  #2) compute average signal among those open
  df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
  df0 = avgActiveSignals(df0)
  signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
  return signal1

def backtest(y_test, y_score_test, y_pred_test, df, prices, stepSize: float = 0.05) -> pd.DataFrame:
  idx = y_test.shape[0]
  events = df.swaplevel().sort_index()[['side', 't1']].iloc[-idx:]
  signals = getSignal(events, stepSize=stepSize, prob=pd.Series(y_score_test, index=events.index), pred=pd.Series(y_pred_test, index=events.index), numClasses=2)
  price_idx = prices.index.searchsorted(events.index[0][0])
  positions = signals.reindex(index=prices.index[price_idx:]).fillna(method='pad').fillna(0)

  fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True)
  numTrades = positions.abs().sum(axis=1)
  numTrades.plot(ax=ax[0], title="# of Positions")
  #positions = positions.divide(numTrades, axis=0)
  positions.abs().sum(axis=1).plot(ax=ax[1], title="Sum of Gross Positions after Weighting")
  #positions.sum(axis=1).plot(ax=ax[1])
  print("Number of trading days with a position", positions[positions.sum(axis=1) != 0].shape[0])

  breakpoint()
  portfolio_rtn_df = positions * prices.pct_change().iloc[price_idx:]
  portfolio_rtn = portfolio_rtn_df.sum(axis=1)/positions.sum(axis=1)
  portfolio_cum_rtn_df = (1 + portfolio_rtn).cumprod() - 1
  portfolio_cum_rtn_df.plot(ax=ax[2], title="Portfolio PnL, %")
  plt.tight_layout()
  plt.show()

  pf.create_returns_tear_sheet(portfolio_rtn)
  return signals, positions
