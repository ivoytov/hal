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

prefix = '~/bloomberg/'

def read_prices(security: str='loans') -> pd.DataFrame:
    # read in historical prices
    prices_2020 = pd.read_csv(prefix + security + '/price_data_2020_stub.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2019_2020 = pd.read_csv(prefix + security + '/price_data_2019-2020.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2016_2018 = pd.read_csv(prefix + security + '/price_data_2016-2018.csv', parse_dates=['Dates'], index_col='Dates')
    prices_2013_2015 = pd.read_csv(prefix + security + '/price_data_2013-2015.csv', parse_dates=['Dates'], index_col='Dates')

    prices = pd.concat([prices_2013_2015, prices_2016_2018, prices_2019_2020, prices_2020], join="outer", verify_integrity=True).rename_axis("date")
    print("(# dates, # tickers)", prices.shape)
    return prices.fillna(method="pad")

prices = read_prices('loans')

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

HOLD_DAYS = 12
BIG_MOVE = 2
LOOKBACK_DAYS = 10
TARGET_PRICES = [90, 999]  # Disable shorting of loans by setting target price to a number >100
PRICE_RANGE = 5
PT_SL = [1, 3]
MIN_RET = 0.005

plotPricesAfterBigMove(prices, trgtPrice=TARGET_PRICES[0], priceRange=PRICE_RANGE, bigMove=BIG_MOVE, numDays=LOOKBACK_DAYS)
plotPricesAfterBigMove(prices, trgtPrice=TARGET_PRICES[1], priceRange=PRICE_RANGE, bigMove=-BIG_MOVE, numDays=LOOKBACK_DAYS)

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

labels = getLabels(prices, trgtPrices=TARGET_PRICES, priceRange=PRICE_RANGE, lookbackDays=HOLD_DAYS, bigMove=BIG_MOVE)
labels.sum(axis=1).plot(ylabel="# of net buy/(sell) labels generated per day")
plt.show()
plt.figure()
print("Average trades per day: ", labels.abs().sum(axis=1).mean(), "median", labels.abs().sum(axis=1).median())
labels.abs().sum(axis=1).plot(ylabel="# of gross buy/(sell) labels generated per day", logy=True)
plt.show()

def pricesToBins(labels: pd.DataFrame, prices: pd.DataFrame, ptSl: Tuple[float, float] = [1., 1.], minRet: float = 0.015, holdDays = 10) -> pd.DataFrame:
  out = pd.DataFrame()
  for ticker in labels.columns:
    dates = labels[ticker][labels[ticker] != 0].index
    t1 = getVertBarrier(prices[ticker], dates, holdDays)
    trgt = getDailyVol(prices[ticker])
    events = get_events(prices[ticker], dates, pt_sl=ptSl, target=trgt, min_ret=minRet, num_threads=1, vertical_barrier_times=t1, side_prediction=labels[ticker])
    bins = get_bins(events, prices[ticker])
    bins['ticker'] = ticker   
    out = pd.concat([out, bins])

  return out.set_index('ticker', append=True).swaplevel().rename_axis(['ticker', 'date'])

bins = pricesToBins(labels, prices, ptSl=PT_SL, minRet=MIN_RET, holdDays=HOLD_DAYS)
print(bins.head())


print(pd.pivot_table(bins, index=['side'], columns=['bin'], values='ret', aggfunc=[len]))

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

pd.pivot_table(bins, index=['side'], values='ret', aggfunc=[len, np.mean,  percentile(25),
                                                            percentile(50), percentile(75),
                                                            max, min, np.sum])

bins[(bins.side == 1) & (bins.ret <.1) & (bins.ret >-.1)].ret.plot.hist(bins=20)
plt.show()
bins[(bins.side == -1) & (bins.ret <.1) & (bins.ret >-.1)].ret.plot.hist(bins=20)
plt.show()
  
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

signals = avgActiveSignals(bins[['t1', 'side']], signalCol='side')

clfW = getWeightColumn(bins, prices)
bins = pd.concat([bins, clfW], axis=1).rename_axis(['ticker', 'date'])
print(bins.head())

desc = pd.read_csv(prefix + 'loans/loans_desc.csv', parse_dates=['date_issued', 'maturity', ], index_col='id').rename_axis('ticker')
desc = desc.rename(columns={'ticker': 'name'})

# some basic data cleaning steps to make the data ready for the pipeline
desc = desc[['name', 'cpn', 'date_issued', 'maturity', 'amt_out', 'covi_lite', 'loan_type']]
desc.cpn = pd.to_numeric(desc.cpn, errors='coerce')
yes_or_no = pd.Series({'Y': True, 'N': False})
desc.covi_lite = desc.covi_lite.map(yes_or_no).astype(bool)
desc.maturity = pd.to_datetime(desc.maturity, errors='coerce')
desc.date_issued = pd.to_datetime(desc.date_issued, errors='coerce')
desc

df = bins.join(desc).drop(columns='name')
df = df.join(prices.rename_axis("ticker", axis='columns').unstack().rename('close'))
df

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
    t1 = (t1, t0[1])
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

  X = df.drop(columns=['bin', 'ret', 'clfW', 't1'])
  y = df[['bin', 'ret', 't1']]
  clfW = df.clfW
  avgU = getAvgUniqueness(getIndMatrix(X.index, y.t1))
  rf = MyPipeline([
          ('bin', bin_pipeline),
          ('rf', BaggingClassifier(base_estimator=clf2, n_estimators=1000, max_samples=avgU, max_features=1.)),
  ])

  if test_size:
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, clfW, test_size=test_size, shuffle=False)
  else:
    X_train, y_train, W_train = X, y, clfW

  print("Training model with {X_train.shape} samples")
  rf.fit(X_train, y_train.bin, rf__sample_weight=W_train)
  
  cat_columns = [item for item in bin_pipeline.named_transformers_['cat'].get_feature_names(cat_attribs)]
  columns = [*cat_columns, *num_attribs, *bool_attribs,]
  feature_importances = np.mean([
    tree.feature_importances_ for tree in rf['rf'].estimators_], axis=0)
  pd.Series(feature_importances, index=columns).sort_values(ascending=True).plot(kind="barh")
  plt.show()

  if test_size:
    print(f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, Test Score: {rf.score(X_test, y_test.bin):2.2f}")
    y_pred_train, y_pred_test = rf.predict(X_train), rf.predict(X_test)
    y_score_train, y_score_test = rf.predict_proba(X_train)[:,1], rf.predict_proba(X_test)[:,1]

    print(cross_val_score(rf, X, y.bin, cv=5, scoring='f1'))
    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test, avgU
  else:
    print(f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, No Test Run")
    return rf

num_attribs = ['cpn', 'date_issued', 'maturity', 'amt_out', 'close']
cat_attribs = ['side', 'loan_type']


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
  positions = positions.divide(numTrades, axis=0)
  positions.abs().sum(axis=1).plot(ax=ax[1], title="Sum of Gross Positions after Weighting")
  #positions.sum(axis=1).plot(ax=ax[1])
  print("Number of trading days with a position", positions[positions.sum(axis=1) != 0].shape[0])

  portfolio_rtn_df = positions.multiply(prices.pct_change().fillna(0).iloc[price_idx:]).sum(axis=1)
  portfolio_cum_rtn_df = (1 + portfolio_rtn_df).cumprod() - 1
  portfolio_cum_rtn_df.plot(ax=ax[2], title="Portfolio PnL, %")
  plt.tight_layout()
  plt.show()
  return signals, positions



moody = pd.read_csv(prefix + 'loans/loans_moody.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('moody')
snp = pd.read_csv(prefix + 'loans/loans_snp.csv', index_col='id').rename_axis('ticker').rename_axis('month',axis=1).unstack().rename('snp')
moody.index.set_levels(pd.PeriodIndex(pd.to_datetime(moody.index.levels[0]), freq='M'), level=0, inplace=True)
snp.index.set_levels(pd.PeriodIndex(pd.to_datetime(snp.index.levels[0]), freq='M'), level=0, inplace=True)
print(moody.unique(), snp.unique(), sep='\n')

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

moody_num = moody.map(moody_scale).fillna(5)
snp_num = snp.map(snp_scale).fillna(5)
avg_rating = pd.concat([moody_num,snp_num],axis=1).mean(axis=1).rename('avg_rating')

df_rating = df.reset_index(level='ticker')
df_rating['month'] = pd.PeriodIndex(df_rating.index, freq='M') - 1
df_rating = df_rating.set_index('ticker', append=True).swaplevel()
df_rating = df_rating.join(avg_rating, on=['month', 'ticker'])
df_rating = df_rating.join(moody_num, on=['month', 'ticker'])
print(df_rating)

num_attribs += ['avg_rating']
cat_attribs += ['moody']

X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test, avgU = trainModel(num_attribs, cat_attribs, ['covi_lite'], df_rating)

printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
printCurve(X_test, y_test.bin, y_pred_test, y_score_test)

signals, positions = backtest(y_test, y_score_test, y_pred_test, df, prices)

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
