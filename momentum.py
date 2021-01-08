import numpy as np
import pandas as pd
import datetime
import sys
from typing import Tuple, List, Dict
from tqdm import tqdm
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pyfolio as pf
import QuantLib as ql

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

class YieldAdder(BaseEstimator, TransformerMixin):
    def __init__(self, security: str = 'bonds'): # no *args or **kwargs
        self.security = security

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _get_ql_dates(start: pd.Timestamp, maturity: pd.Timestamp, settlement: pd.Timestamp) -> Tuple[ql.Date]:
        start = ql.Date(start.day, start.month, start.year)
        maturity = ql.Date(maturity.day, maturity.month, maturity.year)
        settlement = ql.Date(settlement.day, settlement.month, settlement.year)
        return start, maturity, settlement


    @staticmethod
    def get_loan_ytm(row):
        start, maturity, settlement = YieldAdder._get_ql_dates(row['date_issued'], row['maturity'], row.name[0])
        schedule = ql.MakeSchedule(start, maturity, ql.Period('6M'))
        interest = ql.FixedRateLeg(schedule, ql.Actual360(), [100.], [ row['cpn'] / 10000 + .02]) #FIXME: hardcorded LIBOR at 2%
        bond = ql.Bond(0, ql.TARGET(), start, interest)

        return bond.bondYield(row['close'], ql.Actual360(), ql.Compounded, ql.Semiannual, settlement)
        
    @staticmethod
    def get_bond_ytm(row):
        start, maturity, settlement = YieldAdder._get_ql_dates(row['date_issued'], row['maturity'], row.name[0])
        schedule = ql.MakeSchedule(start, maturity, ql.Period('6M'))
        interest = ql.FixedRateLeg(schedule, ql.Actual360(), [100.], [ row['cpn'] / 100 ])
        bond = ql.Bond(0, ql.TARGET(), start, interest)

        return bond.bondYield(row['close'], ql.Actual360(), ql.Compounded, ql.Semiannual, settlement)
        
    def transform(self, X, y=None):
        X['ytm'] = X.apply(self.get_bond_ytm if self.security == 'bonds' else self.get_loan_ytm, axis=1)
        return X


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
  security = 'loans' if 'covi_lite' in bool_attribs else 'bonds'
  num_pipeline = Pipeline([
      ('ytm', YieldAdder(security=security)),
      ('day_counter', DayCounterAdder()),
      ('imputer', SimpleImputer(strategy='median')),
      ('std_scaler', StandardScaler()),
  ])

  bin_pipeline = ColumnTransformer([
          ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
          ("num", num_pipeline, num_attribs),
          ("bool", "passthrough", bool_attribs),
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
  
  print("Getting feature importances")
  cat_columns = [item for item in bin_pipeline.named_transformers_['cat'].get_feature_names(cat_attribs)]
  columns = [*cat_columns, *num_attribs, 'ytm', *bool_attribs,]
  feature_importances = np.mean([
    tree.feature_importances_ for tree in rf['rf'].estimators_], axis=0)
  pd.Series(feature_importances, index=columns).sort_values(ascending=True).plot(kind="barh")
  plt.show()

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

  portfolio_rtn_df = positions * prices.pct_change().iloc[price_idx:]
  portfolio_rtn = portfolio_rtn_df.sum(axis=1)/positions.abs().sum(axis=1)
  portfolio_cum_rtn_df = (1 + portfolio_rtn).cumprod() - 1
  portfolio_cum_rtn_df.plot(ax=ax[2], title="Portfolio PnL, %")
  plt.tight_layout()
  plt.show()

  fig = pf.create_returns_tear_sheet(portfolio_rtn, return_fig=True)
  print("Saving backtest as backtest.png")
  fig.savefig('backtest.png')
  return signals, positions
def get_prices(security: str) -> pd.DataFrame:
    file_list = {
            'bonds': [ '2015_2016', '2017_2018', '2019_2020', '2021',],
            'loans': [ '2013-2015', '2016-2018', '2019-2020', '2020_stub'],
            }
    # read in historical prices
    df_list = [pd.read_csv(prefix + f'{security}/price_data_{year}.csv', parse_dates=['Dates'], index_col='Dates') for year in file_list[security]]

    prices = pd.concat(df_list, join="outer", verify_integrity=True).rename_axis("date")
    return prices.fillna(method="pad")

def get_desc(security: str) -> pd.DataFrame:
    desc = pd.read_csv(prefix + f'{security}/{security}_desc.csv', parse_dates=['date_issued', 'maturity', ], index_col='id').rename_axis('ticker')
    desc = desc.rename(columns={'ticker': 'name'})
    # some basic data cleaning steps to make the data ready for the pipeline
    #desc = desc[['name', 'cpn', 'date_issued', 'maturity', 'amt_out', 'convertible', 'industry_sector']] # 'industry_sector', 'industry_subgroup' ]]
    if security == 'bonds':
        desc.convertible = desc.convertible.map(yes_or_no).astype(bool)
    elif security == 'loans':
        desc.covi_lite = desc.covi_lite.map(yes_or_no).astype(bool)

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

    # drop convertible bonds (our data doesn't include underlying stock prices)
    if 'convertible' in df.columns:
        df.drop(df[df.convertible].index, inplace=True)

    df.drop(columns=['month'], inplace=True)
    return df.dropna()


def train_and_backtest(df: pd.DataFrame, prices: pd.DataFrame, num_attribs: List[str], cat_attribs: List[str], bool_attribs: List[str]) -> None:
    print("Training bond model")
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_score_train, y_score_test,avgU = trainModel(num_attribs, cat_attribs, bool_attribs, df.copy())
    printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
    printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
    signals, positions = backtest(y_test, y_score_test, y_pred_test, df, prices, stepSize=.05)


def train_production(df: pd.DataFrame, num_attribs: List[str], cat_attribs: List[str], bool_attribs: List[str], t_params: Dict[str, any]) -> None:
    rf = trainModel(num_attribs, cat_attribs, bool_attribs, df, test_size=0)
    target = df['trgt']
    current_events = df.swaplevel().sort_index().loc['2021-01-06'].drop(columns=['bin', 'ret', 'clfW', 't1', 'trgt'])

    if not len(current_events):
        print("Hal has no profitable trades to suggest, try again tomorrow")
        return

    pred_now = rf.predict(current_events)
    score_now = rf.predict_proba(current_events)[:,1]



    data = {"pred": pred_now, "side": current_events.side.values, "close": current_events.close.values, "trgt": target.swaplevel().loc[current_events.index].values}

    trades = pd.DataFrame(data, index=current_events.index)
    trades = trades.join(current_events[['name', 'cpn', 'maturity']], how='left')
    trades['t1'] = pd.NaT
    signals = getSignal(trades, stepSize=0.05, prob=pd.Series(score_now, index=trades.index), pred=pd.Series(pred_now, index=trades.index), numClasses=2)
    trades['signal'] = signals.unstack().swaplevel()
    trades['profit_take'] = trades.close * (1 + trades.trgt * trades.side * t_params['ptSl'][0])
    trades['stop_loss'] = trades.close * (1 - trades.trgt * trades.side * t_params['ptSl'][1])
    for index, trade in trades[trades.pred == 1].loc['2021-01-06'].iterrows():
        print(trade)
        

def main():
    if 'bonds' in sys.argv:
        sec_type = 'bonds'
    elif 'loans' in sys.argv:
        sec_type = 'loans'
    else:
        print("Pass bonds or loans as arguments")
        return

    t_params = {
            'bonds': {
                'holdDays':     15,
                'bigMove':      3,
                'lookbackDays': 10,
                'targetPrices': [None, None],
                'priceRange':   5,
                'ptSl':         [1, 3],
                'minRet':       0.01,
            },
            'loans': {
                'holdDays':     15,
                'bigMove':      3,
                'lookbackDays': 10,
                'targetPrices': [None, 999],
                'priceRange':   5,
                'ptSl':         [1, 3],
                'minRet':       0.01,
            },
    }

    prices = get_prices(sec_type)
    desc = get_desc(sec_type)
    ratings_df = get_ratings(sec_type)

    df = data_pipeline(prices, desc, ratings_df, t_params[sec_type])

    num_attribs = ['cpn', 'date_issued', 'maturity', 'amt_out', 'close', 'avg_rating']
    cat_attribs = ['side', 'moody']
    if sec_type == 'loans': 
        cat_attribs += ['loan_type']
    elif sec_type == 'bonds':
        cat_attribs += ['industry_sector']

    bool_attribs = {
            'bonds': ['convertible'],
            'loans': ['covi_lite'],
            }

    if 'prod' in sys.argv:
        train_production(df, num_attribs, cat_attribs, bool_attribs[sec_type], t_params)
    elif 'test' in sys.argv:
        train_and_backtest(df, prices, num_attribs, cat_attribs, bool_attribs[sec_type])
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
