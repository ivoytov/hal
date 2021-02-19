import numpy as np
import multiprocessing as mp
import pandas as pd
from multiprocessing import Pool
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=1, progress_bar=True)
import datetime
import sys
import os
from typing import Tuple, List, Dict
from tqdm import tqdm
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pyfolio as pf
import QuantLib as ql
from ib_insync import *
from pylab import plt, mpl
import requests
import xml.etree.ElementTree as ET
import joblib

from util.multiprocess import mp_pandas_obj
from util.utils import get_daily_vol
from labeling.labeling import *
from sample_weights.attribution import *
from sampling.bootstrapping import *
from cross_validation.cross_validation import *
from ensemble.sb_bagging import *
from bet_sizing.bet_sizing import *
from data_structures.run_data_structures import *

plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"

prefix = "bloomberg/"
yes_or_no = pd.Series({"Y": True, "N": False})

moody_scale = {
    "A1": 1,
    "A2": 1,
    "A3": 1,
    "Baa1": 2,
    "Baa2": 2,
    "(P)Baa2": 2,
    "Baa3": 2,
    "(P)Baa3": 2,
    "Ba1": 3,
    "Ba1u": 3,
    "(P)Ba1": 3,
    "(P)Ba2": 3,
    "Ba2": 3,
    "Ba2u": 3,
    "(P)Ba3": 3,
    "Ba3": 3,
    "B1": 4,
    "(P)B1": 4,
    "B2": 5,
    "B2u": 5,
    "(P)B2": 5,
    "B3": 6,
    "Caa1": 7,
    "Caa2": 7,
    "Caa3": 7,
    "Ca": 7,
    "C": 7,
}

snp_scale = {
    "A+": 1,
    "A": 1,
    "A-": 1,
    "BBB+": 2,
    "BBB": 2,
    "BBB-": 2,
    "BB+": 3,
    "BB+u": 3,
    "BB": 3,
    "BB-": 3,
    "(P)BB-": 3,
    "B+": 4,
    "(P)B+": 4,
    "B": 5,
    "B-": 6,
    "CCC+": 7,
    "CCC": 7,
    "CCC-": 7,
    "CC": 7,
    "C": 7,
    "D": 7,
}


def get_day_ticks(contract: Bond, day: pd.Timestamp, ib: IB):
    dt = day  # store original day for checking at the end
    ticksList = []
    while True:
        ticks = ib.reqHistoricalTicks(
            contract,
            startDateTime=dt,
            endDateTime="",
            whatToShow="Trades",
            numberOfTicks=1000,
            useRth=False,
        )
        if not ticks:
            break
        dt = ticks[-1].time
        if len(ticksList) >= 1:
            if dt == ticksList[-1][-1].time:
                break

        ticksList.append(ticks)
        if len(ticks) < 1000:
            break

    if not len(ticks):
        return pd.DataFrame()

    allTicks = [t for ticks in reversed(ticksList) for t in ticks]
    allTicks = pd.DataFrame(
        allTicks,
        columns=[
            "date_time",
            "tickAttribLast",
            "price",
            "volume",
            "exchange",
            "specialCon",
        ],
    )
    attribs = allTicks.apply(
        lambda x: x.tickAttribLast.dict(), axis=1, result_type="expand"
    )
    allTicks = allTicks.join(attribs).drop(columns="tickAttribLast")
    allTicks["date_time"] = pd.to_datetime(allTicks.date_time)
    allTicks = allTicks.set_index("date_time")
    return allTicks[allTicks.index.date == day.date()]


def get_bond_data(row: dict, end: pd.Timestamp, ib: IB):
    print("in get_bond_data", row)
    contract = Bond(symbol=row.ISIN, exchange="SMART", currency="USD")
    period = pd.bdate_range(start=row.head_timestamp, end=end)
    out = pd.DataFrame()
    if len(period) != 0:
        # bdate_range resets all HH:MM to 00:00. We change the first value back to `start` with exact datetime 
        period = period[1:].union([row.head_timestamp])

    for day in period:
        out = out.append(get_day_ticks(contract, day, ib))

    out['ticker'] = row.name
    out['ISIN'] = row.ISIN
    return out.set_index("ticker", append=True).swaplevel()


def fetch_treasury_data() -> None:
    url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/XmlView.aspx?data=yield"
    r = requests.get(url).text
    tree = ET.fromstring(r)
    dates = []
    rates = []
    ns = {"meta": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"}
    for entry in tree.findall(".//meta:properties", ns):
        dates.append(pd.Timestamp(entry.find('*[@meta:type="Edm.DateTime"]', ns).text))
        values = entry.findall('*[@meta:type="Edm.Double"]', ns)
        rates.append({pt.tag.split("BC_")[1]: float(pt.text) for pt in values})
    out = pd.DataFrame(rates, index=dates)
    out.drop(columns=[*out.columns[0:4], out.columns[-1]], inplace=True)
    out.columns = [1, 2, 3, 5, 7, 10, 20, 30]
    return out


def fetch_head_timestamps(store: pd.io.pytables.HDFStore, ib):
    desc = store.get("desc")
    tickers = desc[(desc.ISIN != "") & (pd.isnull(desc.head_timestamp))].index

    def get_head(ISIN):
        ib.sleep(2)
        contract = Bond(symbol=ISIN, exchange="SMART", currency="USD")
        return ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=False)

    return desc.ISIN.loc[tickers].apply(get_head)

def fetch_price_data(
    store: pd.io.pytables.HDFStore,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ib: IB
) -> None:
    tickers = store.select("desc", where=f'head_timestamp > pd.Timestamp("{start}")')[
        ["ISIN", "head_timestamp"]
    ]
    last = store.select('prices').sort_index(level="date_time")
    last = last.reset_index('date_time').groupby('ticker').date_time.last()
    last = last.dt.tz_convert("America/New_York").dt.tz_localize(None)
    tickers['start'] = start
    tickers.head_timestamp = pd.concat([tickers.head_timestamp, last, tickers.start],axis=1).max(axis=1)
    # add one second to avoid duplicating the last tick
    tickers.head_timestamp += pd.Timedelta(seconds=1)
    
    # remove archived tickers
    archived = np.loadtxt("bloomberg/bonds/stopped_trading.dat", str)
    tickers = tickers[(~tickers.ISIN.isin(archived)) & ~(tickers.head_timestamp > end)]

    ticks = tickers.progress_apply(get_bond_data, args=(end, ib), axis=1)

    ticks = pd.concat(ticks.values)
    ticks = ticks[store.get('prices', start=1, stop=1).columns]

    store.append("prices", ticks, format="t", data_columns=True)


def get_ratings(store) -> pd.DataFrame:
    ratings = store.get("ratings")
    moody_num = ratings.moody.map(moody_scale).fillna(5)
    snp_num = ratings.snp.map(snp_scale).fillna(5)
    out_df = pd.concat([moody_num, snp_num], axis=1)
    out_df["avg_rating"] = out_df.mean(axis=1)
    return out_df


def plotPricesAfterBigMove(
    prices: pd.DataFrame,
    trgtPrice: float = None,
    priceRange: float = 2.0,
    bigMove: float = 3.0,
    numDays: int = 10,
) -> None:
    out = pd.DataFrame()
    for ticker in prices:
        close = prices[ticker]
        price_filter = (
            (close > trgtPrice - priceRange) & (close < trgtPrice + priceRange)
            if trgtPrice
            else 1
        )
        try:
            if bigMove > 0:
                t0 = close[
                    (close.diff(periods=numDays) > bigMove) & price_filter
                ].index[0]
            else:
                t0 = close[
                    (close.diff(periods=numDays) < bigMove) & price_filter
                ].index[0]
        except:
            # loan never met criteria, skip
            continue
        yx = close.iloc[
            close.index.get_loc(t0) - numDays : close.index.get_loc(t0) + numDays
        ]
        yx.index = range(-numDays, len(yx.index) - numDays)
        out = pd.concat([out, yx], axis=1)

    if len(out):
        p = "{:2.2f}".format(out.loc[0].median())
        out = out / out.loc[0] - 1
        print("Median price of event trigger", p)
        out = out.rename_axis("Trading Days before/after Event")
        out.plot(
            kind="line",
            legend=False,
            colormap="binary",
            linewidth=0.3,
            ylabel=f"Price (% Gain from {p})",
            title="All Tickers",
        )
        plt.show()
        plt.figure()
        out.T.median().plot(
            linewidth=3, ylabel=f"Price (% Gain from {p})", title="Median"
        )
        plt.show()


def getLabels(
    yields: pd.DataFrame,
    trgtPrices: Tuple[float, float] = [None, None],
    priceRange: float = 0.005,
    lookbackDays: int = 10,
    bigMove: float = 0.02,
) -> pd.DataFrame:
    yields_ma = yields.ewm(span=3).mean()
    yield_chg = yields_ma.diff(periods=lookbackDays)
    # for backtesting purposes, lag the actual signal by 1
    sell = yield_chg.shift(1) > bigMove
    buy = yield_chg.shift(1) < -bigMove
    # set to 0 to only buy a loan when the purchase price is in trgtPrice; set to 1 to buy the loan when the event trigger was within trgtPrice (but purchase price might be materially different)
    prices = yields.shift(0)
    if trgtPrices[0]:
        buy &= (yields > trgtPrices[0] - priceRange) & (
            yields < trgtPrices[0] + priceRange
        )
    if trgtPrices[1]:
        sell &= (yields > trgtPrices[1] - priceRange) & (
            yields < trgtPrices[1] + priceRange
        )

    return buy * 1.0 - sell * 1.0


def pricesToBins(
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    ptSl: Tuple[float, float] = [1.0, 1.0],
    minRet: float = 0.015,
    holdDays=10,
) -> pd.DataFrame:
    out = pd.DataFrame()
    for ticker, group in tqdm(labels.groupby("ticker"), desc="Getting bins"):
        dates = group[ticker][group[ticker] != 0].index
        t1 = add_vertical_barrier(dates, prices.loc[ticker], num_days=holdDays)
        trgt = get_daily_vol(prices.loc[ticker])
        events = get_events(
            prices.loc[ticker],
            dates,
            pt_sl=ptSl,
            target=trgt,
            min_ret=minRet,
            num_threads=2,
            vertical_barrier_times=t1,
            side_prediction=labels.loc[ticker],
        )
        bins = get_bins(events, prices.loc[ticker], vert_barrier_ret=True)
        bins["ticker"] = ticker
        out = pd.concat([out, bins])

    return (
        out.set_index("ticker", append=True).swaplevel().rename_axis(["ticker", "date"])
    )


def _get_ql_dates(
    start: pd.Timestamp, maturity: pd.Timestamp, settlement: pd.Timestamp
) -> Tuple[ql.Date]:
    start = ql.Date(start.day, start.month, start.year)
    maturity = ql.Date(maturity.day, maturity.month, maturity.year)
    settlement = ql.Date(settlement.day, settlement.month, settlement.year)
    return start, maturity, settlement

def get_ytm(
    price: pd.Series,
    issue_date: pd.Timestamp,
    maturity: pd.Timestamp,
    coupon: float,
    call_schedule: List[Tuple[pd.Timestamp, float]] = [],
):
    """
    Gets the yield of the bond at price `price`
    coupon should be of form "6.75"
    index value of price is the settlement date
    """
    t0 = price.name[1]
    if pd.isnull(maturity) or pd.isnull(issue_date) or maturity < t0:
        return np.nan
    start, maturity, settlement = _get_ql_dates(issue_date, maturity, t0)
    if np.isnan(price["close"]) or np.isnan(coupon):
        return np.nan
    schedule = ql.MakeSchedule(start, maturity, ql.Period("6M"))
    interest = ql.FixedRateLeg(schedule, ql.Actual360(), [100.0], [coupon / 100])

    putCallSchedule = ql.CallabilitySchedule()

    it = iter(call_schedule.split())
    for call in it:
        call = pd.Timestamp(call)
        callability_price = ql.CallabilityPrice(
            float(next(it)), ql.CallabilityPrice.Clean
        )
        putCallSchedule.append(
            ql.Callability(
                callability_price,
                ql.Callability.Call,
                ql.Date(call.day, call.month, call.year),
            )
        )

    bond = ql.CallableFixedRateBond(
        3,
        100,
        schedule,
        [coupon / 100.0],
        ql.Actual360(),
        ql.ModifiedFollowing,
        100,
        start,
        putCallSchedule,
    )
    try:
        return bond.bondYield(
            price["close"], ql.Actual360(), ql.Compounded, ql.Semiannual, settlement
        )
    except:
        return np.nan



class DayCounterAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.maturity = (X.maturity - X.index.get_level_values("date")).dt.days
        X.date_issued = (X.index.get_level_values("date") - X.date_issued).dt.days
        return X


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
            # fit_params[self.steps[-1][0] + "__time_index"] = X.index
        return super(MyPipeline, self).fit(X, y, **fit_params)


def add_gspread(X: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    years = X.maturity / 365.25
    X["nearest_index"] = np.searchsorted(treasury.columns, years)
    grate = X.apply(
        lambda x: treasury.loc[
            x.name[0], treasury.columns[min(6, int(x.nearest_index))]
        ],
        axis=1,
    )
    X["spread"] = X.ytm - grate / 100
    X.drop(columns=["nearest_index"], inplace=True)
    return X


def days_to_mat(raw: pd.Series, maturities: pd.Series, index_years: Tuple[int]):
    maturity = maturities.loc[raw.name]
    years = np.maximum((maturity - raw.index.get_level_values("date")).days, 0) / 365.25
    return pd.Series(
        np.minimum(np.searchsorted(index_years, years), 6), index=raw.index
    )


def trainModel(
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
    df: pd.DataFrame,
    test_size: float = 0.30,
) -> Tuple[any]:
    security = "loans" if "covi_lite" in bool_attribs else "bonds"

    def reset_index(dataframe):
        return dataframe.reset_index(inplace=False)

    num_pipeline = Pipeline(
        [
            ("day_counter", DayCounterAdder()),
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    bin_pipeline = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
            ("num", num_pipeline, num_attribs),
            ("bool", "passthrough", bool_attribs),
        ]
    )

    clf2 = RandomForestClassifier(
        n_estimators=1,
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample",
    )

    # sort dataset by event date
    df = df.swaplevel().sort_index().dropna(subset=["bin", "t1", "ret"])

    X = df.drop(columns=["bin", "ret", "clfW", "t1", "trgt"])
    y = df[["bin", "ret", "t1"]]
    clfW = df.clfW
    print("Getting average uniqueness", len(X.index), y.t1.shape)
    avgU = []
    for ticker in X.index.get_level_values("ticker").unique():
        ind_matrix = get_ind_matrix(
            y.t1.swaplevel().loc[ticker], X.swaplevel().loc[ticker]
        )
        avgU.append(get_ind_mat_average_uniqueness(ind_matrix))
    avgU = np.mean(avgU)
    rf = MyPipeline(
        [
            ("bin", bin_pipeline),
            (
                "rf",
                BaggingClassifier(
                    base_estimator=clf2,
                    n_estimators=1000,
                    max_samples=avgU,
                    max_features=1.0,
                ),
                # SequentiallyBootstrappedBaggingClassifier(
                #    y.t1, X, base_estimator=clf2, n_estimators=200, n_jobs=-1, verbose=True
                # ),
            ),
        ]
    )

    if test_size:
        # cv_gen = PurgedKFold(n_splits=3, samples_info_sets=df.t1, pct_embargo=0.05)
        # scores_array = ml_cross_val_score(
        #    rf, X, y.bin, cv_gen, sample_weight=clfW, scoring="neg_log_loss"
        # )
        # print("CV scores", scores_array, "avg", np.mean(scores_array))
        X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
            X, y, clfW, test_size=test_size, shuffle=False
        )
    else:
        X_train, y_train, W_train = X, y, clfW

    print(f"Training model with {X_train.shape} samples", pd.Timestamp.now())
    rf.fit(X_train, y_train.bin, rf__sample_weight=W_train)
    # rf.fit(X_train, y_train.bin, rf__sample_weight=W_train, rf__time_index=X_train.index)
    # save the model if we are running in production
    if test_size == 0:
        filename = "models/bond_model_{}.joblib".format(
            pd.Timestamp.now().strftime("%y%m%d")
        )
        joblib.dump(rf, filename)

    print("Getting feature importances")
    cat_columns = [
        item
        for item in bin_pipeline.named_transformers_["cat"].get_feature_names(
            cat_attribs
        )
    ]

    columns = [
        *cat_columns,
        *num_attribs,
        *bool_attribs,
    ]

    feature_importances = np.mean(
        [tree.feature_importances_ for tree in rf["rf"].estimators_], axis=0
    )
    pd.Series(feature_importances, index=columns).sort_values(ascending=True).plot(
        kind="barh"
    )
    plt.show()

    if test_size:
        print(
            f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, Test Score: {rf.score(X_test, y_test.bin):2.2f}"
        )
        y_pred_train, y_pred_test = rf.predict(X_train), rf.predict(X_test)
        y_score_train, y_score_test = (
            rf.predict_proba(X_train)[:, 1],
            rf.predict_proba(X_test)[:, 1],
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_pred_train,
            y_pred_test,
            y_score_train,
            y_score_test,
            avgU,
        )
    else:
        print(f"Train Score: {rf.score(X_train, y_train.bin):2.2f}, No Test Run")
        return rf


def printCurve(X, y, y_pred, y_score):
    print(
        f"Precision: {precision_score(y, y_pred):2.2f}, Recall: {recall_score(y, y_pred):2.2f}, Area under curve: {roc_auc_score(y, y_pred):2.2f}"
    )
    print(confusion_matrix(y, y_pred))

    fpr, tpr, thresholds = roc_curve(y, y_score)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([0, 1, 0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

    plot_roc_curve(fpr, tpr)
    plt.show()


def get_prices(
    store: pd.io.pytables.HDFStore, include_bval: bool = False
) -> pd.DataFrame:
    tickers = store.select("desc", where='ISIN != "" & columns=["bbid"]').index
    prices = pd.Series(
        index=pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"]), dtype=float
    )
    for ticker in tickers:
        ticks = store.select(
            "prices", where=f'ticker={ticker} & columns=["price", "volume"]'
        )
        # remove tz-info, which causes many many bugs below
        if not ticks.empty:
            ticks.index = ticks.index.set_levels(
                ticks.index.levels[1].tz_convert("America/New_York").tz_localize(None),
                level=1,
            )
        # many ticks happen at the same time, so we take a cumulative volume and avg price
        # FIXME should take a volume weighted average price instead
        grouped = ticks.groupby(["ticker", "date_time"])
        ticks = pd.concat([grouped.price.mean(), grouped.volume.sum()], axis=1)
        if ticks.empty:
            continue
        bars = get_volume_run_bars(ticks, 10, 1)
        prices = prices.append(bars.close)

    if include_bval:
        # use BVAL up until the first date available in IB tick
        bval = pd.read_csv(
            "bloomberg/bonds/prices.csv", index_col="date", parse_dates=True
        ).rename_axis("ticker", axis="columns")
        bval = bval[bval.index < "2020-06-29"].unstack()
        prices = prices.append(bval).sort_index()
    return prices


def parse_call_schedule(call_string: str) -> List[Tuple[pd.Timestamp, float]]:
    out = []
    if call_string == "#N/A Field Not Applicable":
        return out

    it = iter(call_string.split())
    return [
        (
            pd.Timestamp(x),
            float(next(it)),
        )
        for x in it
    ]


def get_desc(security: str) -> pd.DataFrame:
    desc = pd.read_csv(
        prefix + f"{security}/{security}_desc.csv",
        parse_dates=["date_issued", "maturity"],
        index_col="id",
    )
    desc = desc.rename_axis("ticker").rename(columns={"ticker": "name"})
    # some basic data cleaning steps to make the data ready for the pipeline
    if security == "bonds":
        desc.convertible = desc.convertible.map(yes_or_no).astype(bool)
        desc.callable = desc.callable.map(yes_or_no).astype(bool)
    elif security == "loans":
        desc.covi_lite = desc.covi_lite.map(yes_or_no).astype(bool)

    desc.cpn = pd.to_numeric(desc.cpn, errors="coerce")
    # loan coupons are in basis points (L+400) while bonds are in points (5.625)
    desc.cpn /= 100.0 if security == "loans" else 1.0
    desc.maturity = pd.to_datetime(desc.maturity, errors="coerce")
    desc.date_issued = pd.to_datetime(desc.date_issued, errors="coerce")
    return desc


def get_ticker_ytm(close: pd.Series, desc: pd.DataFrame) -> pd.Series:
    """
    close is a price series for a single ticker.
    """
    bond = desc.loc[close.index[0][0]]
    return close.to_frame("close").apply(
        get_ytm,
        args=(
            bond["date_issued"],
            bond["maturity"],
            bond["cpn"],
            bond["call_schedule"],
        ),
        axis=1,
    )


def get_grates(
    treasury: pd.DataFrame, blank_df: pd.DataFrame, maturities: pd.Series
) -> pd.DataFrame:
    """
    blank_df is a DataFrame with same cols/index as the `prices` df. Cols are tickers and index values are days
    """
    years = blank_df.groupby("ticker").apply(
        days_to_mat, maturities=maturities, index_years=treasury.columns
    )
    years_idx = dict(list(zip(range(7), treasury.columns)))
    years = years.replace(years_idx)
    years = years.droplevel("ticker")
    date_idx = years.index.date
    years.index = date_idx
    return (
        years.to_frame().T.replace(treasury.to_dict(orient="index")).T / 100
    ).set_index(blank_df.index)[0]


def data_pipeline(
    store: pd.io.pytables.HDFStore,
    prices: pd.DataFrame,
    t_params: Dict[str, any],
) -> pd.DataFrame:
    """
    Processes raw price data into bars, calculating relevent yield/spread attributes
    Identifies events based on event criteria in :t_params:
    """

    # 1) make bars
    desc = store.get("desc")
    ytms = prices.groupby(level="ticker").progress_apply(get_ticker_ytm, desc=desc)
    grates = get_grates(
        store.get("yield_curve"),
        pd.Series(index=prices.index, dtype=float),
        desc.maturity,
    )
    spreads = ytms - grates

    labels = getLabels(
        spreads,
        trgtPrices=t_params["targetSpread"],
        priceRange=t_params["spreadRange"],
        lookbackDays=t_params["lookbackDays"],
        bigMove=t_params["bigMove"],
    )
    bins = pricesToBins(
        labels,
        prices,
        ptSl=t_params["ptSl"],
        minRet=t_params["minRet"],
        holdDays=t_params["holdDays"],
    )
    clfW = getWeightColumn(bins, prices)
    bins = pd.concat([bins, clfW], axis=1)
    df = bins.join(desc, on="ticker")

    df = df.join(prices.rename("close"))
    df = df.join(ytms.rename("ytm"))
    df = df.join(spreads.rename("spread"))

    df["month"] = (
        pd.PeriodIndex(df.index.get_level_values("date"), freq="M") - 1
    ).to_timestamp()
    ratings_df = get_ratings(store)
    df = df.join(ratings_df[["moody", "avg_rating"]], on=["ticker", "month"])

    # drop convertible bonds (our data doesn't include underlying stock prices)
    if "convertible" in df.columns:
        df.drop(df[df.convertible].index, inplace=True)

    df.drop(columns=["month", "head_timestamp"], inplace=True)
    return df


def get_signal_helper(events: pd.DataFrame) -> pd.Series:
    return (
        events.reset_index("ticker")
        .groupby("ticker")
        .apply(
            lambda x: bet_size_probability(
                x, x.score, 2, x.side, 0.05, average_active=True
            )
        )
    )


def train_and_backtest(
    df: pd.DataFrame,
    prices: pd.Series,
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
    test_size: float,
) -> any:
    print("Training model")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        y_score_train,
        y_score_test,
        avgU,
    ) = trainModel(
        num_attribs, cat_attribs, bool_attribs, df.copy(), test_size=test_size
    )
    printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
    printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
    idx = y_test.shape[0]
    events = df.dropna().sort_index(level="date")[["side", "t1"]].iloc[-idx:]
    events["score"] = y_score_test
    events["signal"] = get_signal_helper(events)

    stance = pd.Series(index=prices.index, dtype=float).sort_index()
    for index, row in events.iterrows():
        stance.loc[index[0], index[1] : row.t1] = row.signal

    stance.dropna(inplace=True)
    log_returns = prices.groupby("ticker").apply(
        lambda close: np.log(close / close.shift(1))
    )
    log_returns = log_returns * stance
    log_returns = log_returns.dropna()
    log_returns = log_returns.groupby("ticker").apply(np.cumsum)
    log_returns = log_returns.reset_index("ticker")
    log_returns = log_returns.set_index(log_returns.index.to_period("B"))

    def drop_intraday(group):
        return group.drop_duplicates(subset="date", keep="last").set_index("date")[0]

    log_returns = log_returns.reset_index().groupby("ticker").apply(drop_intraday)
    returns = np.exp(log_returns) - 1
    returns = returns.swaplevel().unstack().fillna(method="pad")
    cum_port_returns = returns.sum(axis=1)
    daily_port_returns = (1 + cum_port_returns).pct_change().to_timestamp()
    daily_port_returns.iloc[0] = cum_port_returns.iloc[0]

    print(
        "Sharpe ratio", "{:4.2f}".format(pf.timeseries.sharpe_ratio(daily_port_returns))
    )
    pf.plotting.plot_rolling_returns(daily_port_returns, ax=plt.subplot(212))
    pf.plotting.plot_monthly_returns_heatmap(daily_port_returns, ax=plt.subplot(221))
    pf.plotting.plot_monthly_returns_dist(daily_port_returns, ax=plt.subplot(222))
    plt.show()


def train_production(
    df: pd.DataFrame,
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
    t_params: Dict[str, float],
) -> None:
    rf = trainModel(num_attribs, cat_attribs, bool_attribs, df, test_size=0)

    today = pd.Timestamp.now() - pd.Timedelta(days=5)
    print("Getting events for today", today)
    target = df.trgt
    current_events = (
        df.swaplevel().sort_index().drop(columns=["bin", "ret", "clfW", "t1", "trgt"])
    ).loc[today:]

    if not current_events.shape[0]:
        print(
            "There are no current events to process, last recorded event is",
            df.index.get_level_values("date").max(),
        )
        return

    pred_now = rf.predict(current_events)
    if not pred_now.sum():
        print("Hal has no profitable trades to suggest, try again tomorrow")
        return

    score_now = rf.predict_proba(current_events)[:, 1]

    data = {
        "pred": pred_now,
        "side": current_events.side.values,
        "close": current_events.close.values,
        "trgt": target.swaplevel().loc[current_events.index].values,
    }

    trades = pd.DataFrame(data, index=current_events.index)
    trades = trades.join(
        current_events[["name", "cpn", "maturity", "ISIN"]], how="left"
    ).swaplevel()
    trades["t1"] = current_events.index.get_level_values("date") + pd.Timedelta(
        days=t_params["holdDays"]
    )

    trades["score"] = score_now
    trades["signal"] = get_signal_helper(trades)

    trades["profit_take"] = trades.close * (
        1 + trades.trgt * trades.side * t_params["ptSl"][0]
    )
    trades["stop_loss"] = trades.close * (
        1 - trades.trgt * trades.side * t_params["ptSl"][1]
    )
    return trades


def get_portfolio(conId, ib) -> pd.DataFrame:
    pos = ib.positions()
    pos_df = pd.DataFrame(
        [(str(x.contract.conId), x.position) for x in pos],
        columns=["conId", "position"],
    )
    b2a = pd.Series(data=conId.index, index=conId.values)
    pos_df["ticker"] = pos_df.conId.map(b2a)
    return pos_df.dropna().set_index("ticker")


def _submit_order(row, ib):
    contract = Contract(conId=row.name, exchange="SMART")
    side = "BUY" if row.order_size > 0 else "SELL"

    # get current bid-ask context  and do not bid above the ask or ask below the bid (duh)
    last_tick = ib.reqHistoricalTicks(contract, '', pd.Timestamp.now(), 10, 'BID_ASK', useRth=True)[-1]
    price = min(row.px_last, last_tick.priceAsk if side == 'BUY' else last_tick.priceBid)

    order = LimitOrder(side, abs(row.order_size), row.px_last)
    limitTrade = ib.placeOrder(contract, order)
    ib.sleep(1)
    assert limitTrade in ib.openTrades()


def run_trades(store, prices: pd.DataFrame, ib, total_size: float = 100) -> pd.DataFrame:
    """
    Based on averaged trades and current portfolio positions, calculate what should be bought/sold
    :total_size: order size is total_size multipled by signal value
    """
    trades = store.select("trades", where=f't1 >= "{pd.Timestamp.now()}"')
    trades = trades.sort_index(level="date").groupby("ticker")
    trades = pd.concat(
        [
            trades.mean(),
            trades.t1.max().dt.date,
        ],
        axis=1,
    )

    # merge in current portfolio
    conId = store.get("desc").conId.dropna()
    portfolio = get_portfolio(conId, ib)
    trades = trades.join(portfolio.position, how="outer").fillna(0)

    px_last = prices.sort_index(level="date").groupby("ticker").last()
    trades = trades.join(px_last.to_frame("px_last"))
    trades.loc[
        (trades.px_last < trades.stop_loss) | (trades.px_last > trades.profit_take),
        "signal",
    ] = 0

    trades["new_position"] = trades.signal * total_size
    trades["order_size"] = (trades.new_position - trades.position).round(0)

    trades.index = trades.index.map(conId)
    trades = trades.round(3)
    trades[np.abs(trades.order_size) >= 2].apply(_submit_order, args=(ib,), axis=1)
    return trades


def main():
    ib = IB()
    if "--store" in sys.argv:
        store_file = "bloomberg/bonds/{}".format(
            sys.argv[sys.argv.index("--store") + 1]
        )
    else:
        store_file = "bloomberg/bonds/bond_data.h5"
    store = pd.HDFStore(store_file, mode="a")

    if "fetch" in sys.argv:
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp.now()
        if "--start" in sys.argv:
            start = sys.argv[sys.argv.index("--start") + 1]
        if "--days_prior" in sys.argv:
            end -= pd.Timedelta(days=int(sys.argv[sys.argv.index("--days_prior") + 1]))

        ib.connect("localhost", 7496, clientId=69)
        fetch_price_data(store, pd.Timestamp(start), end, ib)
        ib.disconnect()

    if "yield_curve" in sys.argv:
        old_yield_curve = store.get("yield_curve")
        new_yield_curve = fetch_treasury_data()
        new_yield_curve = old_yield_curve.append(new_yield_curve).drop_duplicates()
        store.put("yield_curve", new_yield_curve)
        print("Yield curve has been updated")

    t_params = {
        "holdDays": 10,
        "bigMove": 0.01,
        "lookbackDays": 5,
        "targetSpread": [0.10, 0.10],
        "spreadRange": 0.10,
        "ptSl": [1, 3],
        "minRet": 0.01,
    }

    if "--read" in sys.argv:
        filename = sys.argv[sys.argv.index("--read") + 1]
        try:
            df = pd.read_pickle("df_" + filename)
            prices = pd.read_pickle("prices_" + filename)
        except:
            print("Couldn't read", filename)
            raise
    else:
        include_bval = not "--no_bval" in sys.argv
        print("Including bval? ", include_bval)
        prices = get_prices(store, include_bval=include_bval)
        df = data_pipeline(store, prices, t_params)

    if "--write" in sys.argv:
        filename = sys.argv[sys.argv.index("--write") + 1]
        print("Writing prepared data to", filename)
        df.to_pickle("df_" + filename)
        prices.to_pickle("prices_" + filename)
    else:
        df.to_pickle("df_data.pkl")
        prices.to_pickle("prices_data.pkl")

    num_attribs = [
        "cpn",
        "date_issued",
        "maturity",
        "amt_out",
        "close",
        "ytm",
        "spread",
        "avg_rating",
    ]
    cat_attribs = ["side", "moody", "industry_sector"]

    bool_attribs = ["convertible", "callable"]

    if "prod" in sys.argv:
        new_trades = train_production(
            df,
            num_attribs,
            cat_attribs,
            bool_attribs,
            t_params,
        )
        trades = store.get("trades")
        new_trades = new_trades[trades.columns].swaplevel()
        trades = trades.append(new_trades).drop_duplicates()
        store.put("trades", trades, format="t", data_columns=True)

    elif "test" in sys.argv:
        test_size = 0.3
        if "--test_size" in sys.argv:
            idx = sys.argv.index("--test_size") 
            test_size = float(sys.argv[sys.argv.index("--test_size") + 1])

        train_and_backtest(
            df, prices, num_attribs, cat_attribs, bool_attribs, test_size
        )

    if "trade" in sys.argv:
        ib.connect("localhost", 7496, clientId=59)
        trades = run_trades(store, prices, ib)
        ib.disconnect()

        trades.to_csv("bond_trades_todo.csv")
        print(trades)

    print("prod: trains the model on all data")
    print("test: runs a backtest")
    print("trade: analyze current positions and suggest trades")
    ib.disconnect()


if __name__ == "__main__":
    main()
