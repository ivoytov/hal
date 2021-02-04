import numpy as np
import pandas as pd
from pandarallel import pandarallel
import datetime
import sys
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

from util.multiprocess import mp_pandas_obj
from util.utils import get_daily_vol
from labeling.labeling import *
from sample_weights.attribution import *
from sampling.bootstrapping import *
from cross_validation.cross_validation import *
from ensemble.sb_bagging import *
from bet_sizing.bet_sizing import *
from data_structures.run_data_structures import *

ib = IB()
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


def get_day_ticks(contract, day):
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
    ib.sleep(2)
    return allTicks[allTicks.index.date == day.date()]


def get_bond_data(ISIN, start, end):
    contract = Bond(symbol=ISIN, exchange="SMART", currency="USD")
    period = pd.bdate_range(start=start, end=end)
    if len(period) == 0:
        return pd.DataFrame()

    for day in tqdm(period, desc=ISIN, total=len(period)):
        yield get_day_ticks(contract, day.tz_localize(None))


def fetch_price_data(
    start: pd.Timestamp, end: pd.Timestamp = pd.Timestamp.now()
) -> None:
    ib.connect("localhost", 7496, clientId=59)
    store = pd.HDFStore("bloomberg/bonds/bond_data.h5", mode="a")
    tickers = store.select("desc", where=f'head_timestamp > pd.Timestamp("{start}")')[
        ["ISIN", "head_timestamp"]
    ]
    archived = [
        "US899896AC81",
        "US03674PAL76",
        "US674215AG39",
        "US85205TAK60",
        "US26885BAB62",
        "US12513GBC24",
        "US87612BBQ41",
        "US87901JAH86",
    ]

    for ticker, row in tqdm(
        tickers.loc["US432833AF84":].iterrows(), total=len(tickers)
    ):
        if row.ISIN in archived:
            continue
        t0 = row.head_timestamp

        last = store.select(
            "prices", where=f'ticker == "{ticker}" and date_time >= "{t0}"'
        )
        if not last.empty:
            t0 = (
                last.sort_index(level="date_time").index.levels[1][-1].tz_localize(None)
            )

        for ticks in get_bond_data(row.ISIN, max(start, t0), end):
            if not ticks.empty:
                ticks["ISIN"] = row.ISIN
                ticks["ticker"] = ticker
                ticks = ticks.set_index("ticker", append=True).swaplevel()
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
    lagDays: int = 1,
) -> pd.DataFrame:
    yields_ma = yields.ewm(span=3).mean()
    yield_chg = yields_ma.diff(periods=lookbackDays)
    # for backtesting purposes, lag the actual signal by lagDays
    sell = yield_chg.shift(lagDays) > bigMove
    buy = yield_chg.shift(lagDays) < -bigMove
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
            num_threads=1,
            vertical_barrier_times=t1,
            side_prediction=labels.loc[ticker],
        )
        bins = get_bins(events, prices.loc[ticker])
        bins["ticker"] = ticker
        out = pd.concat([out, bins])

    return (
        out.set_index("ticker", append=True).swaplevel().rename_axis(["ticker", "date"])
    )


class YieldAdder(BaseEstimator, TransformerMixin):
    def __init__(self, security: str = "bonds"):  # no *args or **kwargs
        self.security = security

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _get_ql_dates(
        start: pd.Timestamp, maturity: pd.Timestamp, settlement: pd.Timestamp
    ) -> Tuple[ql.Date]:
        start = ql.Date(start.day, start.month, start.year)
        maturity = ql.Date(maturity.day, maturity.month, maturity.year)
        settlement = ql.Date(settlement.day, settlement.month, settlement.year)
        return start, maturity, settlement

    @staticmethod
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
        t0 = price.name[1].tz_localize(None)
        if pd.isnull(maturity) or pd.isnull(issue_date) or maturity < t0:
            return np.nan
        start, maturity, settlement = YieldAdder._get_ql_dates(issue_date, maturity, t0)
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

    @staticmethod
    def get_loan_ytm(row):
        start, maturity, settlement = YieldAdder._get_ql_dates(
            row["date_issued"], row["maturity"], row.name[0]
        )
        schedule = ql.MakeSchedule(start, maturity, ql.Period("6M"))
        interest = ql.FixedRateLeg(
            schedule, ql.Actual360(), [100.0], [row["cpn"] / 100 + 0.02]
        )  # FIXME: hardcorded LIBOR at 2%
        bond = ql.Bond(0, ql.TARGET(), start, interest)

        return bond.bondYield(
            row["close"], ql.Actual360(), ql.Compounded, ql.Semiannual, settlement
        )

    @staticmethod
    def get_bond_ytm(row):
        return YieldAdder.get_ytm(
            pd.Series(row.close, index=[row.name[0]]),
            row["date_issued"],
            row["maturity"],
            row["cpn"],
        )

    def transform(self, X, y=None):
        X["ytm"] = X.apply(
            self.get_bond_ytm if self.security == "bonds" else self.get_loan_ytm, axis=1
        )
        return X


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
    years = (
        np.maximum(
            (maturity - raw.index.get_level_values("date").tz_localize(None)).days, 0
        )
        / 365.25
    )
    return pd.Series(
        np.minimum(np.searchsorted(index_years, years), 6), index=raw.index
    )


def trainModel(
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
    df: pd.DataFrame,
    test_size: float = 0.3,
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
                #  SequentiallyBootstrappedBaggingClassifier(
                #      y.t1,
                #      X,
                #      base_estimator=clf2,
                #      n_estimators=1000,
                #  ),
            ),
        ]
    )

    if test_size:
        cv_gen = PurgedKFold(n_splits=3, samples_info_sets=df.t1, pct_embargo=0.05)
        scores_array = ml_cross_val_score(
            rf, X, y.bin, cv_gen, sample_weight=clfW, scoring="neg_log_loss"
        )
        print("CV scores", scores_array, "avg", np.mean(scores_array))
        X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
            X, y, clfW, test_size=test_size, shuffle=False
        )
    else:
        X_train, y_train, W_train = X, y, clfW

    print(f"Training model with {X_train.shape} samples")
    rf.fit(X_train, y_train.bin, rf__sample_weight=W_train)

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

        #    print(cross_val_score(rf, X, y.bin, cv=5, scoring='f1'))
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


def get_prices(security: str) -> pd.DataFrame:
    if security == "bonds":
        return pd.read_csv(
            prefix + "bonds/prices.csv", index_col="date", parse_dates=True
        ).rename_axis("ticker", axis="columns")
    file_list = {
        "loans": ["2013-2015", "2016-2018", "2019-2020", "2020_stub"],
    }
    # read in historical prices
    df_list = [
        pd.read_csv(
            prefix + f"{security}/price_data_{year}.csv",
            parse_dates=["Dates"],
            index_col="Dates",
        )
        for year in file_list[security]
    ]

    prices = pd.concat(df_list, join="outer", verify_integrity=True).fillna(
        method="pad"
    )
    return prices.rename_axis("date").rename_axis("ticker", axis="columns")


def get_treasury() -> pd.DataFrame:
    filelist = ("treasury_data", "treasury_data_2021")

    df_list = [
        pd.read_csv(
            prefix + f"bonds/{filename}.csv",
            parse_dates=["Dates"],
            index_col="Dates",
        )
        for filename in filelist
    ]
    # column name is the num of years of the treasury index in that column
    out = pd.concat(df_list, join="outer", verify_integrity=True).rename_axis("date")
    out.columns = [1, 2, 3, 5, 7, 10, 30]
    return out


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
        desc.call_schedule = desc.call_schedule.apply(parse_call_schedule)
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
        YieldAdder.get_ytm,
        args=(
            bond["date_issued"],
            bond["maturity"],
            bond["cpn"],
            bond["call_schedule"],
        ),
        axis=1,
    )


def get_grates(blank_df: pd.DataFrame, maturities: pd.Series) -> pd.DataFrame:
    """
    blank_df is a DataFrame with same cols/index as the `prices` df. Cols are tickers and index values are days
    """
    treasury = get_treasury()
    treasury.index = treasury.index.date
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
    t_params: Dict[str, any],
    train_mode: bool = True,
) -> pd.DataFrame:

    # 1) make bars
    start = pd.Timestamp.now() - pd.Timedelta(days=30)
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
                ticks.index.levels[1].tz_localize(None), level=1
            )
        # many ticks happen at the same time, so we take a cumulative volume and avg price
        # FIXME should take a volume weighted average price instead
        grouped = ticks.groupby(["ticker", "date_time"])
        ticks = pd.concat([grouped.price.mean(), grouped.volume.sum()], axis=1)
        if ticks.empty:
            continue
        bars = get_volume_run_bars(ticks, 10, 10)
        prices = prices.append(bars.close)

    desc = store.get("desc")
    ytms = prices.groupby(level="ticker").apply(get_ticker_ytm, desc)
    grates = get_grates(pd.Series(index=prices.index, dtype=float), desc.maturity)
    spreads = ytms - grates

    labels = getLabels(
        spreads,
        trgtPrices=t_params["targetSpread"],
        priceRange=t_params["spreadRange"],
        lookbackDays=t_params["lookbackDays"],
        bigMove=t_params["bigMove"],
        lagDays=train_mode,
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

    df["month"] = (pd.PeriodIndex(df.index.levels[1], freq="M") - 1).to_timestamp()
    ratings_df = get_ratings(store)
    df = df.join(ratings_df[["moody", "avg_rating"]], on=["ticker", "month"])

    # drop convertible bonds (our data doesn't include underlying stock prices)
    if "convertible" in df.columns:
        df.drop(df[df.convertible].index, inplace=True)

    df.drop(columns=["month", "ISIN", "head_timestamp"], inplace=True)
    return df


def train_and_backtest(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
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
    ) = trainModel(num_attribs, cat_attribs, bool_attribs, df.copy())
    printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
    printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
    idx = y_test.shape[0]
    events = df.dropna().sort_index(level="date")[["side", "t1"]].iloc[-idx:]
    pred = pd.Series(y_score_test, index=events.index)

    signals = pd.DataFrame()
    for ticker in events.index.get_level_values("ticker").unique():
        signals = signals.join(
            bet_size_probability(
                events.loc[ticker], pred[ticker], 2, events.side.loc[ticker], 0.05
            ).rename(ticker),
            how="outer",
        )

    price_idx = prices.index.searchsorted(events.index[0][1])
    # drop dates where model didn't want to hold a position
    signals.drop(signals.index[signals.abs().sum(axis=1) == 0], inplace=True)
    positions = signals.reindex(index=prices.index[price_idx:]).fillna(0)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    positions.abs().sum(axis=1).plot(
        ax=ax[1], title="Sum of Gross Positions after Weighting"
    )
    print(
        "Number of trading days with a position",
        positions[positions.sum(axis=1) != 0].shape[0],
    )

    returns = df.ret.loc[events.index] * y_pred_test
    print("Worst and Best trades in backtest")
    print(returns[returns.abs() > 0.15].sort_values())

    portfolio_rtn_df = positions * np.log(prices / prices.shift(1)).loc[positions.index]

    portfolio_rtn = portfolio_rtn_df.sum(axis=1) / positions.abs().sum(axis=1)
    portfolio_cum_rtn_df = portfolio_rtn.cumsum().apply(np.exp)
    portfolio_cum_rtn_df.fillna(method="pad").plot(ax=ax[2], title="Portfolio PnL, %")
    plt.tight_layout()
    plt.show()
    positions.to_csv("bond_test_positions.csv")
    prices.pct_change().loc[positions.index].to_csv("bond_test_price_chg.csv")

    fig = pf.create_returns_tear_sheet(portfolio_rtn, return_fig=True)
    return fig


def train_production(
    df: pd.DataFrame,
    current_events: pd.DataFrame,
    num_attribs: List[str],
    cat_attribs: List[str],
    bool_attribs: List[str],
    t_params: Dict[str, float],
) -> None:
    rf = trainModel(num_attribs, cat_attribs, bool_attribs, df, test_size=0)

    today = current_events.index.get_level_values("date").sort_values()[-1]
    print("Getting events for today", today)
    target = current_events.trgt
    current_events = (
        current_events.swaplevel()
        .sort_index()
        .xs(today, drop_level=False)
        .drop(columns=["bin", "ret", "clfW", "t1", "trgt"])
    )

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
        current_events[["name", "cpn", "maturity"]], how="left"
    ).swaplevel()
    trades["t1"] = today + pd.Timedelta(days=t_params["holdDays"])

    pred = pd.Series(score_now, index=trades.index)

    signals = pd.DataFrame()
    for ticker in trades.index.get_level_values("ticker").unique():
        signals = signals.join(
            bet_size_probability(
                trades.loc[ticker], pred[ticker], 2, trades.side.loc[ticker], 0.05
            ).rename(ticker),
            how="outer",
        )

    trades["signal"] = signals.unstack()
    trades["profit_take"] = trades.close * (
        1 + trades.trgt * trades.side * t_params["ptSl"][0]
    )
    trades["stop_loss"] = trades.close * (
        1 - trades.trgt * trades.side * t_params["ptSl"][1]
    )
    print("Saving trades to bond_trades.csv")
    trades = trades[trades.pred == 1].sort_values("signal", ascending=False)
    trades.to_csv("trades_bonds_{}.csv".format(today.strftime("%y%m%d")))
    for index, trade in trades.iterrows():
        print(trade)


def main():
    if "bonds" in sys.argv:
        sec_type = "bonds"
    elif "loans" in sys.argv:
        sec_type = "loans"
    elif "--start" in sys.argv:
        start = sys.argv[sys.argv.index("--start") + 1]
        fetch_price_data(pd.Timestamp(start))
    else:
        print("Pass bonds or loans as arguments")
        return

    t_params = {
        "bonds": {
            "holdDays": 10,
            "bigMove": 0.01,
            "lookbackDays": 5,
            "targetSpread": [0.10, 0.10],
            "spreadRange": 0.10,
            "ptSl": [1, 2],
            "minRet": 0.01,
        },
        "loans": {
            "holdDays": 15,
            "bigMove": 0.02,
            "lookbackDays": 10,
            "targetSpread": [None, None],
            "spreadRange": None,
            "ptSl": [1, 3],
            "minRet": 0.01,
        },
    }

    store = pd.HDFStore("bloomberg/bonds/bond_data.h5 copy", mode="a")
    ratings_df = get_ratings(store)
    prices = get_prices(sec_type)
    desc = get_desc(sec_type)
    missing = set(prices.columns) - set(desc.index)
    if len(missing):
        print("Following tickers are missing descriptions:", *missing, sep="\n")
    prices = prices.drop(columns=missing)

    if "--read" in sys.argv:
        filename = sys.argv[sys.argv.index("--read") + 1]
        try:
            df = pd.read_pickle(filename)
        except:
            print("Couldn't read", filename)
            raise
    else:
        df = data_pipeline(prices, desc, ratings_df, t_params[sec_type])

    if "--write" in sys.argv:
        filename = sys.argv[sys.argv.index("--write") + 1]
        print("Writing prepared data to", filename)
        df.to_pickle(filename)

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
    if sec_type == "loans":
        cat_attribs += ["loan_type"]

    bool_attribs = {
        "bonds": ["convertible", "callable"],
        "loans": ["covi_lite"],
    }

    if "prod" in sys.argv:
        current_events = data_pipeline(
            store,
            t_params[sec_type],
            train_mode=False,
        )
        train_production(
            df,
            current_events,
            num_attribs,
            cat_attribs,
            bool_attribs[sec_type],
            t_params[sec_type],
        )
    elif "test" in sys.argv:
        fig = train_and_backtest(
            df, prices, num_attribs, cat_attribs, bool_attribs[sec_type]
        )
        print(f"Saving backtest as `backtest_{sec_type}.pdf`")
        fig.savefig(f"backtest_{sec_type}.pdf")
    else:
        print("prod: trains the model on all data")
        print("test: runs a backtest")


if __name__ == "__main__":
    main()
