import numpy as np
import multiprocessing as mp
import pandas as pd
import datetime
import sys
import os
from typing import Tuple, List, Dict
from tqdm import tqdm

tqdm.pandas()
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

yes_or_no = pd.Series({"Y": True, "N": False})
bars_len = {}

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


class HalIB(IB):
    def get_day_ticks(self, contract: Contract, day: datetime.date) -> List:
        dt = day
        ticksList = []
        while True:
            ticks = self.reqHistoricalTicks(contract, dt, "", 1000, "TRADES", False)
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


class Hal:
    bonds = {}
    margin_cushion = 0.1

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
    t_params = {
        "holdDays": 10,
        "bigMove": 0.01,
        "lookbackDays": 5,
        "targetSpread": [0.10, 0.10],
        "spreadRange": 0.10,
        "ptSl": [1, 3],
        "minRet": 0.01,
    }

    def __init__(
        self, store_file: str, model_file: str, ip: str, port: int, client_id: int = 69
    ):
        self.ib = HalIB()
        self.ib.RaiseRequestErrors = True
        self.ib.connect(ip, port, clientId=client_id)
        self.store = pd.HDFStore(store_file, mode="a")

        # instatiate all of the bonds in the database
        desc = self.store.get("desc")
        for ticker, row in desc.iterrows():
            self.bonds[ticker] = BondHal(self, ticker, row)

        if model_file:
            self.rf = joblib.load(model_file)

    def get_trade(self, ticker: str = None) -> pd.DataFrame:
        query = ['t1 >= pd.Timestamp.now()']
        if ticker:
            query += ['ticker = ticker']
        trades = self.store.select("trades", where=" and ".join(query))
        trades = trades.groupby("ticker")
        trades = pd.concat([trades.mean(), trades.t1.max()], axis=1)
        if ticker:
            assert len(trades) in [0, 1]
        return trades


    def add_trade(self, trade: pd.Series) -> None:
        """
        trade should have following columns:
            'side', 'close', 't1', 'signal', 'profit_take', 'stop_loss', 'score'
        and multiindex:
            ['date_time', 'ticker']
        """
        test = self.store.select('trades', start=1, stop=1)
        assert trade.columns.tolist() == test.columns.tolist()
        assert trade.index.names == test.index.names
        self.store.append("trades", trade, format="t", data_columns=True)

    def set_ticks(self, ticks: pd.DataFrame):
        pass

    def get_desc(self, is_trading: bool = False):
        query = ['head_timestamp > "2021-01-01"']
        if is_trading:
            query += ['conId < "a"']
        return hal.store.select("desc", where=" and ".join(query), columns=['conId', 'name', 'cpn', 'maturity', 'head_timestamp'])



    def train_and_backtest(
        self,
        df: pd.DataFrame,
        prices: pd.Series,
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
        ) = self.train_model(df.copy(), test_size=test_size)
        printCurve(X_train, y_train.bin, y_pred_train, y_score_train)
        printCurve(X_test, y_test.bin, y_pred_test, y_score_test)
        idx = y_test.shape[0]
        events = df.dropna().sort_index(level="date_time")[["side", "t1"]].iloc[-idx:]
        events["score"] = y_score_test
        events["signal"] = bet_size_probability(
            events, events.score, 2, events.side, 0.05
        )

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
            return group.drop_duplicates(subset="date_time", keep="last").set_index(
                "date_time"
            )[0]

        log_returns = log_returns.reset_index().groupby("ticker").apply(drop_intraday)
        returns = np.exp(log_returns) - 1
        returns = returns.swaplevel().unstack().fillna(method="pad")
        cum_port_returns = returns.sum(axis=1)
        daily_port_returns = (1 + cum_port_returns).pct_change().to_timestamp()
        daily_port_returns.iloc[0] = cum_port_returns.iloc[0]

        print(
            "Sharpe ratio",
            "{:4.2f}".format(pf.timeseries.sharpe_ratio(daily_port_returns)),
        )
        pf.plotting.plot_rolling_returns(daily_port_returns, ax=plt.subplot(212))
        pf.plotting.plot_monthly_returns_heatmap(
            daily_port_returns, ax=plt.subplot(221)
        )
        pf.plotting.plot_monthly_returns_dist(daily_port_returns, ax=plt.subplot(222))
        plt.show()

        feb = events.loc[
            (events.index.get_level_values("date_time") > "2021-01-31")
            & (events.signal != 0)
        ].sort_index()
        feb = feb.join(df[["ret", "name", "cpn", "maturity", "close"]])
        idx = pd.Index(zip(feb.index.get_level_values("ticker"), feb.t1.values))
        feb["exit_px"] = prices[idx].values
        feb.round(2).to_csv("ytd_backtest.csv")

    def get_portfolio(self, conId: pd.Series) -> pd.DataFrame:
        pos = self.ib.positions()
        pos_df = pd.DataFrame(
            [(x.contract.conId, x.position) for x in pos],
            columns=["conId", "position"],
        )
        b2a = pd.Series(data=conId.index, index=conId.values)
        pos_df["ticker"] = pos_df.conId.map(b2a)
        return pos_df.dropna().set_index("ticker")

    def get_open_orders(self):
        orders = self.ib.trades()
        return '\n'.join(["{} {} {}K @{}: {} {} filled".format(bonds.get(trade.contract.conId, trade.contract.tradingClass), trade.order.action, trade.order.totalQuantity, trade.order.lmtPrice, trade.orderStatus.status, trade.orderStatus.filled) for trade in orders])

    def update_yield_curve(self) -> None:
        old_yield_curve = self.store.get("yield_curve")
        new_yield_curve = fetch_treasury_data()
        new_yield_curve = old_yield_curve.append(new_yield_curve).drop_duplicates()
        self.store.put("yield_curve", new_yield_curve)
        print("Yield curve has been updated")

    def get_ratings(self) -> pd.DataFrame:
        ratings = self.store.get("ratings")
        moody_num = ratings.moody.map(moody_scale).fillna(5)
        snp_num = ratings.snp.map(snp_scale).fillna(5)
        out_df = pd.concat([moody_num, snp_num], axis=1)
        out_df["avg_rating"] = out_df.mean(axis=1)
        return out_df

    def data_pipeline(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Processes bar price data into labels, calculating relevent yield/spread attributes
        Identifies events based on event criteria in :t_params:
        """

        desc = self.store.get("desc")
        ytms = prices.groupby(level="ticker").apply(get_ticker_ytm, desc=desc)
        grates = get_grates(
            self.store.get("yield_curve"),
            pd.Series(index=prices.index, dtype=float),
            desc.maturity,
        )
        spreads = ytms - grates

        labels = getLabels(
            spreads,
            trgtPrices=self.t_params["targetSpread"],
            priceRange=self.t_params["spreadRange"],
            lookbackDays=self.t_params["lookbackDays"],
            bigMove=self.t_params["bigMove"],
        )
        bins = pricesToBins(
            labels,
            prices,
            ptSl=self.t_params["ptSl"],
            minRet=self.t_params["minRet"],
            holdDays=self.t_params["holdDays"],
        )
        clfW = getWeightColumn(bins, prices)
        bins = pd.concat([bins, clfW], axis=1)
        df = bins.join(desc, on="ticker")

        df = df.join(prices.rename("close"))
        df = df.join(ytms.rename("ytm"))
        df = df.join(spreads.rename("spread"))

        df["month"] = (
            pd.PeriodIndex(df.index.get_level_values("date_time"), freq="M") - 1
        ).to_timestamp()
        ratings_df = self.get_ratings()
        df = df.join(ratings_df[["moody", "avg_rating"]], on=["ticker", "month"])

        # drop convertible bonds (our data doesn't include underlying stock prices)
        if "convertible" in df.columns:
            df.drop(df[df.convertible].index, inplace=True)

        df.drop(columns=["month", "head_timestamp"], inplace=True)
        return df

    def plot_feature_importance(
        self, estimators: List[any], columns: List[str]
    ) -> None:
        feature_importances = np.mean(
            [tree.feature_importances_ for tree in estimators], axis=0
        )
        pd.Series(feature_importances, index=columns).sort_values(ascending=True).plot(
            kind="barh"
        )
        # disable for handsfree operation
        plt.show()

    def train_model(
        self,
        df: pd.DataFrame,
        test_size: float = 0.30,
    ) -> Tuple[any]:

        num_pipeline = Pipeline(
            [
                ("day_counter", DayCounterAdder()),
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scaler", StandardScaler()),
            ]
        )

        bin_pipeline = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_attribs),
                ("num", num_pipeline, self.num_attribs),
                ("bool", "passthrough", self.bool_attribs),
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

        cat_columns = [
            item
            for item in bin_pipeline.named_transformers_["cat"].get_feature_names(
                self.cat_attribs
            )
        ]

        columns = [
            *cat_columns,
            *self.num_attribs,
            *self.bool_attribs,
        ]

        if test_size:
            self.plot_feature_importance(rf["rf"].estimators_, columns)
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
            self.rf = rf
            return rf

    def gen_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        today = pd.Timestamp.now() - pd.Timedelta(days=5)
        print("Getting events starting from ", today)
        target = df.trgt
        current_events = (
            df.swaplevel()
            .sort_index()
            .drop(columns=["bin", "ret", "clfW", "t1", "trgt"])
        ).loc[today:]

        if not current_events.shape[0]:
            return

        pred_now = self.rf.predict(current_events)
        if not pred_now.sum():
            return None

        score_now = self.rf.predict_proba(current_events)[:, 1]

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
        trades["t1"] = current_events.index.get_level_values(
            "date_time"
        ) + pd.Timedelta(days=self.t_params["holdDays"])

        trades["score"] = score_now
        trades["signal"] = bet_size_probability(
            trades.reset_index("ticker"), trades.score, 2, trades.side, 0.05
        )

        trades["profit_take"] = trades.close * (
            1 + trades.trgt * trades.side * self.t_params["ptSl"][0]
        )
        trades["stop_loss"] = trades.close * (
            1 - trades.trgt * trades.side * self.t_params["ptSl"][1]
        )
        return trades[['side', 'close', 't1', 'signal', 'profit_take', 'stop_loss', 'score']].swaplevel()

    def run_trades(
        self, paper_mode: bool = True, total_size: float = 100
    ) -> pd.DataFrame:
        """
        Based on averaged trades and current portfolio positions, calculate what should be bought/sold
        :total_size: order size is total_size multipled by signal value
        """
        prices = self.get_bars()
        trades = self.store.select("trades", where=f't1 >= "{pd.Timestamp.now()}"')
        trades = trades.sort_index(level="date_time").groupby("ticker")
        trades = pd.concat(
            [
                trades.mean(),
                trades.t1.max().dt.date,
            ],
            axis=1,
        )

        # merge in current portfolio
        desc = self.store.get("desc")
        conId = desc.conId.dropna().astype(int)
        portfolio = self.get_portfolio(conId)
        trades = trades.join(portfolio.position, how="outer").fillna(0)

        px_last = prices.sort_index(level="date_time").groupby("ticker").last()
        trades = trades.join(px_last.to_frame("px_last"))
        trades.loc[
            (trades.px_last < trades.stop_loss) | (trades.px_last > trades.profit_take),
            "signal",
        ] = 0

        trades["new_position"] = trades.signal * total_size
        trades["order_size"] = (trades.new_position - trades.position).round(0)

        # format output with human readable names
        tickers = desc.apply(
            lambda x: "{} {} {}".format(x["name"], x.cpn, x.maturity.year), axis=1
        )
        trades["bbid"] = trades.index.map(tickers)
        trades.index = trades.index.map(conId)
        trades = (
            trades[np.abs(trades.order_size) >= 2].round(3).sort_values("order_size")
        )
        if paper_mode == False:
            trades.apply(self.submit_order, axis=1)

        return trades.set_index("bbid")

    def get_last_price(self) -> pd.Series:
        prices = self.store.select('prices', columns=['price'])
        return prices.sort_index(level="date_time").groupby("ticker").price.last()

    def get_bars(self, include_bval: bool = False) -> pd.DataFrame:
        ticks = pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["ticker", "date_time"]),
            dtype=float,
        )
        for bond in tqdm(
            self.bonds.values(), total=len(self.bonds), desc="Getting bars"
        ):
            ticks = ticks.append(bond.get_bars(include_bval))

        return ticks

    def submit_order(self, row):
        assert type(row.name) == int
        contract = Contract(conId=row.name, exchange="SMART")

        side = "BUY" if row.order_size > 0 else "SELL"

        # get current bid-ask context  and do not bid above the ask or ask below the bid
        last_tick = self.ib.reqHistoricalTicks(
            contract, "", pd.Timestamp.now(), 10, "BID_ASK", useRth=True
        )
        if not len(last_tick):
            return
        last_tick = last_tick[-1]

        if side == "BUY":
            price = row.px_last
            if last_tick.priceAsk != 0:
                price = min(row.px_last, last_tick.priceAsk)
            if price - row.close > 1:
                print(
                    row.name,
                    "trade canceled; last price",
                    price,
                    "compared with event-causing price",
                    row.close,
                )
                return
        else:
            price = row.px_last
            if last_tick.priceBid !=0:
                price = max(last_tick.priceBid, row.px_last)

        order = LimitOrder(side, abs(row.order_size), price)
        order.eTradeOnly = None
        order.firmQuoteOnly = None

        # check if order already exists
        for trade in self.ib.trades():
            if (
                trade.contract.conId == row.name
                and trade.orderStatus.status == "Submitted"
                and trade.order.action == side
            ):
                order = trade.order
                order.action = side
                order.lmtPrice = price
                order.totalQuantity = abs(row.order_size)

        # check if we are out of cash if the order is a new order
       # if not order.orderId:
       #     whatif = self.ib.whatIfOrder(contract, order)
       #     if side == "BUY" and (equity := float(whatif.equityWithLoanAfter)) < (
       #         margin := float(whatif.initMarginAfter)
       #     ) * (1 - self.margin_cushion):
       #         print(
       #             "equity",
       #             equity,
       #             "too low for order; init margin after",
       #             margin,
       #             "using margin_cushion",
       #             self.margin_cushion,
       #         )
       #         print(whatif)
       #         return

        limitTrade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        #assert limitTrade in self.ib.openTrades()

def get_ticks(store, ticker: str = None, from_date: pd.Timestamp = None):
    query = []
    if ticker:
        query.append("ticker=ticker")
    if from_date:
        query.append("date_time>=from_date")

    querystring = " and ".join(query)
    ticks = store.select(
        "prices", where=querystring, columns=["price", "volume"]
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
    return pd.concat([grouped.price.mean(), grouped.volume.sum()], axis=1)

class BondHal:
    archived = np.loadtxt("data/stopped_trading.dat", str)
    bars = None

    def __init__(self, hal: Hal, ticker: str, row: pd.Series):
        self.hal = hal
        for key in row.keys():
            setattr(self, key, row[key])
        self.ticker = ticker

    def get_ticks(self):
        ticks = self.hal.store.select(
            "prices", where="ticker=self.ticker", columns=["price", "volume"]
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
        return pd.concat([grouped.price.mean(), grouped.volume.sum()], axis=1)

    def get_bars(self, include_bval: bool = False) -> pd.DataFrame:
        if self.bars is not None:
            return self.bars

        ticks = self.get_ticks()

        if not ticks.empty:
            bars = get_volume_run_bars(ticks, 10, 1)["close"].rename_axis(
                ["ticker", "date_time"]
            )

        else:
            return None

        if include_bval:
            # use BVAL up until the first date available in IB tick
            bval = pd.read_csv(
                "data/prices.csv", index_col="date", parse_dates=True
            ).rename_axis("ticker", axis="columns")
            if self.ticker in bval:
                bval = bval.loc[bval.index < "2020-06-29", self.ticker]
                bval = bval.to_frame(self.ticker).unstack()
                bars = bars.append(bval).sort_index()

        self.bars = bars
        return bars


    def get_last_tick_time(self) -> pd.Timestamp:
        last = self.hal.store.select(
            "prices", where="ticker = self.ticker", columns=["date_time"]
        )
        last = last.index.get_level_values("date_time").max()
        return last.tz_convert("America/New_York").tz_localize(None)


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


def get_head(ISIN):
    ib.sleep(1)
    contract = Bond(symbol=ISIN, exchange="SMART", currency="USD")
    ib.qualifyContracts(contract)
    ts = ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=False)
    return pd.Series(
        {"head_timestamp": ts if ts != [] else None, "conId": str(int(contract.conId))}
    )


def fetch_head_timestamps(tickers: List[str], ib: IB) -> pd.DataFrame:
    """
    Returns DataFrame with 2 columns
    head_timestamp
    conId
    """
    desc = store.get("desc")
    tickers = desc[(desc.ISIN != "") & (pd.isnull(desc.head_timestamp))].index

    return desc.ISIN.loc[tickers].apply(get_head)


def import_new_desc(
    store,
    filepath: str,
    cusip_path: str,
) -> None:
    new_desc = pd.read_csv(filepath, index_col=0, parse_dates=[3, 4])
    old_desc = store.get("desc")
    new_tickers = set(new_desc.index) - set(old_desc.index)
    new_desc = new_desc.loc[new_tickers]
    new_desc = new_desc.rename({"ticker": "name"})
    new_desc.curr = new_desc.curr.astype("category")
    new_desc.callable = new_desc.callable.astype(bool)
    new_desc.convertible = new_desc.convertible.astype(bool)
    new_desc.amt_out = new_desc.amt_out.astype(float)
    cusips = pd.read_csv(cusip_path, usecols=["bbid", "CUSIP"]).rename(
        {"CUSIP": "ISIN"}
    )
    cusips.bbid = cusips.bbid.apply(lambda x: x[:8])
    cusips = cusips.set_index("bbid")
    new_desc = new_desc.join(cusips.CUSIP)
    new_desc = new_desc.rename(columns={"CUSIP": "ISIN", "ticker": "name"})
    new_desc = new_desc.rename_axis("ticker")
    new_desc = pd.concat([new_desc, new_desc.ISIN.apply(get_head)], axis=1)
    return old_desc.append(new_desc)


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
    for ticker, group in labels.groupby("ticker"):
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
        bins = get_bins(events, prices.loc[ticker], vert_barrier_ret=True)
        bins["ticker"] = ticker
        out = pd.concat([out, bins])

    return (
        out.set_index("ticker", append=True)
        .swaplevel()
        .rename_axis(["ticker", "date_time"])
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
        X.maturity = (X.maturity - X.index.get_level_values("date_time")).dt.days
        X.date_issued = (X.index.get_level_values("date_time") - X.date_issued).dt.days
        return X


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
            # fit_params[self.steps[-1][0] + "__time_index"] = X.index
        return super(MyPipeline, self).fit(X, y, **fit_params)


def days_to_mat(raw: pd.Series, maturities: pd.Series, index_years: Tuple[int]):
    maturity = maturities.loc[raw.name]
    years = (
        np.maximum((maturity - raw.index.get_level_values("date_time")).days, 0)
        / 365.25
    )
    return pd.Series(
        np.minimum(np.searchsorted(index_years, years), 6), index=raw.index
    )


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
        f"{security}_desc.csv",
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


def automated_strategy(hal: Hal, include_bval: bool):
    global bars_len
    for ticker, bond in tqdm(
        hal.bonds.items(), desc="fetching ticks", total=len(hal.bonds)
    ):
        timestamp = bond.head_timestamp
        # don't check tickers where we just refreshed them less than an hour ago
        if pd.isnull(timestamp) or timestamp > pd.timestamp.now() - pd.timedelta(
            hours=1
        ):
            continue

        volume_ticks = bond.fetch_price_data(timestamp, pd.timestamp.now())
        # don't bother trying to create bars if less than $1m has traded since last run (if nothing has traded, value is 0)
        if volume_ticks < 1000:
            continue

        bars = bond.get_bars(include_bval=include_bval)

        # if we have already processed ticker, check if any new bars were added
        if ticker in bars_len and len(bars) - bars_len[ticker] == 0:
            continue

        bars_len[ticker] = len(bars)
        events = hal.data_pipeline(bars)
        events = events[
            events.index.get_level_values("date_time")
            > pd.timestamp.now() - pd.timedelta(days=4)
        ]
        if not len(events):
            continue
        new_trades = hal.gen_trades(events)
        hal.add_trade(new_trades)
        if new_trades is none:
            continue
        print(new_trades)
        if isinstance(new_trades, pd.dataframe):
            print("identified", len(new_trades), "new trades:")
            print(new_trades)

            hal.run_trades(false)


def main():
    paper_mode = false if "--live" in sys.argv else true

    if "--store_file" in sys.argv:
        store_file = sys.argv[sys.argv.index("--store_file") + 1]
    else:
        store_file = "data/bond_data.h5"

    if "--read_model" in sys.argv:
        model_file = sys.argv[sys.argv.index("--read_model") + 1]
    else:
        model_file = None

    assert "--host" in sys.argv
    assert "--port" in sys.argv
    hostname = sys.argv[sys.argv.index("--host") + 1]
    port = int(sys.argv[sys.argv.index("--port") + 1])

    hal = Hal(store_file, model_file, hostname, port)

    if "yield_curve" in sys.argv:
        hal.update_yield_curve()

    include_bval = not "--no_bval" in sys.argv
    if "--read" in sys.argv:
        try:
            df = pd.read_pickle("df_data.pkl")
            prices = pd.read_pickle("prices_data.pkl")
        except:
            print("Couldn't read")
            raise
    elif any(x in sys.argv for x in ["train", "test"]):
        print("Including bval? ", include_bval)
        prices = hal.get_bars(include_bval=include_bval)
        df = hal.data_pipeline(prices)

        df.to_pickle("df_data.pkl")
        prices.to_pickle("prices_data.pkl")

    if "train" in sys.argv:
        rf = hal.train_model(df, test_size=0)

    elif "test" in sys.argv:
        test_size = 0.3
        if "--test_size" in sys.argv:
            idx = sys.argv.index("--test_size")
            test_size = float(sys.argv[sys.argv.index("--test_size") + 1])

        hal.train_and_backtest(df, prices, test_size)



    

    docstring = """
    train: trains the model on all data, saves it to .joblib
    test: runs a backtest
    trade: analyze current positions and place trades (one shot)
    cycle: run endless cycle of fetching new data and placing trades
    yield_curve: fetches latest treasury yields from the US Treasury and updates the HDF5 file

    Arguments:
    --store_file <filename> which HD5 file to use
    --read                  expects to open two files, df_data.pkl and prices_data.pkl
    --read_model <filename> reads in RF model (.joblib file)
    --test_size <float>     what % of dataset to reserve for test set
    --no_bval               only uses tick data (post 6/29/2020)
    --live                  live mode that will place live trades
    """
    print(docstring)

    hal.ib.disconnect()


if __name__ == "__main__":
    main()
