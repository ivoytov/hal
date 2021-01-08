"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

import numpy as np
import pandas as pd

# Snippet 3.1, page 44, Daily Volatility Estimates
from util.multiprocess import mp_pandas_obj

# triple barrier method function
# close:  a pandas series of prices
# events: a pandas dataframe wiht columns
#   - t1:   the timestamp of vertical barrier. When np.nan, no vertical barrier
# .  - trgt: the unit width of the horizontal barriers
# ptSl: a list of two non negative floats
#   - ptSl[0] the factor that multiplies trgt to set the width of the upper barrier. if 0, no upper barrier
#   - ptSl[1] the factor tha multiplesi trgt ot set the width of the lower barrier. if 0, no lower barrier
# molecule a list with the subset of event indices that will be processed by single thread


def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index, dtype="float64")  # NaNs

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index, dtype="float64")  # NaNs

    for loc, t1 in events_["t1"].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking
    return out


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
    """
    Snippet 3.2, page 45, Triple Barrier Labeling Method
    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.
    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.
    :param close: (series) close prices
    :param events: (series) of indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (array) element 0, indicates the profit taking level; element 1 is stop loss level
    :param molecule: (an array) a set of datetime index values for processing
    :return: DataFrame of timestamps of when first barrier was touched
    """
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_["trgt"]
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_["trgt"]
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    # Get events
    for loc, vertical_barrier in events_["t1"].fillna(close.index[-1]).iteritems():
        closing_prices = close[loc:vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[
            loc, "side"
        ]  # Path returns
        out.loc[loc, "sl"] = cum_returns[
            cum_returns < stop_loss[loc]
        ].index.min()  # Earliest stop loss date
        out.loc[loc, "pt"] = cum_returns[
            cum_returns > profit_taking[loc]
        ].index.min()  # Earliest profit taking date

    return out


def getVertBarrier(gRaw, tEvents, numDays: int) -> pd.Series:
    t1 = gRaw.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < gRaw.shape[0]]
    t1 = pd.Series(gRaw.index[t1], index=tEvents[: t1.shape[0]])
    t1 = t1.append(
        pd.Series(pd.NaT, index=set(tEvents) - set(t1.index))
    )  # NaNs at the end
    t1 = t1.rename("t1")
    return t1


# find the time of the first barrier touch
#  close: pandas series of prices
#  tEvents pandas timeindex of timestamps that will seed every ttriple barrier
#  ptSl a non negative flat that sets the width of th two barriers (symm)
# t1 pandas series with the timestamps of the vert barriers. pass false to disable
# trgt: pandas series of targets, expressed in terms of absolute returns
# minRet minimum target return required for running a triple barrier search
# numThreads  not used yet

# output:
# - t1: timestamp of when the first barrier is touched
# - trgt: the target that was used to generate the horizontal barrier


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]

    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents, dtype="datetime64")

    # 3) form events object, apply stop loss on t1
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        ptSl_ = [ptSl, ptSl]
    else:
        side_ = side.loc[trgt.index]
        ptSl_ = ptSl[:2]

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )

    df0 = applyPtSlOnT1(close, events, ptSl_, events.index)
    events["t1"] = df0.apply(lambda x: x.min(), axis=1)
    if side is None:
        events = events.drop("side", axis=1)
    return events


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(
    close,
    t_events,
    pt_sl,
    target,
    min_ret,
    num_threads,
    vertical_barrier_times=False,
    side_prediction=None,
):
    """
    Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.
    :param close: (series) Close prices
    :param t_events: (series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (series) Side of the bet (long/short) as decided by the primary model
    :return: (data frame) of events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] Profit taking multiple
            -events['sl'] Stop loss multiple
    """

    # 1) Get target
    target = target.loc[t_events]
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.loc[
            target.index
        ]  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat(
        {"t1": vertical_barrier_times, "trgt": target, "side": side_}, axis=1
    )
    events = events.dropna(subset=["trgt"])

    # Apply Triple Barrier
    if not len(target):
        return events
    first_touch_dates = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=("molecule", events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl_,
    )

    events["t1"] = first_touch_dates.min(axis=1)  # pd.min ignores nan

    if side_prediction is None:
        events = events.drop("side", axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events["pt"] = pt_sl[0]
    events["sl"] = pt_sl[1]

    return events


# daily vol, reindexed to close
def getDailyVol(close, span=14):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    df0 = close.loc[df0.index] / close.loc[df0].values - 1  # daily returns
    df0 = df0.ewm(span=span).std().rename("trgt")
    return df0


def barrier_touched(out_df, events):
    """
    Snippet 3.9, pg 55, Question 3.3
    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.
    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0
    :param out_df: (DataFrame) containing the returns and target
    :param events: (DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (DataFrame) containing returns, target, and labels
    """
    store = []
    for date_time, values in out_df.iterrows():
        ret = values["ret"]
        target = values["trgt"]

        pt_level_reached = ret > target * events.loc[date_time, "pt"]
        sl_level_reached = ret < -target * events.loc[date_time, "sl"]

        if ret > 0.0 and pt_level_reached:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(
                np.sign(ret)
            )  # replace with 0 if we want vert barrier to be labeled as 0

    # Save to 'bin' column and return
    out_df["bin"] = store
    return out_df


def get_bins(triple_barrier_events, close):
    """
    Snippet 3.7, page 51, Labeling for Side & Size with Meta Labels
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.
    :param triple_barrier_events: (data frame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (series) close prices
    :return: (data frame) of meta-labeled events
    """

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=["t1"])
    all_dates = events_.index.union(other=events_["t1"].values).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=triple_barrier_events.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df["ret"] = np.log(prices.loc[events_["t1"].values].values) - np.log(
        prices.loc[events_.index]
    )
    out_df["trgt"] = triple_barrier_events["trgt"]
    out_df["t1"] = triple_barrier_events["t1"]

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events_:
        out_df["ret"] = out_df["ret"] * events_["side"]  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = barrier_touched(out_df, triple_barrier_events)

    # Meta labeling: label incorrect events with a 0
    if "side" in events_:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    # Transform the log returns back to normal returns.
    out_df["ret"] = np.exp(out_df["ret"]) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if "side" in tb_cols:
        out_df["side"] = triple_barrier_events["side"]

    return out_df
