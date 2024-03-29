"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
"""

import numpy as np
import pandas as pd

from sampling.concurrent import (
    num_concurrent_events,
    get_av_uniqueness_from_triple_barrier,
)
from util.multiprocess import mp_pandas_obj


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    """
    Snippet 4.10, page 69, Determination of Sample Weight by Absolute Return Attribution
    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) close prices
    :param molecule: (an array) a set of datetime index values for processing.
    :return: (pd.Series) of sample weights based on number return and concurrency for molecule
    """

    ret = np.log(close_series).diff()  # Log-returns, so that they are additive
    weights = pd.Series(index=molecule)

    for t_in, t_out in label_endtime.loc[weights.index].iteritems():
        # Weights depend on returns and label concurrency
        weights.loc[t_in] = (
            ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]
        ).sum()
    return weights.abs()


def get_weights_by_return(triple_barrier_events, close_series, num_threads=5):
    """
    Snippet 4.10(part 2), page 69, Determination of Sample Weight by Absolute Return Attribution
    This function is orchestrator for generating sample weights based on return using mp_pandas_obj.
    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param close_series: (pd.Series) close prices
    :param num_threads: (int) the number of threads concurrently used by the function.
    :return: (pd.Series) of sample weights based on number return and concurrency
    """

    has_null_events = bool(triple_barrier_events.isnull().values.any())
    has_null_index = bool(triple_barrier_events.index.isnull().any())
    assert (
        has_null_events is False and has_null_index is False
    ), "NaN values in triple_barrier_events, delete nans"

    num_conc_events = mp_pandas_obj(
        num_concurrent_events,
        ("molecule", triple_barrier_events.index),
        num_threads,
        close_series_index=close_series.index,
        label_endtime=triple_barrier_events["t1"],
    )
    num_conc_events = num_conc_events.loc[
        ~num_conc_events.index.duplicated(keep="last")
    ]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
    weights = mp_pandas_obj(
        _apply_weight_by_return,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=num_conc_events,
        close_series=close_series,
    )
    weights *= weights.shape[0] / weights.sum()
    return weights


def get_weights_by_time_decay(
    triple_barrier_events, close_series, num_threads=5, decay=1
):
    """
    Snippet 4.11, page 70, Implementation of Time Decay Factors
    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param close_series: (pd.Series) close prices
    :param num_threads: (int) the number of threads concurrently used by the function.
    :param decay: (int) decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old
        - decay = 0 means that weights converge linearly to zero, as they become older
        - decay < 0 means that the oldes portion c of the observations receive zero weight (i.e they are erased from memory)
    :return: (pd.Series) of sample weights based on time decay factors
    """
    assert (
        bool(triple_barrier_events.isnull().values.any()) is False
        and bool(triple_barrier_events.index.isnull().any()) is False
    ), "NaN values in triple_barrier_events, delete nans"

    # Apply piecewise-linear decay to observed uniqueness
    # Newest observation gets weight=1, oldest observation gets weight=decay
    av_uniqueness = get_av_uniqueness_from_triple_barrier(
        triple_barrier_events, close_series, num_threads
    )
    decay_w = av_uniqueness["tW"].sort_index().cumsum()
    if decay >= 0:
        slope = (1 - decay) / decay_w.iloc[-1]
    else:
        slope = 1 / ((decay + 1) * decay_w.iloc[-1])
    const = 1 - slope * decay_w.iloc[-1]
    decay_w = const + slope * decay_w
    decay_w[decay_w < 0] = 0  # Weights can't be negative
    return decay_w


def getWeights(events, prices):
    events = events.dropna(subset=["t1"])
    numCoEvents = mpNumCoEvents(prices.index, events["t1"], events.index)
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep="last")]
    numCoEvents = numCoEvents.reindex(prices.index).fillna(0)
    retW = mpSampleW(events["t1"], numCoEvents, prices, events.index)
    return getTimeDecay(retW, clfLastW=0.5)


def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed

    Any event that starts before t1[modelcule].max() impacts the count.
    """
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[
        : t1[molecule].max()
    ]  # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(pd.to_datetime([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0] : iloc[1] + 1])
    t1 = t1.tz_localize(None).dt.tz_localize(None)
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1.0
    return count.loc[molecule[0] : t1[molecule].max()]


def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule, dtype="float64")
    for tIn, tOut in t1.loc[wght.index].iteritems():
        try:
            beg, end = ret.index.get_loc(tIn), ret.index.get_loc(tOut.round('U'))
        except:
            breakpoint()
        wght.loc[tIn] = (ret.iloc[beg:end] / numCoEvents.iloc[beg:end]).sum()
    return wght.abs()


def getTimeDecay(tW, clfLastW=1.0):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    # if none of the weights are above zero, make sure the last weight is '1'
    if not clfW.iloc[-1]:
        clfW.iloc[-1] = 1
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    return clfW


def getWeightColumn(bins: pd.DataFrame, prices: pd.DataFrame):
    out = pd.Series(index=bins.index, name='clfW')
    for ticker in bins.dropna(subset=["t1"]).index.unique("ticker"):
        clfW = getWeights(bins.loc[ticker], prices[ticker])
        out.loc[ticker, clfW.index] = clfW.values
    return out
