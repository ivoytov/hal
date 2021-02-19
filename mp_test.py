from multiprocessing import Pool
import pandas as pd
from ib_insync import *

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

def f(x):
    return x*x

if __name__ == '__main__':
    ib = IB()
    ib.connect("localhost", 7496, clientId=69)
    ISIN = ['US15135BAJ08',
        'US118230AQ44',
        'US118230AQ44',
        'US118230AQ44',
        'US118230AQ44',]

    start = pd.Timestamp.now() - pd.Timedelta(days=5)
    end = pd.Timestamp.now()


