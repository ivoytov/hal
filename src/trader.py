from momentum import *
from dataclasses import dataclass
import argparse
import pathlib
from ipaddress import ip_address
import datetime
import zmq

def logger_monitor(message, time=True, sep=True):
    with open(args.log_file, 'a') as f:
        t = str(datetime.datetime.now())
        msg = ''
        if time:
            msg += '\n' + t + '\n'
        if sep:
            msg += 66 * '=' + '\n'
        msg += message + '\n\n'
        
        socket.send_string(msg)
        f.write(msg)

def report_positions():
    out = '\n\n' + 50 * '=' + '\n'
    out += hal.get_portfolio(conId)
    out += hal.get_open_orders()

    logger_monitor(out)
    print(out)


@dataclass
class BondDC:
    ticker: str
    conId: int
    name: str
    coupon: float
    maturity: pd.Timestamp
    ticks: pd.DataFrame
    bars: pd.DataFrame
    
    def __repr__(self):
        return "{} {} {}".format(self.name, self.coupon, self.maturity.year)

def get_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bars = get_volume_run_bars(ticks, 10, 1)
    if bars is None:
        return None
    return bars["close"].rename_axis(["ticker", "date_time"])

def automated_strategy(tickers):
    for t in tickers:
        ticks = t.ticks
        ticks = [t for t in ticks if t.tickType == 4]
        new_ticks = pd.DataFrame(columns=["price", "volume", "exchange", "specialCon", "pastLimit", "unreported"])
        bond = bonds[t.contract.conId]
        for tick in ticks:
            logger_monitor("{} {} {} {} {}".format(bond, tick.time, tick.price, tick.size, tick.tickType)) 
            date_time = pd.to_datetime(tick.time).tz_convert("America/New_York").tz_localize(None)
            tick_df = pd.DataFrame({"price": tick.price, "volume": tick.size}, index=[(bond.ticker, date_time)])

            bond.ticks = bond.ticks.append(tick_df)
            bond.bars = get_bars(bond.ticks)
            if bond.bars is None:
                continue

            events = hal.data_pipeline(bond.bars)
            if not len(events) or events.iloc[-1].name[1] != date_time:
                continue

            new_trades = hal.gen_trades(events)
            if new_trades is None:
                continue
            hal.add_trade(new_trades)
            run_trade(bond.ticker)
            

def run_trade(ticker: str = None) -> None:
    trade = hal.get_trade(ticker = ticker)

    # merge in current portfolio
    portfolio = hal.get_portfolio(conId)
    trade = trade.join(portfolio.position, how="outer").fillna(0)
    trade['px_last'] = conId.loc[trade.index].apply(lambda x: bonds[x].bars.iloc[-1])
    
    trade.loc[(trade.px_last < trade.stop_loss) | (trade.px_last > trade.profit_take), "signal"] = 0

    trade["new_position"] = trade.signal * total_size
    trade["order_size"] = (trade.new_position - trade.position).round(0)

    trade.index = trade.index.map(conId)
    trade = trade[np.abs(trade.order_size) >= 2].round(3).sort_values("order_size")
    trade.apply(hal.submit_order, axis=1)
    report_positions()


def fetch_price_data() -> None:
    t0 = hal.get_last_price().index
    for conId, bond in tqdm(bonds.items(), desc="Fetching ticks"):
        if not len(bond.ticks):
            continue
        start = bond.ticks.index.get_level_values('date_time')[-1] + pd.Timedelta(seconds=1)
        if pd.Timestamp.now() - start < pd.Timedelta(hours=1) or pd.Timestamp.now().time() > datetime.time(17,0):
            continue
        period = pd.bdate_range(start=start, end=pd.Timestamp.now())
        ticks = pd.DataFrame()
        if len(period) != 0:
            # bdate_range resets all HH:MM to 00:00. We change the first value back to `start` with exact datetime
            period = period[1:].union([start])

        for day in period:
            ticks = ticks.append(hal.ib.get_day_ticks(Bond(conId=conId, exchange="SMART"), day))

        ticks["ticker"] = bond.ticker
        ticks = ticks.set_index("ticker", append=True).swaplevel()

        if not len(ticks):
            continue
        ticks = ticks[
            ["price", "volume", "exchange", "specialCon", "pastLimit", "unreported"]
        ]
        ticks = ticks.rename_axis(["ticker", "date_time"])
        ticks.volume = ticks.volume.astype(int)
        ticks.pastLimit = ticks.pastLimit.astype(bool)
        ticks.unreported = ticks.unreported.astype(bool)

        hal.store.append("prices", ticks, format="t", data_columns=True)


parser = argparse.ArgumentParser(description='Trade some bonds.')
parser.add_argument('--host', type=str, required=True, help="IP address of IB Gateway") 
parser.add_argument('--port', type=int, required=True, help="Port of IB Gateway")
parser.add_argument('--store_file', type=pathlib.Path, required=True, help="Path to HDF5 database file (.h5)")
parser.add_argument('--model_file', type=pathlib.Path, required=True, help="Path to trained random forest model file (.joblib)")
parser.add_argument('--log_file', type=pathlib.Path, required=True, help="Path to logfile to append to")

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://0.0.0.0:5555')

args = parser.parse_args()
hal = Hal(args.store_file, args.model_file, str(args.host), args.port, 2)
total_size = 100 # max trade size, in $ thousands

desc = hal.store.select("desc", where="conId < 'a'", columns=['conId', 'name', 'cpn', 'maturity', 'head_timestamp'])
conId = desc.conId.astype(int)

bonds = {} 
for ticker, row in desc.iterrows():
    ticks = get_ticks(hal.store, ticker, pd.Timestamp.now() - pd.Timedelta(days=20))
    bonds[int(row.conId)] = BondDC(ticker, int(row.conId), row['name'], row.cpn, row.maturity, ticks, get_bars(ticks)) 

contracts = [Bond(conId=conId, exchange="SMART") for conId in desc['conId']]
size = 50 # metering to avoid warnings 
for pos in range(0, len(contracts), size):
    hal.ib.qualifyContracts(*contracts[pos:pos + size])
    hal.ib.sleep(1)

hal.update_yield_curve()
fetch_price_data()

for contract in contracts:
    hal.ib.reqMktData(contract, '', False, False)

hal.ib.pendingTickersEvent += automated_strategy
# every N seconds refresh portfolio and see if anything should be sold
for t in hal.ib.timeRange(datetime.datetime.now(), datetime.time(23,59), 60):
    run_trade()
hal.ib.pendingTickersEvent -= automated_strategy
hal.ib.disconnect()
