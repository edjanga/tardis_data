import concurrent.futures
import pdb
import typing
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
#import nest_asyncio
from dotenv import load_dotenv, find_dotenv
import os
from tardis_dev import datasets
#nest_asyncio.apply()
logging.basicConfig(level=logging.DEBUG)
import pytz
import glob
from helpers import symbol_ls
import concurrent.futures


load_dotenv(find_dotenv())

my_api_key = os.environ.get('API_KEY')
exchanges = ['deribit']
side_action_dd = {'buy': 'ask', 'sell': 'bid'}
data_types = ['book_snapshot_5']

def file_name_nested(exchange, data_type, date, symbol, format):
    directory = f'./{exchange}/{str(date.year)}/{data_type}'
    if not os.path.exists(directory):
        os.mkdir(directory)
    return (f"{exchange}/{str(date.year)}/{data_type}/"
            f"{date.strftime('%Y-%m-%d')}_{symbol.replace('-', '')}.{format}.gz")


def parser_data(df: pd.DataFrame) -> None:
    df = df[['symbol', 'timestamp', 'asks[0].price', 'bids[0].price']].set_index('timestamp')
    df.index = [datetime.fromtimestamp(idx // 1_000_000, tz=pytz.utc) for idx in df.index]
    df = df.assign(price=df[['asks[0].price', 'bids[0].price']].mean(axis=1))
    df.drop(['asks[0].price', 'bids[0].price'], axis=1, inplace=True)
    df = df.resample('1T').last()
    df = pd.pivot(df, columns='symbol', values='price')
    df.columns.name = None
    df.index.name = 'timestamp'
    if df.empty:
        return None
    year = df.index[0].year
    date = df.index[0].strftime('%Y-%m-%d')
    symbol = df.columns[0]
    df.to_csv(f'./binance/{str(year)}/{date}_{symbol}.csv.gz', compression='gzip')


def parser_raw_book_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    original_columns_to_be_deleted = ['asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount']
    df = df[['exchange', 'symbol', 'timestamp']+original_columns_to_be_deleted]
    df = df.set_index('timestamp')
    df.index = [datetime.fromtimestamp(int(idx)/ 1_000_000, tz=pytz.timezone('Etc/UTC')) for idx in df.index]
    df = df.resample('T').last()
    df = df.assign(bidPx=df['bids[0].price'], bidQty=df['bids[0].amount'],
                   askPx=df['asks[0].price'], askQty=df['asks[0].amount'])
    mid = df.filter(regex='Px').mean(axis=1)
    df = df.assign(pret_1M=np.log(mid.div(mid.shift())))
    df.drop(original_columns_to_be_deleted, axis=1, inplace=True)
    df = \
        df.assign(timeHMs=[''.join((h, m)) for h, m in zip(df.index.strftime('%H'), df.index.strftime('%M'))])
    df = df.assign(timeHMe=df['timeHMs'].shift(-1).fillna('0000'))
    return df


def parser_raw_book_trades_side(df: pd.DataFrame) -> pd.DataFrame:
    side = df['side'].unique()[0]
    nrTrades = df.resample('T').count()['symbol']
    df = df.resample('T').agg({'symbol': 'last', 'amount': 'last', 'notional': 'sum'})
    df[f'vol{side_action_dd[side].title()}NrTrades_lit'] = nrTrades
    df = df.rename(columns={'amount': f'vol{side_action_dd[side].title()}Qty_lit',
                            'notional': f'vol{side_action_dd[side].title()}Notional_lit'})
    return df


def parser_raw_book_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('timestamp')
    df.index = [datetime.fromtimestamp(int(idx) / 1_000_000, tz=pytz.timezone('Etc/UTC')) for idx in df.index]
    df.drop(['local_timestamp', 'id', 'exchange'], axis=1, inplace=True)
    df = df.assign(notional=df['price'] * df['amount'])
    trades_group = df.groupby(by='side').apply(lambda x: parser_raw_book_trades_side(x))
    buy = trades_group.loc[trades_group.index.get_level_values(0) == 'buy', :].dropna(axis=1, how='all').droplevel(0, 0)
    sell = \
        trades_group.loc[trades_group.index.get_level_values(0) == 'sell', :].dropna(axis=1, how='all').droplevel(0, 0)
    df = pd.concat([buy, sell], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def download_data(start: datetime, exchange: str = 'binance') -> None:
    from_date_string = start.strftime('%Y-%m-%d')
    to_date_string = (start+relativedelta(days=1)).strftime('%Y-%m-%d')
    if not os.path.exists(f'./binance/{from_date_string[:4]}'):
        os.mkdir(f'./binance/{from_date_string[:4]}')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(datasets.download, exchange=exchange, data_types=data_types,
                                   symbols=[symbol], from_date=from_date_string, to_date=to_date_string,
                                   format='csv', api_key=my_api_key, download_dir=f'./', get_filename=file_name_nested)
                   for symbol in symbol_ls]
        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        for future in concurrent.futures.as_completed(done):
            book_snapshot_dd = \
                {symbol: parser_raw_book_snapshot(
                    pd.read_csv(f'./binance/{str(start.year)}/book_snapshot_5/{from_date_string}_{symbol}.csv.gz',
                                compression='gzip'))
                 for symbol in symbol_ls}
            # trades_dd = \
            #     {symbol: parser_raw_book_trades(
            #         pd.read_csv(f'./binance/{str(start.year)}/trades/{from_date_string}_{symbol}.csv.gz',
            #                     compression='gzip')) for symbol in symbol_ls}
            print(f'[DATA PARSING]: Individual data parsing for {from_date_string} is completed.')
            book_snapshot = pd.concat(book_snapshot_dd).droplevel(0, 0)
            df = book_snapshot
            #trades = pd.concat(trades_dd).droplevel(0, 0)
            #df = pd.concat([book_snapshot, trades], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df.to_csv(f'./binance/{str(start.year)}/{from_date_string}.csv.gz', compression='gzip')
            print(f'[FILE GENERATION]: Aggregated file for {from_date_string} has been generated.')
            book_snapshot_dd = \
                {symbol: os.remove(f'./binance/{start.year}/book_snapshot_5/{from_date_string}_{symbol}.csv.gz')
                 for symbol in symbol_ls}
            # trades_dd = \
            #     {symbol: os.remove(f'./binance/{start.year}/trades/{from_date_string}_{symbol}.csv.gz')
            #      for symbol in symbol_ls}


def add_file(file: str, feature: str = 'Px') -> None:
    bid = pd.read_csv(file, compression='gzip', usecols=['Unnamed: 0', 'symbol', f'bid{feature}'])
    bid = pd.pivot_table(bid, values=f'bid{feature}', columns='symbol', index='Unnamed: 0')
    ask = pd.read_csv(file, compression='gzip', usecols=['Unnamed: 0', 'symbol', f'ask{feature}'])
    ask = pd.pivot_table(ask, values=f'ask{feature}', columns='symbol', index='Unnamed: 0')
    tmp = .5*(bid+ask) if feature == 'Px' else (bid+ask)
    tmp.index.name = 'timestamp'
    tmp.columns.name = None
    data[file] = tmp


def aggregate_per_year(year: int, feature: str='Px') -> None:
    files = glob.glob(f'./binance/{str(year)}/*.csv.gz')
    global data
    data = dict()
    for file in files:
        add_file(file, feature)
    data = pd.concat(data).droplevel(axis=0, level=0)
    data.to_parquet(f'./binance/aggregate{str(year)}') if feature == 'Px' else \
        data.to_parquet(f'./binance/aggregate{str(year)}_volume')


if __name__ == '__main__':

    dates = list(pd.date_range(start=datetime(2021, 1, 1, tzinfo=pytz.utc),
                               end=datetime(2023, 7, 1, tzinfo=pytz.utc),
                               freq='1D', inclusive='left'))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_data, date) for _, date in enumerate(dates)]
    for year in list(range(2021, 2024)):
        aggregate_per_year(year)
        aggregate_per_year(year, 'Qty')