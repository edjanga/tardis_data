import concurrent.futures
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from dotenv import load_dotenv, find_dotenv
import os
from tardis_dev import datasets
logging.basicConfig(level=logging.DEBUG)
import pytz
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
    df.index = [datetime.fromtimestamp(int(idx) // 1_000_000, tz=pytz.timezone('Etc/UTC')) for idx in df.index]
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


def download_data(start: str, end: str, exchange: str = 'binance') -> None:
    date_range = pd.date_range(start=start, end=end, inclusive='both', freq='1D')
    for date in date_range:
        price_dd = dict()
        volume_dd = dict()
        from_date_string = date.strftime('%Y-%m-%d')
        to_date_string = (date+relativedelta(days=1)).strftime('%Y-%m-%d')
        datasets.download(exchange=exchange, data_types=data_types,
                          symbols=symbol_ls, from_date=from_date_string, to_date=to_date_string,
                          format='csv', api_key=my_api_key, download_dir=f'./')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=symbol) for symbol in symbol_ls]
            for future in concurrent.futures.as_completed(futures):
                symbol = future.result()
                file = pd.read_csv(f'./binance_book_snapshot_5_{from_date_string}_{symbol}.csv.gz')
                price_dd[symbol] = parser_raw_book_snapshot(file).filter(regex='Px').mean(axis=1)
                volume_dd[symbol] = parser_raw_book_snapshot(file).filter(regex='Qty').mean(axis=1)
                os.remove(f'./binance_book_snapshot_5_{from_date_string}_{symbol}.csv.gz')
        price = pd.concat(price_dd, axis=1)
        price.index.name = 'timestamp'
        volume = pd.concat(volume_dd, axis=1)
        volume.index.name = 'timestamp'
        price.to_csv(f'./aggregate{date.year}', mode='a', header=not os.path.exists(f'./aggregate{date.year}'))
        volume.to_csv(f'./aggregate{date.year}_volume', mode='a',
                      header=not os.path.exists(f'./aggregate{date.year}_volume'))
        print(f'[FILE GENERATION]: Aggregated file for {from_date_string} has been generated.')


if __name__ == '__main__':

    for year in list(range(2024, 2026)):
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31' if year < 2025 else f'{year}-03-31'
        download_data(start=start_date, end=end_date, exchange='binance')
