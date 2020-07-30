# ---------------------------------------------------------------------------------------------------------------------#
'''
This code is to break up a set of large tickers and run them simultaneously through your desired backtest and then
compile those results to see what your portfolio return would have been given that you didn't rebalance throughout the
time period.
'''
# ---------------------------------------------------------------------------------------------------------------------#


import csv
import datetime as dt
import pickle
import random
from datetime import timedelta

import alpaca_trade_api as tradeapi
import bs4 as bs
import pandas as pd
import requests
'''
from Functions.Alpaca_Key_Store import initiate_API_keys
from Functions.Stock_Filter import sift_strategy
from Functions.Stock_Filter import ticker_selection
'''

from Alpaca_Key_Store import initiate_API_keys
from Stock_Filter import sift_strategy
from Stock_Filter import ticker_selection

# ---------------------------------------------------------------------------------------------------------------------#
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL = initiate_API_keys()
ALPACA_PAPER = True
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL, 'v2')
# ---------------------------------------------------------------------------------------------------------------------#

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class gen_tickers():
    def __init__(self, year, smonth, mrange):
        self.year = year
        self.smonth = smonth
        # Defining start and end dates
        weekend_test = dt.date(year=int(year), month=int(smonth), day=3).weekday()
        period_start = dt.date(year=int(year), month=int(smonth), day=3)
        # ---------------------------------------------------------------

        if (weekend_test == 5):
            self.simstart = period_start + timedelta(days=2)
            self.start = period_start - timedelta(days=22)
        elif (weekend_test == 6):
            self.simstart = period_start + timedelta(days=1)
            self.start = period_start - timedelta(days=22)
        else:
            self.simstart = period_start
            self.start = period_start - timedelta(days=22)

        weekend_test = dt.date(year=int(year), month=int(smonth + mrange), day=3).weekday()
        period_end = dt.date(year=int(year), month=int(smonth + mrange), day=3)

        if (weekend_test == 5):
            self.end = period_end + timedelta(days=2)
        elif (weekend_test == 6):
            self.end = period_end + timedelta(days=1)
        else:
            self.end = period_end

    def sp500_grab(self, size):
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, "lxml")
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers_trial = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            mapping = str.maketrans(".", "-")
            ticker = ticker.translate(mapping)
            tickers_trial.append(ticker)

        with open("sp500tickers.pickle", "wb") as f:
            pickle.dump(tickers_trial, f)

        tickers = []
        while int(len(tickers)) < int(size):
            ticker = random.choices(tickers_trial, k=1)
            try:
                data = api.polygon.historic_agg_v2(str(ticker[0]), 1, timespan='day', _from=str(self.simstart), to=str(self.end)).df
                df = data.reset_index()
                tickers.append(str(ticker[0]))
            except:
                continue

        return tickers

    def russell_grab(self, sector, strat, period, z):
        tickers = ticker_selection(self.year, sector)
        tickers = sift_strategy(strat, self.year, self.smonth, period, z, tickers)

        return tickers

    def main_exchanges(self, size):

        with open('Stock_Exchange_Lists/NYSE.csv') as csv_file:
            data_1 = csv.reader(csv_file, delimiter=",")

            companies_1 = []
            tickers_1 = []
            sector_1 = []

            for row in data_1:
                tickers_1.append(row[0])
                sector_1.append(row[5])

                tickers_1 = [x.strip(' ') for x in tickers_1]
                sector_1 = [x.strip(' ') for x in sector_1]

            while ("" in tickers_1):
                tickers_1.remove("")

        del tickers_1[0]
        del sector_1[0]

        with open('Stock_Exchange_Lists/NASDAQ.csv') as csv_file:
            data_2 = csv.reader(csv_file, delimiter=",")

            companies_2 = []
            tickers_2 = []
            sector_2 = []

            for row in data_2:
                tickers_2.append(row[0])
                sector_2.append(row[5])

                tickers_2 = [x.strip(' ') for x in tickers_2]
                sector_2 = [x.strip(' ') for x in sector_2]

            while ("" in tickers_2):
                tickers_2.remove("")

        del tickers_2[0]
        del sector_2[0]

        # Compile all tickers from each stock exchange of interest.
        total_tickers = []
        total_tickers.extend(tickers_1)
        total_tickers.extend(tickers_2)

        # Deleting any duplicates from those exchanges. Successful.
        total_tickers = list(dict.fromkeys(total_tickers))

        tickers = []
        while int(len(tickers)) < int(size):
            ticker = random.choices(total_tickers, k=1)
            try:
                data = api.polygon.historic_agg_v2(str(ticker[0]), 1, timespan='day', _from=self.simstart, to=self.end).df
                df = data.reset_index()
                tickers.append(str(ticker[0]))
            except:
                continue

        return tickers



