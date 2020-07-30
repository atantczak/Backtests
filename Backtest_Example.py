'''
Andrew Antczak
theoreticallyprofitable.com

The purpose of this code is to illustrate the general powers that a backtest provides and provide insight on how
to construct a backtest.

This is not a proposed strategy! In fact I would bet against this strategy.
'''

import os
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import style
import alpaca_trade_api as tradeapi
import numpy as np
import random as rand
import pandas as pd
import time
import pickle
import multiprocessing as mp
from multiprocessing import Lock, Value
from dateutil.parser import parse
import time as tm
from Alpaca_Key_Store import initiate_API_keys
from Ticker_Select import gen_tickers
# ---------------------------------------------------------------------------------------------------------------------#
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL = initiate_API_keys()
ALPACA_PAPER = True
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL, 'v2')
# ---------------------------------------------------------------------------------------------------------------------#
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

style.use('seaborn')


class Indicators():
    def __init__(self, tickers, start_date, end_date):

        #####
        self.tickers = tickers

        # Times and Dates
        self.start_date = start_date
        self.end_date = end_date

        # Data Frame Initialization
        self.df = {}
        self.in_s = {}
        self.in_e = {}

        # Price Tracking
        self.price = {}
        self.start_price = {}
        self.close_prices = {}
        self.sell_price = {}
        self.buy_price = {}
        self.sell = 0
        self.max_change = {}
        self.max = {}

        # Cash
        self.cash = {}
        self.total_cash = 0

        self.Port_Change = 0
        self.bm = 0
        self.temp_bm = []

        # Indicators
        self.moving_avg = {}

        # Iterative Measures
        self.b = {}

    def df_func_new(self, day):
        for ticker in self.tickers:
            self.df["{}".format(ticker)] = False
            if os.path.exists('../../Quant_System/Data_Store/SP500_Price_Data/{}_{}.pk'.format(ticker, day)):
                filename = '../../Quant_System/Data_Store/SP500_Price_Data/{}_{}.pk'.format(ticker, day)
                with open(filename, 'rb') as file:
                    try:
                        self.df["{}".format(ticker)] = pd.DataFrame(pickle.load(file))
                    except:
                        self.df["{}".format(ticker)] = False
                continue
            else:
                self.df["{}".format(ticker)] = False

        return

    def sim(self):
        self.total_cash = 30000.0
        day_count = 0
        wc = 0
        lc = 0
        w_avg = 0
        l_avg = 0
        pr = []
        br = []
        dates_i = []

        self.df_func_new(self.start_date)

        for ticker in self.tickers:
            self.cash["{}".format(ticker)] = float(self.total_cash) / (float(len(self.tickers)))
            self.close_prices["{}".format(ticker)] = []
            self.sell_price["{}".format(ticker)] = np.nan
            self.b["{}".format(ticker)] = 0
            self.max["{}".format(ticker)] = 0
            self.max_change["{}".format(ticker)] = 0
            self.buy_search["{}".format(ticker)] = False
            self.buy_act["{}".format(ticker)] = False
            try:
                self.price["{}".format(ticker)] = self.df["{}".format(ticker)]['open'][0]
                self.start_price["{}".format(ticker)] = self.price["{}".format(ticker)]
            except:
                self.start_price["{}".format(ticker)] = np.nan
                continue

        today = time.mktime(dt.datetime.strptime(self.start_date, "%Y-%m-%d").timetuple())
        s_today = str(dt.datetime.fromtimestamp(today))[0:10]
        end_date = False

        while end_date is False:
            today = s_today
            print(s_today)
            today = time.mktime(dt.datetime.strptime(today, "%Y-%m-%d").timetuple())
            self.df_func_new(s_today)

            self.Port_Change = (self.total_cash - 30000.0) / (30000.0) * 100
            for ticker in self.tickers:
                try:
                    self.temp_bm.append((self.price["{}".format(ticker)] - self.start_price["{}".format(ticker)])/(
                        self.start_price["{}".format(ticker)]) * 100.0)
                except:
                    self.temp_bm.append(np.nan)

            self.bm = np.nanmean(self.temp_bm)
            self.temp_bm = []

            pr.append(self.Port_Change)
            br.append(self.bm)
            dates_i.append(s_today)
            plt.cla()
            plt.plot(dates_i, pr, label='Portfolio Return')
            plt.plot(dates_i, br, label='Benchmark Return')
            x_ticks = np.arange(0, len(dates_i), 10)
            plt.xticks(x_ticks)
            plt.xlabel('Date')
            plt.ylabel('Percent Return')
            plt.title('Equity Curve')
            plt.legend()
            plt.pause(1)

            for ticker in self.tickers:
                try:
                    self.price["{}".format(ticker)] = self.df["{}".format(ticker)]['close'].iloc[-1]
                except:
                    continue

                if self.price["{}".format(ticker)] > self.max["{}".format(ticker)]:
                    self.max["{}".format(ticker)] = self.price["{}".format(ticker)]

                self.close_prices["{}".format(ticker)].append(self.price["{}".format(ticker)])
                self.close_prices["{}".format(ticker)] = self.close_prices["{}".format(ticker)][-10:]

                self.moving_avg["{}".format(ticker)] = np.nanmean(self.close_prices["{}".format(ticker)])

                if self.price["{}".format(ticker)] < self.moving_avg["{}".format(ticker)] and self.b["{}".format(ticker)] == 0:
                    self.cash["{}".format(ticker)] = (float(self.total_cash))/(float(len(self.tickers)))
                    self.buy_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                    self.max["{}".format(ticker)] = self.price["{}".format(ticker)]
                    self.b["{}".format(ticker)] = 1
                else:
                    pass

                if self.b["{}".format(ticker)] == 1:
                    self.max_change["{}".format(ticker)] = (self.price["{}".format(ticker)] -
                                                            self.max["{}".format(ticker)])/\
                                                           (self.max["{}".format(ticker)]) * 100.0

                    if self.max_change["{}".format(ticker)] < self.sell:
                        self.sell_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                        pct_r = (self.sell_price["{}".format(ticker)] - self.buy_price["{}".format(ticker)])/(self.buy_price["{}".format(ticker)]) + 1
                        n_cash = self.cash["{}".format(ticker)] * float(pct_r)
                        change = n_cash - self.cash["{}".format(ticker)]
                        self.total_cash += change
                        if pct_r > 1.0:
                            pct_r = (pct_r - 1) * 100.0
                            w_avg = (w_avg * wc + pct_r) / float(wc + 1)
                            wc+=1
                        else:
                            pct_r = (pct_r - 1) * 100.0
                            l_avg = (l_avg * lc + pct_r)/ float(lc+1)
                            lc+=1

                        self.b["{}".format(ticker)] = 0
                    else:
                        pass
                else:
                    pass

            s_today = str(dt.datetime.fromtimestamp(today) + dt.timedelta(days=1))[0:10]
            if s_today == self.end_date:
                plt.savefig('Equity_Curve.png', dpi=1000)
                end_date = True
            else:
                day_count += 1
                continue

        for ticker in self.tickers:
            if self.b["{}".format(ticker)] == 1:
                self.sell_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                pct_r = (self.sell_price["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                self.buy_price["{}".format(ticker)]) + 1
                n_cash = self.cash["{}".format(ticker)] * float(pct_r)
                change = n_cash - self.cash["{}".format(ticker)]
                self.total_cash += change
                if pct_r > 1.0:
                    pct_r = (pct_r - 1) * 100.0
                    w_avg = (w_avg * wc + pct_r) / float(wc + 1)
                    wc += 1
                else:
                    pct_r = (pct_r - 1) * 100.0
                    l_avg = (l_avg * lc + pct_r) / float(lc + 1)
                    lc += 1

                self.b["{}".format(ticker)] = 0
            else:
                continue

        # End of loop.
        self.Port_Change = (self.total_cash - 30000.0) / (30000.0) * 100

        temp_bm = []
        for ticker in self.tickers:
            try:
                temp_bm.append((self.price["{}".format(ticker)] - self.start_price["{}".format(ticker)]) / (
                    self.start_price["{}".format(ticker)]) * 100.0)
            except:
                temp_bm.append(np.nan)

        self.bm = np.nanmean(temp_bm)

        w_pct = (float(wc))/(float(wc+lc)) * 100.0

        return self.Port_Change, w_pct, w_avg, l_avg, self.bm

    def run_sim(self):

        now = dt.datetime.now()
        
        self.sell = -0.05

        port, w_pct, w_avg, l_avg, bm = self.sim()

        print("Portfolio Return was {}%.".format("%.2f" % port))
        print("Benchmark Return was {}%.".format("%.2f" % bm))

        print("Winning Percentage: {}".format("%.2f" % w_pct))

        print("Average Win was {}%.".format("%.2f" % w_avg))
        print("Average Loss was {}%.".format("%.2f" % l_avg))

        later = dt.datetime.now()
        diff = later - now
        print("This sim took: {}".format(str(diff)))

        return


year = 2020
smonth = 1
mrange = 1
gen = gen_tickers(year, smonth, mrange)
tickers = gen.sp500_grab(30)

ind_env = Indicators(tickers, '2020-01-02', '2020-07-16')
ind_env.run_sim()


