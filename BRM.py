'''
The purpose of this code is to backtest a combination of the three indicators: Bollinger Bands, RSI, and MACD.
The hope is that this code will be flexible enough to make quick changes and enable sufficient research into what
method is the most succesful, if any.
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
#from Functions.Alpaca_Key_Store import initiate_API_keys
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
    def __init__(self, tickers, sell, start_date, end_date):

        #####
        self.tickers = tickers

        # Times and Dates
        self.start_date = start_date
        self.end_date = end_date
        self.list_days = []
        self.list_minutes = []

        # Data Frame Initialization
        self.df = {}
        self.in_s = {}
        self.in_e = {}

        # Price Tracking
        self.price = {}
        self.start_price = {}
        self.close_prices = {}
        self.daily_close = {}
        self.n_open = {}
        self.sell_price = {}
        self.buy_price = {}
        self.sell = sell

        # Cash
        self.cash = {}
        self.total_cash = 0

        self.Port_Change = 0
        self.bm = 0
        self.temp_bm = []

        # Indicators
        self.macd = {}
        self.macd_sig = {}
        self.ema_short = {}
        self.ema_long = {}
        self.ema_signal = {}
        self.ema = {}

        self.macd_sig_c = {}
        self.macd_hist = {}

        # Iterative Measures
        self.c = 0
        self.b = {}
        self.s = {}
        self.x = {}
        self.max = {}
        self.buy_search = {}
        self.buy_act = {}

    def df_func_new(self, day):
        for ticker in self.tickers:
            self.df["{}".format(ticker)] = False
            if os.path.exists('../../../../Data_Store/SP500_Price_Data/{}_{}.pk'.format(ticker, day)):
                filename = '../../../../Data_Store/SP500_Price_Data/{}_{}.pk'.format(ticker, day)
                with open(filename, 'rb') as file:
                    try:
                        self.df["{}".format(ticker)] = pd.DataFrame(pickle.load(file))
                    except:
                        self.df["{}".format(ticker)] = False
                continue
            else:
                self.df["{}".format(ticker)] = False

        return

    def signal_line(self, macd_sig, ticker):
        k = 2.0/(9+1)
        if len(macd_sig) == 9:
            self.ema_signal["{}".format(ticker)] = np.sum(macd_sig)/9.0
        elif len(macd_sig) > 9:
            self.ema_signal["{}".format(ticker)] = self.macd["{}".format(ticker)]*k + self.ema_signal["{}".format(ticker)]*(1-k)
        else:
            self.ema_signal["{}".format(ticker)] = np.nan
        return

    def short_ema(self, i, close_prices, price, ticker):
        k = 2.0/(12 + 1)
        if i == 12:
            self.ema_short["{}".format(ticker)] = np.sum(close_prices)/9.0
        elif i > 12:
            self.ema_short["{}".format(ticker)] = price*k + self.ema_short["{}".format(ticker)]*(1-k)
        else:
            self.ema_short["{}".format(ticker)] = np.nan
        return

    def long_ema(self, i, close_prices, price, ticker):
        k = 2.0/(26+1)
        if i == 26:
            self.ema_long["{}".format(ticker)] = np.sum(close_prices)/26.0
        elif i > 26:
            self.ema_long["{}".format(ticker)] = price*k + self.ema_long["{}".format(ticker)]*(1-k)
        else:
            self.ema_long["{}".format(ticker)] = np.nan
        return

    def macd_calc(self, ticker):
        self.macd["{}".format(ticker)] = self.ema_short["{}".format(ticker)] - self.ema_long["{}".format(ticker)]
        return

    def macd_cross(self, i, close_prices, price, macd_sig, ticker):
        self.short_ema(i, close_prices, price, ticker)
        self.long_ema(i, close_prices, price, ticker)
        self.macd_calc(ticker)
        self.signal_line(macd_sig, ticker)
        self.macd_sig_c["{}".format(ticker)] = self.ema_signal["{}".format(ticker)]

        if self.macd["{}".format(ticker)] < self.macd_sig_c["{}".format(ticker)]:
            self.buy_search["{}".format(ticker)] = True

        if self.macd["{}".format(ticker)] > self.macd_sig_c["{}".format(ticker)] and self.buy_search["{}".format(ticker)] is True:
            self.buy_act["{}".format(ticker)] = True

    def macd_cross_zero(self, i, close_prices, price, macd_sig, ticker):
        self.short_ema(i, close_prices, price, ticker)
        self.long_ema(i, close_prices, price, ticker)
        self.macd_calc(ticker)
        self.signal_line(macd_sig, ticker)
        self.macd_sig_c["{}".format(ticker)] = self.ema_signal["{}".format(ticker)]

        if self.macd["{}".format(ticker)] < self.macd_sig_c["{}".format(ticker)]:
            self.buy_search["{}".format(ticker)] = True

        if self.macd_sig_c["{}".format(ticker)] < self.macd["{}".format(ticker)] <= 0.0 and self.buy_search["{}".format(ticker)] is True:
            self.buy_act["{}".format(ticker)] = True

    def macd_hist_f(self, i, close_prices, price, macd_sig, s, ticker):
        self.short_ema(i, close_prices, price, ticker)
        self.long_ema(i, close_prices, price, ticker)
        self.macd_calc(ticker)
        self.signal_line(macd_sig, ticker)
        self.macd_sig_c["{}".format(ticker)] = self.ema_signal["{}".format(ticker)]
        self.macd_hist["{}_{}".format(i, ticker)] = self.macd["{}".format(ticker)] - self.macd_sig_c["{}".format(ticker)]

        try:
            hist_o = self.macd["{}".format(ticker)] - self.macd_sig_c["{}".format(ticker)]
            hist_t = self.macd_hist["{}_{}".format(i-1, ticker)]
            hist_th = self.macd_hist["{}_{}".format(i-2, ticker)]
        except:
            hist_o = 0.0
            hist_t = 0.0
            hist_th = 0.0

        if s == 1:
            if hist_o < hist_t:
                s = 0

        if s == 0:
            if self.macd["{}".format(ticker)] < self.macd_sig_c["{}".format(ticker)]:
                self.buy_search["{}".format(ticker)] = True

            if hist_th < hist_t < hist_o and self.buy_search["{}".format(ticker)] is True:
                self.buy_act["{}".format(ticker)] = True

        return s

    def macd_hist_zero(self, i, close_prices, price, macd_sig, s, ticker):
        self.short_ema(i, close_prices, price, ticker)
        self.long_ema(i, close_prices, price, ticker)
        self.macd_calc(ticker)
        self.signal_line(macd_sig, ticker)
        self.macd_sig_c["{}".format(ticker)] = self.ema_signal["{}".format(ticker)]
        self.macd_hist["{}_{}".format(i, ticker)] = self.macd["{}".format(ticker)] - self.macd_sig_c["{}".format(ticker)]

        try:
            hist_o = self.macd["{}".format(ticker)] - self.macd_sig_c["{}".format(ticker)]
            hist_t = self.macd_hist["{}_{}".format(i - 1, ticker)]
            hist_th = self.macd_hist["{}_{}".format(i - 2, ticker)]
            hist_f = self.macd_hist["{}_{}".format(i-3, ticker)]
        except:
            hist_o = 0.0
            hist_t = 0.0
            hist_th = 0.0
            hist_f = 0.0

        if s == 1:
            if hist_o < hist_t < hist_th:
                s = 0

        if s == 0:
            if self.macd["{}".format(ticker)] < self.macd_sig_c["{}".format(ticker)]:
                self.buy_search["{}".format(ticker)] = True

            if hist_th < hist_t < hist_o and self.buy_search["{}".format(ticker)] is True and self.macd["{}".format(ticker)] <= 0.0:
                self.buy_act["{}".format(ticker)] = True

        return s

    def sim(self, strat):
        self.total_cash = 30000.0
        self.c = 0
        day_count = 0

        w = []
        l = []
        pr = []
        br = []
        dates_i = []

        self.df_func_new(self.start_date)
        for ticker in self.tickers:
            self.cash["{}".format(ticker)] = float(self.total_cash) / (float(len(self.tickers)))
            self.close_prices["{}".format(ticker)] = []
            self.macd_sig["{}".format(ticker)] = []
            self.macd["{}".format(ticker)] = np.nan
            self.macd_sig_c["{}".format(ticker)] = np.nan
            self.macd_hist["{}".format(ticker)] = np.nan
            self.ema_short["{}".format(ticker)] = np.nan
            self.ema_long["{}".format(ticker)] = np.nan
            self.ema_signal["{}".format(ticker)] = np.nan
            self.ema["{}".format(ticker)] = np.nan
            self.b["{}".format(ticker)] = 0
            self.s["{}".format(ticker)] = 0
            self.x["{}".format(ticker)] = 0
            self.max["{}".format(ticker)] = 0
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

            print("Portfolio: {}% -- Benchmark: {}%".format("%.2f" % self.Port_Change, "%.2f" % self.bm))

            '''
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
            '''

            for period in range(0,391):
                i = (period + 1) + (day_count) * 390.0

                for ticker in self.tickers:

                    try:
                        self.price["{}".format(ticker)] = self.df["{}".format(ticker)]['close'][period]
                    except:
                        continue
                    try:
                        self.n_open["{}".format(ticker)] = self.df["{}".format(ticker)]['close'][period + 1]
                    except:
                        self.n_open["{}".format(ticker)] = self.df["{}".format(ticker)]['close'][period]

                    if self.start_price["{}".format(ticker)] is np.nan:
                        self.start_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                    else:
                        pass

                    self.close_prices["{}".format(ticker)].append(self.price["{}".format(ticker)])

                    if self.macd["{}".format(ticker)] > 0 or self.macd["{}".format(ticker)] < 0:
                        self.macd_sig["{}".format(ticker)].append(self.macd["{}".format(ticker)])

                    if self.price["{}".format(ticker)] > self.max["{}".format(ticker)]:
                        self.max["{}".format(ticker)] = self.price["{}".format(ticker)]

                    if strat == 'macd_cross':
                        self.macd_cross(i, self.close_prices["{}".format(ticker)], self.price["{}".format(ticker)],
                                        self.macd_sig["{}".format(ticker)], ticker)
                    elif strat == 'macd_cross_zero':
                        self.macd_cross_zero(i, self.close_prices["{}".format(ticker)], self.price["{}".format(ticker)],
                                             self.macd_sig["{}".format(ticker)], ticker)
                    elif strat == 'macd_hist':
                        self.s["{}".format(ticker)] = self.macd_hist_f(i, self.close_prices["{}".format(ticker)],
                                                                       self.price["{}".format(ticker)],
                                                                       self.macd_sig["{}".format(ticker)],
                                                                       self.s["{}".format(ticker)], ticker)
                    elif strat == 'macd_hist_zero':
                        self.s["{}".format(ticker)] = self.macd_hist_zero(i, self.close_prices["{}".format(ticker)],
                                                                          self.price["{}".format(ticker)],
                                                                          self.macd_sig["{}".format(ticker)],
                                                                          self.s["{}".format(ticker)], ticker)

                    if self.buy_search["{}".format(ticker)]:
                        if self.buy_act["{}".format(ticker)]:
                            if self.b["{}".format(ticker)] == 1:
                                self.c += 1
                                self.sell_price["{}".format(ticker)] = self.n_open["{}".format(ticker)]
                                pct_r = (self.sell_price["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                                self.buy_price["{}".format(ticker)]) + 1
                                if pct_r >= 1.0:
                                    w.append(pct_r)
                                else:
                                    l.append(pct_r)

                                n_cash = self.cash["{}".format(ticker)] * float(pct_r)
                                change = n_cash - self.cash["{}".format(ticker)]
                                self.total_cash += change
                            else:
                                pass

                            self.cash["{}".format(ticker)] = (float(self.total_cash)) / (float(len(self.tickers)))
                            self.buy_price["{}".format(ticker)] = self.n_open["{}".format(ticker)]
                            print("Buy at: {}".format(self.df["{}".format(ticker)]['timestamp'][period]))
                            print("Price: {}".format(self.price["{}".format(ticker)]))
                            self.max["{}".format(ticker)] = self.price["{}".format(ticker)]
                            self.buy_search["{}".format(ticker)] = False
                            self.buy_act["{}".format(ticker)] = False
                            self.b["{}".format(ticker)] = 1
                            self.s["{}".format(ticker)] = 1
                            self.x["{}".format(ticker)] = 0
                            continue
                        else:
                            continue

                    if self.buy_act["{}".format(ticker)] is False:
                        if self.b["{}".format(ticker)] == 1:
                            pct_change = (self.price["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                            self.buy_price["{}".format(ticker)]) * 100.0
                            max_change = (self.price["{}".format(ticker)] - self.max["{}".format(ticker)]) / (
                            self.max["{}".format(ticker)]) * 100.0

                            if pct_change > self.sell:
                                self.sell_price["{}".format(ticker)] = self.n_open["{}".format(ticker)]
                                pct_r = (self.sell_price["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                                self.buy_price["{}".format(ticker)]) + 1
                                if pct_r > 1.0:
                                    w.append(pct_r)
                                else:
                                    l.append(pct_r)

                                n_cash = self.cash["{}".format(ticker)] * float(pct_r)
                                change = n_cash - self.cash["{}".format(ticker)]
                                self.total_cash += change
                                self.b["{}".format(ticker)] = 0
                                self.s["{}".format(ticker)] = 0
                                continue
                            else:
                                continue
                    else:
                        continue

            s_today = str(dt.datetime.fromtimestamp(today) + dt.timedelta(days=1))[0:10]
            if s_today == self.end_date:
                end_date = True
            else:
                day_count += 1
                continue

        # plt.show()
        for ticker in self.tickers:
            if self.buy_act["{}".format(ticker)] is False:
                if self.b["{}".format(ticker)] == 1:
                    self.c += 1
                    self.sell_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                    pct_r = (self.sell_price["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                    self.buy_price["{}".format(ticker)]) + 1
                    if pct_r >= 1.0:
                        w.append(pct_r)
                    else:
                        l.append(pct_r)
                    self.b["{}".format(ticker)] = 0
                    n_cash = self.cash["{}".format(ticker)] * float(pct_r)
                    change = n_cash - self.cash["{}".format(ticker)]
                    self.total_cash += change

        # End of loop.
        self.Port_Change = (self.total_cash - 30000.0) / (30000.0) * 100
        print("Total Change: " + "%.3f" % self.Port_Change + "%.")

        temp_bm = []
        for ticker in self.tickers:
            try:
                temp_bm.append((self.price["{}".format(ticker)] - self.start_price["{}".format(ticker)]) / (
                    self.start_price["{}".format(ticker)]) * 100.0)
            except:
                temp_bm.append(np.nan)

        self.bm = np.nanmean(temp_bm)
        print("Benchmark Change: " + "%.3f" % self.bm + "%.")

        return self.total_cash, w, l, self.c, self.bm

    def run_sim(self, strat):

        cash, w, l, c, bm = self.sim(str(strat))

        return cash, w, l, c, bm


now = dt.datetime.now()
'''
year = 2019
smonth = 2
mrange = 2
gen = gen_tickers(year, smonth, mrange)
tickers = gen.sp500_grab(100)
'''

tickers = ['GE']

test = Indicators(tickers, 0.50, '2019-06-10', '2019-06-17')
test.run_sim('macd_hist_zero')
later = dt.datetime.now()
diff = later - now
print(diff)




