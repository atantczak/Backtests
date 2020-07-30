'''
This program has two functions within it. One simply pulls the tickers from a given year and sector. The other applies
whatever filter you want and returns the tickers that have met that filters specifications.
'''

import datetime as dt
import os
from datetime import timedelta

import numpy as np
from pandas.plotting import register_matplotlib_converters
from scipy import stats

register_matplotlib_converters()
import pandas as pd
import csv

# ----------------------------------------------------------------------------------------------------------------------
# SETTING UP IEX CLOUD API ENVIRONMENT

#REAL IEX_TOKEN
IEX_TOKEN = "***********"

'''
IF BELOW IS GRAY THEN YOU'RE USING THE REAL TOKEN FROM ABOVE.
'''
# Sandbox Testing API Token
#IEX_TOKEN = "************"

# This indicates that is is sandbox testing so iex cloud can access. Comment out if you're using for non-testing.
#os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
# ----------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def ticker_selection(Year, Sector):
    tick_pull = os.listdir('../../Data_Store/Russell_OHLCV_Data/{}/'.format(Year))

    tickers = []
    for ticker in tick_pull:
        tickers.append(os.path.splitext(ticker)[0])

    with open('../../Data_Store/Russell_Additions/{}_Adds.csv'.format(Year)) as csv_file:
        data = csv.reader(csv_file, delimiter=",")

        temp_tick = []
        temp_sector = []

        for row in data:
            temp_tick.append(row[5])
            temp_sector.append(row[9])

            temp_tick = [x.strip(' ') for x in temp_tick]
            temp_sector = [x.strip(' ') for x in temp_sector]

        while ("" in tickers):
            temp_tick.remove("")

    t_sector = pd.DataFrame({'Sector': temp_sector}, columns=['Sector'], index=temp_tick)

    sector = []
    for ticker in tickers:
        sector.append(t_sector.loc[ticker][0])

    tickers_tech = []
    tickers_hc = []
    tickers_fs = []
    tickers_durables = []
    tickers_staples = []
    tickers_mat_proc = []
    tickers_disc = []
    tickers_energy = []
    tickers_utility = []

    for i in range(0, len(sector)):
        if (sector[i] == 'Technology'):
            tickers_tech.append(tickers[i])
        elif (sector[i] == 'Health Care'):
            tickers_hc.append(tickers[i])
        elif (sector[i] == 'Financial Services'):
            tickers_fs.append(tickers[i])
        elif (sector[i] == 'Producer Durables'):
            tickers_durables.append(tickers[i])
        elif (sector[i] == 'Consumer Staples'):
            tickers_staples.append(tickers[i])
        elif (sector[i] == 'Materials & Processing'):
            tickers_mat_proc.append(tickers[i])
        elif (sector[i] == 'Consumer Discretionary'):
            tickers_disc.append(tickers[i])
        elif (sector[i] == 'Energy'):
            tickers_energy.append(tickers[i])
        elif (sector[i] == 'Utilities'):
            tickers_utility.append(tickers[i])

    if (Sector == 'Technology'):
        tickers_s = tickers_tech
    elif (Sector == 'Health Care'):
        tickers_s = tickers_hc
    elif (Sector == 'Financial Services'):
        tickers_s = tickers_fs
    elif (Sector == 'Producer Durables'):
        tickers_s = tickers_durables
    elif (Sector == 'Consumer Staples'):
        tickers_s = tickers_staples
    elif (Sector == 'Materials & Processing'):
        tickers_s = tickers_mat_proc
    elif (Sector == 'Consumer Discretionary'):
        tickers_s = tickers_disc
    elif (Sector == 'Energy'):
        tickers_s = tickers_energy
    elif (Sector == 'Utilities'):
        tickers_s = tickers_utility
    elif (Sector == 'All'):
        tickers_s = tickers

    return tickers_s


def sift_strategy(sift, Year, smonth, period, z, tickers):
    if (sift == 'raw'):
        return tickers

    momentum = []
    volume_avg = []
    tickers_f = []
    for ticker in tickers:
        try:
            with open('../../Data_Store/Russell_OHLCV_Data/{}/{}.csv'.format(Year, ticker)) as csv_file:
                data = csv.reader(csv_file, delimiter=",")
                data = pd.DataFrame(data)
                new_header = data.iloc[0]
                df = data[1:]
                df.columns = new_header

            # This section is setting the period start index for this ticker.
            weekend_test = dt.date(year=int(Year), month=int(smonth), day=1).weekday()
            analysis_end = dt.date(year=int(Year), month=int(smonth), day=1)
            # ---------------------------------------------------------------
            if (weekend_test == 5):
                analysis_end = analysis_end + timedelta(days=2)
                analysis_start = analysis_end - timedelta(days=period)
            elif (weekend_test == 6):
                analysis_end = analysis_end + timedelta(days=1)
                analysis_start = analysis_end - timedelta(days=period)
            else:
                analysis_end = analysis_end
                analysis_start = analysis_end - timedelta(days=period)

            analysis_end = df[df['Date'] == str(analysis_end)].index.values[0]
            analysis_start = df[df['Date'] == str(analysis_start)].index.values[0]

            tickers_f.append(ticker)

        except (IndexError) as e:
            continue

        if (sift == 'mom'):
            momentum.append((float(df['Close'][analysis_end]) -
                             float(df['Close'][analysis_start]))/(float(df['Close'][analysis_start])))

        elif (sift == 'dva'):
            df = df.truncate(before=analysis_start, after=analysis_end)

            df = np.array(df['Volume'])
            for i in range(0, len(df)):
                df[i] = np.float(df[i])

            volume = df.sum()
            volume_avg.append(volume)

    if (sift == 'mom'):
        df = pd.DataFrame({'Momentum': momentum}, columns=['Momentum'], index=tickers_f)

        total = df['Momentum']

    elif (sift == 'dva'):
        df = pd.DataFrame({'Daily Volume Average': volume_avg}, columns=['Daily Volume Average'], index=tickers_f)

        total = df['Daily Volume Average']

    # Getting rid of negative outliers.
    z_scores = stats.zscore(total)
    filtered_entries = (z_scores > -3)

    # Filtering for 50th percentile.
    total = total[filtered_entries]
    z_scores = stats.zscore(total)
    filtered_entries = (z_scores > z)

    N_total = total[filtered_entries]

    tickers_F = []
    tickers_F.append(N_total.index.get_level_values(0).values)

    final_tickers = []
    df2 = pd.DataFrame(tickers_F, index=None).loc[0]
    for i in range(0,len(df2)):
        final_tickers.append(df2.loc[i])

    return final_tickers
