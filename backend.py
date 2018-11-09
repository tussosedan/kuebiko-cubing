from datetime import datetime, timedelta
from math import floor, ceil, isnan

from numba import jit
from numpy import sort, repeat, NaN, array
from pandas import read_csv, to_datetime, concat, notnull, DataFrame

from plotly.offline import plot
import plotly.graph_objs as go

import json

from collections import OrderedDict


@jit
def binary_search_2d(a, x, size):
    lo = 0
    hi = size
    while lo < hi:
        mid = (lo + hi) // 2
        if (a[mid]['IsDNF'] < x['IsDNF']) \
                or ((a[mid]['IsDNF'] == x['IsDNF']) and (a[mid]['TimeCentiSec'] < x['TimeCentiSec'])):
            lo = mid + 1
        else:
            hi = mid
    return lo


@jit
def round_half_up(n):
    if n - floor(n) < 0.5:
        return floor(n)
    return ceil(n)


@jit
def rolling_trimmed_mean(data, window_size, outliers_to_trim):
    x = sort(data[:window_size], order=['IsDNF', 'TimeCentiSec'])
    means = repeat(NaN, len(data))

    for i in range(window_size, len(data) + 1):
        if x[window_size - outliers_to_trim - 1]['IsDNF'] == 1:
            means[i - 1] = NaN
        else:
            means[i - 1] = round_half_up(x[outliers_to_trim: window_size - outliers_to_trim]['TimeCentiSec'].mean())

        if i != len(data):
            idx_old = binary_search_2d(x, data[i - window_size], window_size)
            idx_new = binary_search_2d(x, data[i], window_size)
            if idx_old < idx_new:
                x[idx_old:idx_new - 1] = x[idx_old + 1:idx_new]
                x[idx_new - 1] = data[i]
            elif idx_new < idx_old:
                x[idx_new + 1:idx_old + 1] = x[idx_new:idx_old]
                x[idx_new] = data[i]
            else:
                x[idx_new] = data[i]

    return means


def calculate_and_store_running_ao(solves_data, puzzle, category, ao_len):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)][[
        'TimeCentiSec', 'IsDNF']]
    solves_data_part_array = array(solves_data_part.to_records())

    if ao_len == 3:
        outliers_to_trim = 0
        prefix = 'mo'
    else:
        outliers_to_trim = ceil(ao_len * (5 / 100))
        prefix = 'ao'

    means = rolling_trimmed_mean(solves_data_part_array, ao_len, outliers_to_trim)

    solves_data.loc[solves_data_part.index, prefix + str(ao_len)] = means / 100


def sec2dtstr(seconds):
    if isnan(seconds):
        return seconds

    s = str(timedelta(seconds=seconds))

    if s[:6] == '0:00:0':
        s = s[6:]
    elif s[:5] == '0:00:':
        s = s[5:]
    elif s[:3] == '0:0':
        s = s[3:]
    elif s[:2] == '0:':
        s = s[2:]

    if s[-4:] == '0000':
        s = s[:-4]

    return s


def sec2dt(seconds):
    if isnan(seconds):
        return seconds
    s = '1970-01-01 ' + str(timedelta(seconds=seconds))
    return s


def get_pb_progression(solves_data, puzzle, category, col_name, has_dates):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    # create a column for solve num
    solves_data_part.reset_index(inplace=True, drop=True)
    solves_data_part.index += 1
    solves_data_part.reset_index(inplace=True)

    if has_dates:
        solves_data_pb = solves_data_part[(solves_data_part[col_name + '_cummin'].diff() != 0)
                                          & (solves_data_part[col_name + '_cummin'].notnull())][
            ['index', 'DatetimeUTC', col_name]]

        solves_data_pb['PB For Time'] = solves_data_pb['DatetimeUTC'].diff(periods=-1) * -1

        last_date = solves_data_pb.tail(1)['DatetimeUTC']
        solves_data_pb.loc[last_date.index, 'PB For Time'] = str(
            datetime.now().replace(microsecond=0) - last_date.iloc[0]) + ' and counting'

        solves_data_pb.rename(inplace=True, columns={"DatetimeUTC": "Date & Time [UTC]"})
    else:
        solves_data_pb = solves_data_part[(solves_data_part[col_name + '_cummin'].diff() != 0)
                                          & (solves_data_part[col_name + '_cummin'].notnull())][
            ['index', col_name]]

    solves_data_pb['PB For # Solves'] = (solves_data_pb['index'].diff(periods=-1) * -1).dropna().apply(
        lambda x: str(int(x)))

    last_index = solves_data_pb.tail(1)['index']
    solves_data_pb.loc[last_index.index, 'PB For # Solves'] = str(
        solves_data_part.iloc[-1]['index'] - last_index.iloc[0]) + ' and counting'

    solves_data_pb['Result'] = solves_data_pb[col_name].apply(sec2dtstr)
    solves_data_pb.rename(inplace=True, columns={"index": "Solve #"})

    return solves_data_pb


def get_all_solves_details(solves_data, has_dates):
    # generate a nested dict puzzle -> category -> pb progression df
    puzcats = {k: g['Category'].tolist()
               for k, g in solves_data[['Puzzle', 'Category']].drop_duplicates().groupby('Puzzle')}

    resdict = OrderedDict()
    for puz, cats in puzcats.items():
        catdict = OrderedDict()
        for cat in cats:
            pbs = get_pb_progression(solves_data, puz, cat, 'single', has_dates).iloc[::-1]
            if has_dates:
                pbs_display = pbs[['Result', 'Date & Time [UTC]', 'PB For Time', 'Solve #', 'PB For # Solves']]
            else:
                pbs_display = pbs[['Result', 'Solve #', 'PB For # Solves']]
            catdict[cat] = pbs_display, get_solves_plot(solves_data, puz, cat, has_dates)
        resdict[puz] = catdict

    return resdict


def get_overall_pbs(solves_data):
    pbs = solves_data[solves_data['Penalty'] != 2][
        ['Puzzle', 'Category', 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000']]. \
        groupby(['Puzzle', 'Category']).min().applymap(sec2dtstr)

    counts = solves_data.groupby(['Puzzle', 'Category']).size().rename('#')

    pbs_with_count = concat([counts, pbs], axis=1)

    if (pbs_with_count.index.levels[0] == 'Sessions').all():
        pbs_with_count.index = pbs_with_count.index.droplevel(level=0)
        pbs_with_count.rename_axis(None, inplace=True)

    return pbs_with_count


def get_solves_plot(solves_data, puzzle, category, has_dates):
    plot_data = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]

    if has_dates:
        data_plot_x = plot_data['DatetimeUTC']
    else:
        plot_data.reset_index(inplace=True, drop=True)
        data_plot_x = plot_data.index + 1

    data = []
    colors = {'single': 'red',
              'mo3': 'rgba(55, 128, 191, 1.0)',
              'ao5': 'rgba(50, 171, 96, 1.0)',
              'ao12': 'rgba(128, 0, 128, 1.0)',
              'ao50': 'rgba(219, 64, 82, 1.0)',
              'ao100': 'rgba(0, 128, 128, 1.0)',
              'ao1000': 'blue'}

    for series in 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000':
        if not plot_data[series].isnull().all():
            marker = {'size': 2, 'color': 'red'}

            if series == 'single':
                mode = 'markers'
            else:
                mode = 'lines'

            data.append(go.Scatter(
                x=data_plot_x,
                y=plot_data[series + '_dt'],
                name=series,
                text=plot_data[series + '_str'],
                hoverinfo="x+name+text",
                mode=mode,
                marker=marker,
                visible='legendonly',
                line={'color': colors.get(series, 'black'),
                      'width': 1.3}
            ))

            pbs = get_pb_progression(solves_data, puzzle, category, series, has_dates)

            if has_dates:
                pb_plot_x = pbs['Date & Time [UTC]']
            else:
                pb_plot_x = pbs['Solve #']

            data.append(go.Scatter(
                x=pb_plot_x,
                y=pbs[series].apply(lambda x: to_datetime(x, unit='s')),
                name='PB ' + series,
                text=pbs['Result'],
                hoverinfo="x+name+text",
                mode="lines+markers",
                marker={'size': 4,
                        'color': colors.get(series, 'black')},
                visible='legendonly',
                # fill='tozeroy',
                line={'color': colors.get(series, 'black'),
                      'width': 1.3,
                      'dash': 'dash',
                      'shape': 'vh'}
            ))

    layout = go.Layout(margin={'l': 50,
                               'r': 50,
                               'b': 50,
                               't': 50,
                               'pad': 4
                               },
                       yaxis={'tickformat': '%M:%S'})

    fig = dict(data=data, layout=layout)

    # display only the singles and the two largest aoX
    fig['data'][0].visible = True
    if len(fig['data']) >= 3:
        fig['data'][-2].visible = True
    if len(fig['data']) >= 5:
        fig['data'][-4].visible = True

    return plot(fig, include_plotlyjs=False, output_type='div')


def create_dataframe(file):
    file.seek(0)
    headers = file.readline().decode()
    file.seek(0)

    if headers.startswith('Puzzle,Category,Time(millis),Date(millis),Scramble,Penalty,Comment'):
        # TwistyTimer
        df = read_csv(file.stream, sep=';', skiprows=1, header=None)
        df.columns = headers.strip().split(sep=',')
        has_dates = True
        return df.rename_axis('MyIdx').sort_values(['Date(millis)', 'MyIdx']), has_dates
    elif headers.startswith('{"session1"'):
        # cstimer
        data = json.load(file)
        solves = []
        for session_id, session_values in json.loads(json.loads(data['properties'])['sessionData']).items():
            for times, scramble, notes in json.loads(data['session' + session_id]):
                if times[0] == 2000:
                    final_time = times[1] + 2000
                else:
                    final_time = times[1]

                if times[0] == 2000:
                    penalty = 1
                elif times[0] == -1:
                    penalty = 2
                else:
                    penalty = 0

                solves.append([session_values['name'], final_time, penalty])

        df = DataFrame(data=solves, columns=['Category', 'Time(millis)', 'Penalty'])
        has_dates = False
        df['Puzzle'] = 'Sessions'
        return df, has_dates
    else:
        raise NotImplementedError('Unrecognized file type')


def process_data(file):
    solves_data, has_dates = create_dataframe(file)

    if has_dates:
        solves_data['DatetimeUTC'] = to_datetime(solves_data['Date(millis)'], unit='ms').astype('datetime64[s]')

    solves_data['TimeCentiSec'] = (solves_data['Time(millis)'] / 10).astype(int)

    solves_data['single'] = [NaN if row.Penalty == 2 else (row.TimeCentiSec / 100)
                             for row in solves_data[['Penalty', 'TimeCentiSec']].itertuples()]

    solves_data['IsDNF'] = [1 if row == 2 else 0 for row in solves_data['Penalty']]

    for idx, puzzle, category in solves_data[['Puzzle', 'Category']].drop_duplicates().itertuples():
        for ao_len in (3, 5, 12, 50, 100, 1000):
            calculate_and_store_running_ao(solves_data, puzzle, category, ao_len)

    for series in 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000':
        solves_data[series + '_str'] = solves_data[series][notnull(solves_data[series])].apply(sec2dtstr)
        solves_data[series + '_dt'] = solves_data[series].apply(sec2dt)

    solves_data[
        ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
         'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']] = \
        solves_data[solves_data['Penalty'] != 2].groupby(['Puzzle', 'Category']).cummin()[
            ['single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000']]

    solves_data[
        ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
         'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']] = \
        solves_data.groupby(['Puzzle', 'Category'])[
            ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
             'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']].fillna(method='ffill')

    return get_all_solves_details(solves_data, has_dates), get_overall_pbs(solves_data)

# TODO add histogram of solves with dynamic time/index range filters?
