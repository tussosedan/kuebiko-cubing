from datetime import datetime, timedelta
from math import floor, ceil, isnan

from numba import jit
from numpy import sort, repeat, NaN, array, all, select
from pandas import read_csv, to_datetime, concat, notnull, DataFrame, Series, cut, set_option, melt, merge

from plotly.offline import plot
import plotly.graph_objs as go

import json
import re
import dateparser
from html import escape

from collections import OrderedDict

import zipfile
from io import BytesIO

import os
WCA_DATA_FOLDER = r'C:\downloads'


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
    rsd = repeat(NaN, len(data))

    if outliers_to_trim >= window_size - outliers_to_trim:
        return means, rsd

    for i in range(window_size, len(data) + 1):
        if x[window_size - outliers_to_trim - 1]['IsDNF'] == 1:
            means[i - 1] = NaN
        else:
            counting_solves = x[outliers_to_trim: window_size - outliers_to_trim]['TimeCentiSec']
            counting_solves_mean = counting_solves.mean()
            means[i - 1] = round_half_up(counting_solves_mean)
            # consistency score = mean / stdev (both trimmed)
            rsd_std = counting_solves.std()
            if rsd_std > 0:
                rsd[i - 1] = round(rsd_std / counting_solves_mean, 3)
            # case can happen with large trim resulting in single solve
            elif rsd_std == 0:
                rsd[i - 1] = 0

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

    return means, rsd


def calculate_and_store_running_ao(solves_data, puzzle, category, ao_len, trim_percentage):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)][[
        'TimeCentiSec', 'IsDNF']]
    solves_data_part_array = array(solves_data_part.to_records())

    if ao_len == 3:
        outliers_to_trim = 0
        prefix = 'mo'
    else:
        outliers_to_trim = ceil(ao_len * (trim_percentage / 100))
        prefix = 'ao'

    means, rsd = rolling_trimmed_mean(solves_data_part_array, ao_len, outliers_to_trim)

    solves_data.loc[solves_data_part.index, prefix + str(ao_len)] = means / 100
    solves_data.loc[solves_data_part.index, 'rsd' + str(ao_len)] = rsd


def get_subx_threshold(solves_data, puzzle, category):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)][[
        'single', 'ao100', 'ao1000']]

    last_notnull_index = solves_data_part['ao1000'].last_valid_index()
    if notnull(last_notnull_index):
        subx_base = solves_data.loc[last_notnull_index, 'ao1000']
    else:
        last_notnull_index = solves_data_part['ao100'].last_valid_index()
        if notnull(last_notnull_index):
            subx_base = solves_data.loc[last_notnull_index, 'ao100']
        else:
            subx_base = solves_data_part['single'].median()
        if isnan(subx_base):
            subx_base = 20

    subx = int(subx_base / 5) * 5  # round down to nearest multiple of 5
    if 6 < subx_base <= 10:
        subx = int(subx_base) - 1  # the going gets tough below sub-10

    if subx == 0:
        subx = 0.5

    return subx


# noinspection PyUnresolvedReferences
def calculate_and_store_running_subx(solves_data, puzzle, category, ao_len, subx_threshold):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)][[
        'single']]

    solves_data.loc[solves_data_part.index, 'subx' + str(ao_len)] = \
        solves_data_part[ao_len - 1:].rolling(ao_len, min_periods=0).apply(
            lambda ar: ((ar < subx_threshold).sum()) / ao_len, raw=True)['single']


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


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_pb_progression(solves_data, puzzle, category, ao_len, has_dates, timezone):
    if ao_len == 1:
        series = 'single'
        series_rsd = None
    elif ao_len == 3:
        series = 'mo3'
        series_rsd = 'rsd3'
    else:
        series = 'ao' + str(ao_len)
        series_rsd = 'rsd' + str(ao_len)

    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    # create a column for solve num
    solves_data_part.reset_index(inplace=True, drop=True)
    solves_data_part.index += 1
    solves_data_part.reset_index(inplace=True)

    if has_dates:
        column_list = ['index', 'SolveDatetime', series]
        if ao_len > 1:
            column_list.append(series_rsd)

        solves_data_pb = solves_data_part[(solves_data_part[series + '_cummin'].diff() != 0)
                                          & (solves_data_part[series + '_cummin'].notnull())][column_list]

        solves_data_pb['PB For Time'] = solves_data_pb['SolveDatetime'].diff(periods=-1) * -1

        last_date = solves_data_pb.tail(1)['SolveDatetime']
        last_date_value = Series(datetime.utcnow().replace(microsecond=0)).dt.tz_localize(
            'UTC').dt.tz_convert(timezone).dt.tz_localize(None).iloc[0]
        last_date_diff = last_date_value - last_date.iloc[0]
        if notnull(last_date_diff):
            solves_data_pb.loc[last_date.index, 'PB For Time'] = str(last_date_diff) + ' and counting'

        solves_data_pb.rename(inplace=True, columns={"SolveDatetime": "Date & Time"})
    else:
        column_list = ['index', series]
        if ao_len > 1:
            column_list.append(series_rsd)

        solves_data_pb = solves_data_part[(solves_data_part[series + '_cummin'].diff() != 0)
                                          & (solves_data_part[series + '_cummin'].notnull())][column_list]

    solves_data_pb['PB For # Solves'] = (solves_data_pb['index'].diff(periods=-1) * -1).dropna().apply(
        lambda x: str(int(x)))

    last_index = solves_data_pb.tail(1)['index']
    solves_data_pb.loc[last_index.index, 'PB For # Solves'] = str(
        solves_data_part.iloc[-1]['index'] - last_index.iloc[0]) + ' and counting'

    solves_data_pb['PB ' + series] = solves_data_pb[series].apply(sec2dtstr)
    solves_data_pb.rename(inplace=True, columns={"index": "Solve #"})

    return solves_data_pb


def get_all_pb_progressions(solves_data, puzzle, category, has_dates, timezone):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    res = OrderedDict()
    for ao_len in (1, 3, 5, 12, 50, 100, 1000):
        if ao_len == 1:
            series = 'single'
        elif ao_len == 3:
            series = 'mo3'
        else:
            series = 'ao' + str(ao_len)

        if not solves_data_part[series].isnull().all():
            res[ao_len] = get_pb_progression(solves_data, puzzle, category, ao_len, has_dates, timezone)
    return res


def generate_pbs_display(pb_progressions, has_dates):
    res = OrderedDict()

    for ao_len, pbs in pb_progressions.items():
        if ao_len == 1:
            series = 'single'
            series_rsd = None
        elif ao_len == 3:
            series = 'mo3'
            series_rsd = 'rsd3'
        else:
            series = 'ao' + str(ao_len)
            series_rsd = 'rsd' + str(ao_len)

        pbs = pbs.iloc[::-1]
        if has_dates:
            column_list = ['PB ' + series, 'Date & Time', 'PB For Time', 'Solve #', 'PB For # Solves']
            if ao_len > 1:
                column_list.append(series_rsd)

            pbs_display = pbs[column_list][:50]

            if pbs_display['Date & Time'].isnull().all():
                pbs_display.drop(labels='Date & Time', axis='columns', inplace=True)
            else:
                pbs_display['Date & Time'] = pbs_display['Date & Time'].fillna(value='--')

            if pbs_display['PB For Time'].isnull().all():
                pbs_display.drop(labels='PB For Time', axis='columns', inplace=True)
            else:
                pbs_display['PB For Time'] = pbs_display['PB For Time'].fillna(value='--')
        else:
            column_list = ['PB ' + series, 'Solve #', 'PB For # Solves']
            if ao_len > 1:
                column_list.append(series_rsd)

            pbs_display = pbs[column_list][:50]

        if ao_len > 1:
            pbs_display[series_rsd] = pbs_display[series_rsd].apply(lambda x: '{:.1%}'.format(x))

        res[series] = pbs_display

    return res


def get_top_solves(solves_data_part, ao_len, top_n, has_dates):
    if ao_len == 1:
        series = 'single'
        series_rsd = None
    elif ao_len == 3:
        series = 'mo3'
        series_rsd = 'rsd3'
    else:
        series = 'ao' + str(ao_len)
        series_rsd = 'rsd' + str(ao_len)

    if has_dates and not solves_data_part['Date & Time'].isnull().all():
        column_list = [series, 'Date & Time', 'Solve #']
    else:
        column_list = [series, 'Solve #']

    if ao_len > 1:
        column_list.append(series_rsd)

    top_solves = solves_data_part[(solves_data_part['Penalty'] != 2) & (solves_data_part[series].notnull())][
        column_list].sort_values([series, 'Solve #']).head(top_n)
    top_solves[series] = top_solves[series].apply(sec2dtstr)

    if ao_len > 1:
        top_solves[series_rsd] = top_solves[series_rsd].apply(lambda x: '{:.1%}'.format(x))

    if ('Date & Time' in column_list) and top_solves['Date & Time'].isnull().all():
        top_solves.drop(labels='Date & Time', axis='columns', inplace=True)
    top_solves.fillna(value='--', inplace=True)

    # create a column for solve rank
    top_solves.reset_index(inplace=True, drop=True)
    top_solves.index += 1
    top_solves.reset_index(inplace=True)
    top_solves.rename(inplace=True, columns={"index": "Rank"})

    if ao_len == 1:
        top_solves.rename(inplace=True, columns={'single': 'Single'})

    return top_solves


def get_all_top_solves(solves_data, puzzle, category, has_dates):
    top_n = 50

    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)].copy(
        deep=True)
    # create a column for solve num
    solves_data_part.reset_index(inplace=True, drop=True)
    solves_data_part.index += 1
    solves_data_part.reset_index(inplace=True)
    solves_data_part.rename(inplace=True, columns={"index": "Solve #"})
    if has_dates:
        solves_data_part.rename(inplace=True, columns={"SolveDatetime": "Date & Time"})

    res = OrderedDict()
    for ao_len in (1, 3, 5, 12, 50, 100, 1000):
        if ao_len == 1:
            series = 'single'
        elif ao_len == 3:
            series = 'mo3'
        else:
            series = 'ao' + str(ao_len)
        if not solves_data_part[series].isnull().all():
            res[series] = get_top_solves(solves_data_part, ao_len, top_n, has_dates)
    return res


def rename_puzzle(puz):
    puz = str(puz)
    if puz in ('222', '333', '444', '555', '666', '777'):
        return puz[0] + 'x' + puz[0]
    else:
        return puz


def get_all_solves_details(solves_data, has_dates, timezone, chart_by, secondary_y_axis, subx_thresholds):
    # generate a nested dict puzzle -> category -> pb progression df
    puzcats = {k: sorted(g['Category'].tolist(), key=lambda s: str(s).casefold())
               for k, g in solves_data[['Puzzle', 'Category']].drop_duplicates().groupby('Puzzle')}

    resdict = OrderedDict()
    for puz, cats in puzcats.items():
        catdict = OrderedDict()
        for cat in cats:
            pb_progressions = get_all_pb_progressions(solves_data, puz, cat, has_dates, timezone)
            pbs_display = generate_pbs_display(pb_progressions, has_dates)

            top_solves = get_all_top_solves(solves_data, puz, cat, has_dates)

            solves_plot = get_solves_plot(solves_data, puz, cat, has_dates, chart_by, pb_progressions, secondary_y_axis,
                                          subx_thresholds)
            histograms_plot = get_histograms_plot(solves_data, puz, cat)

            catdict[cat] = pbs_display, solves_plot, histograms_plot, top_solves

        renamed_puz = rename_puzzle(puz)
        resdict[renamed_puz] = catdict

    return resdict


def create_pbs_display_value(time, date=None, rsd=None):
    if notnull(date):
        date = date.date()
    else:
        date = None

    res = sec2dtstr(time)
    if date or rsd:
        res += '<span class="small">'
        if date:
            res += " <br/> " + str(date)
        if rsd:
            if date:
                res += " | "
            else:
                res += " <br/> "
            res += "RSD: " + '{:.1%}'.format(rsd)
        res += '</span>'
    return res


def get_overall_pbs(solves_data):
    min_idx = solves_data[solves_data['Penalty'] != 2][
        ['Puzzle', 'Category', 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000']]. \
        groupby(['Puzzle', 'Category']).idxmin()

    pbs = DataFrame()

    single_pbs = solves_data.loc[min_idx['single'].dropna()][
        ['Puzzle', 'Category', 'single', 'SolveDatetime']].set_index(
        ['Puzzle', 'Category'])
    pbs['single'] = single_pbs.apply(
        lambda row: create_pbs_display_value(row['single'], row['SolveDatetime']), axis=1)

    for ao_len in (3, 5, 12, 50, 100, 1000):
        if ao_len == 3:
            prefix = 'mo'
        else:
            prefix = 'ao'
        ao_len_str = str(ao_len)
        ao_pbs = solves_data.loc[min_idx[prefix + ao_len_str].dropna()][
            ['Puzzle', 'Category', prefix + ao_len_str, 'rsd' + ao_len_str, 'SolveDatetime']].set_index(
            ['Puzzle', 'Category'])
        if ao_pbs.size > 0:
            pbs[prefix + ao_len_str] = ao_pbs.apply(
                lambda row: create_pbs_display_value(row[prefix + ao_len_str], row['SolveDatetime'],
                                                     row['rsd' + ao_len_str]), axis=1)
        else:
            pbs[prefix + ao_len_str] = NaN

    counts = solves_data.groupby(['Puzzle', 'Category']).size().rename('#')

    pbs_with_count = concat([counts, pbs], axis=1)

    pbs_with_count = pbs_with_count.reindex(
        sorted(pbs_with_count.index, key=lambda s: (str(s[0]).casefold(), str(s[1]).casefold())))

    # escape puzzle and category names for html, as to_html escaping is now disabled for this table to use span
    pbs_with_count.index = pbs_with_count.index.map(lambda tp: tuple(escape(str(x)) for x in tp))

    if all(pbs_with_count.index.levels[0] == 'Sessions'):
        # no need to display if no puzzle data
        pbs_with_count.index = pbs_with_count.index.droplevel(level=0)
        pbs_with_count.rename_axis(None, inplace=True)
    else:
        # rename NNN to NxN for TwistyTimer
        renamed_puzzles = [rename_puzzle(puz) for puz in pbs_with_count.index.levels[0]]
        pbs_with_count.index = pbs_with_count.index.set_levels(renamed_puzzles, level=0)

    return pbs_with_count


def get_solves_plot(solves_data, puzzle, category, has_dates, chart_by, pb_progressions, secondary_y_axis,
                    subx_thresholds):
    plot_data = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]

    if chart_by == 'chart-by-solve-num':
        has_dates = False

    # for cases where a timer add dates support at some point, but some categories can still be completely date-less
    if has_dates and plot_data['SolveDatetime'].isnull().all():
        has_dates = False

    annotations = []
    if has_dates:
        data_plot_x = plot_data['SolveDatetime']
        count_null_dates = plot_data['SolveDatetime'].isnull().sum()
        if count_null_dates > 0:
            annotations.append(dict(x=0,
                                    xref='paper',
                                    y=1.05,
                                    yref='paper',
                                    showarrow=False,
                                    text=f'Not showing {count_null_dates} solves that lack dates',
                                    hovertext='Choose the "Chart by Solve #" option <br>'
                                              'when uploading the file to chart all solves'))
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
              'ao1000': 'blue',
              'rsd5': 'grey',
              'rsd12': 'rgba(128, 177, 211, 0.8999999999999999)',
              'rsd50': 'rgba(251, 128, 114, 1.0)',
              'rsd100': 'rgba(255, 153, 51, 1.0)',
              'rsd1000': 'rgba(128, 177, 211, 1.0)',
              'subx5': 'grey',
              'subx12': 'rgba(128, 177, 211, 0.8999999999999999)',
              'subx50': 'rgba(251, 128, 114, 1.0)',
              'subx100': 'rgba(255, 153, 51, 1.0)',
              'subx1000': 'rgba(128, 177, 211, 1.0)'
              }

    over_60 = False  # for deciding on tickformat
    for ao_len in (1, 3, 5, 12, 50, 100, 1000):
        series_subx = None
        series_rsd = None
        if ao_len == 1:
            series = 'single'
        elif ao_len == 3:
            series = 'mo3'
            if secondary_y_axis == 'subx':
                series_subx = 'subx3'
            elif secondary_y_axis == 'rsd':
                series_rsd = 'rsd3'
        else:
            series = 'ao' + str(ao_len)
            if secondary_y_axis == 'subx':
                series_subx = 'subx' + str(ao_len)
            elif secondary_y_axis == 'rsd':
                series_rsd = 'rsd' + str(ao_len)

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
                legendgroup=series,
                visible='legendonly',
                line={'color': colors.get(series, 'black'),
                      'width': 1.3}
            ))

            if secondary_y_axis and (ao_len > 1):
                if secondary_y_axis == 'subx':
                    y2_data = plot_data[series_subx]
                    y2_name = "% sub-" + sec2dtstr(subx_thresholds[(puzzle, category)]) + " " + series
                    y2_colors = series_subx
                elif secondary_y_axis == 'rsd':
                    y2_data = plot_data[series_rsd]
                    y2_name = series_rsd
                    y2_colors = series_rsd
                else:
                    y2_data = []
                    y2_name = ''
                    y2_colors = ''

                data.append(go.Scatter(
                    x=data_plot_x,
                    y=y2_data,
                    yaxis='y2',
                    name=y2_name,
                    mode='lines',
                    legendgroup=series,
                    showlegend=False,
                    visible='legendonly',
                    line={'color': colors.get(y2_colors, 'black'),
                          'width': 1.3}
                ))

            pbs = pb_progressions[ao_len]

            # add an extra point for continuing the PB line until the last solve
            plot_last_index = plot_data.index.values[-1]
            pbs.loc[plot_last_index, [series, 'PB ' + series, 'Solve #']] = [pbs.iloc[-1][series], '', len(plot_data)]

            if has_dates:
                pbs.loc[plot_last_index, 'Date & Time'] = plot_data.iloc[-1]['SolveDatetime']
                pb_plot_x = pbs['Date & Time']
            else:
                pb_plot_x = pbs['Solve #']

            marker_sizes = [4 for _ in range(len(pbs))]
            marker_sizes[-1] = 0

            data.append(go.Scatter(
                x=pb_plot_x,
                y=pbs[series].apply(lambda x: to_datetime(x, unit='s')),
                name='PB ' + series,
                text=pbs['PB ' + series],
                hoverinfo="x+name+text",
                mode="lines+markers",
                marker={'size': marker_sizes,
                        'color': colors.get(series, 'black')},
                visible='legendonly',
                # fill='tozeroy',
                line={'color': colors.get(series, 'black'),
                      'width': 1.3,
                      'dash': 'dash',
                      'shape': 'hv'}
            ))

            if plot_data[series].max() >= 60:
                over_60 = True

    if not over_60:
        yaxis_tickformat = '%S.%L'
    else:
        yaxis_tickformat = '%M:%S'

    layout_xaxis = dict()
    yaxis2_title = None
    yaxis2_tickformat = None
    yaxis2_range = None
    yaxis2_fixedrange = False
    if secondary_y_axis:
        layout_xaxis['domain'] = [0, 0.97]
        if secondary_y_axis == 'subx':
            yaxis2_title = "% sub-" + sec2dtstr(subx_thresholds[(puzzle, category)])
            yaxis2_tickformat = '.1%'
            yaxis2_range = [0, 1]
            yaxis2_fixedrange = True
        elif secondary_y_axis == 'rsd':
            yaxis2_title = 'RSD'
            yaxis2_tickformat = '.1%'

    layout = go.Layout(margin={'l': 50,
                               'r': 50,
                               'b': 50,
                               't': 50,
                               'pad': 0
                               },
                       yaxis={'tickformat': yaxis_tickformat},
                       yaxis2={'side': 'right',
                               'overlaying': 'y',
                               'position': 0.97,
                               'title': yaxis2_title,
                               'tickformat': yaxis2_tickformat,
                               'range': yaxis2_range,
                               'fixedrange': yaxis2_fixedrange},
                       legend={'tracegroupgap': 0},
                       xaxis=layout_xaxis,
                       annotations=annotations)

    fig = dict(data=data, layout=layout)

    # display only the singles and the two largest aoX
    fig['data'][0].visible = True
    if len(fig['data']) >= 4:
        fig['data'][-2].visible = True
        if secondary_y_axis:
            fig['data'][-3].visible = True
    if len(fig['data']) >= 6:
        if secondary_y_axis:
            fig['data'][-5].visible = True
            fig['data'][-6].visible = True
        else:
            fig['data'][-4].visible = True

    config = {'scrollZoom': True}

    return plot(fig, include_plotlyjs=False, output_type='div', config=config)


def generate_histogram(plot_data_raw, name):
    max_time = int(plot_data_raw.max()) + 1
    min_time = int(plot_data_raw.min())
    intervals = list(range(min_time, max_time + 1))
    intervals_dt = [sec2dt(sec) for sec in intervals]
    labels = [sec2dtstr(sec) + ".00-" + sec2dtstr(sec + 0.99) for sec in intervals[:-1]]

    bins = cut(plot_data_raw, intervals, right=False, labels=labels)
    plot_data = plot_data_raw.groupby(bins).count()

    return go.Bar(
        x=intervals_dt,
        y=plot_data,
        text=plot_data.index,
        hoverinfo="x+y+text",
        visible=False,
        name=name
    )


def get_histograms_plot(solves_data, puzzle, category):
    data = list()
    annotations = dict()
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    solves_count = len(solves_data_part)

    plot_data_raw = solves_data_part['single']
    data.append(generate_histogram(plot_data_raw, 'all'))
    annotations['all'] = 'all ' + str(solves_count) + ' solves'

    if solves_count > 100:
        plot_data_raw = solves_data_part[['single', 'ao100', 'rsd100']][-100:]
        data.append(generate_histogram(plot_data_raw['single'], 'last 100'))
        annotations['last 100'] = 'last ao100: ' + str(plot_data_raw['ao100'].iloc[-1]) + \
                                  '<br>rsd100: ' + '{:.1%}'.format(plot_data_raw['rsd100'].iloc[-1])

    part_reindexed = solves_data_part[['single', 'ao100', 'ao1000', 'rsd100', 'rsd1000']].reset_index()
    idxmin = part_reindexed['ao100'].idxmin()
    if notnull(idxmin):
        data.append(generate_histogram(part_reindexed['single'][idxmin + 1 - 100: idxmin + 1], 'PB ao100'))
        annotations['PB ao100'] = 'PB ao100: ' + str(part_reindexed['ao100'][idxmin]) + \
                                  '<br>rsd100: ' + '{:.1%}'.format(part_reindexed['rsd100'][idxmin])

    if solves_count > 1000:
        plot_data_raw = solves_data_part[['single', 'ao1000', 'rsd1000']][-1000:]
        data.append(generate_histogram(plot_data_raw['single'], 'last 1000'))
        annotations['last 1000'] = 'last ao1000: ' + str(plot_data_raw['ao1000'].iloc[-1]) + \
                                   '<br>rsd1000: ' + '{:.1%}'.format(plot_data_raw['rsd1000'].iloc[-1])

    idxmin = part_reindexed['ao1000'].idxmin()
    if notnull(idxmin):
        data.append(generate_histogram(part_reindexed['single'][idxmin + 1 - 1000: idxmin + 1], 'PB ao1000'))
        annotations['PB ao1000'] = 'PB ao1000: ' + str(part_reindexed['ao1000'][idxmin]) + \
                                   '<br>rsd1000: ' + '{:.1%}'.format(part_reindexed['rsd1000'][idxmin])

    data[0].visible = True

    buttons = list()
    datalen = len(data)
    for i, bar in enumerate(data):
        visibility = [True if trace == i else False for trace in range(datalen)]
        buttons.append(dict(args=[{'visible': visibility},
                                  {'annotations': [dict(x=0,
                                                        xref='paper',
                                                        y=1.05,
                                                        yref='paper',
                                                        showarrow=False,
                                                        align='left',
                                                        text=annotations.get(bar['name'], ''))]}],
                            label=bar['name'],
                            method='update'))

    layout = go.Layout(margin={'l': 50,
                               'r': 50,
                               'b': 50,
                               't': 50,
                               'pad': 4
                               },
                       xaxis={'tickformat': '%M:%S'},
                       annotations=[dict(x=0,
                                         xref='paper',
                                         y=1.05,
                                         yref='paper',
                                         showarrow=False,
                                         text=annotations.get('all'))]
                       )

    updatemenus = list([
        dict(
            buttons=buttons,
            direction='down',
            type='dropdown',
            x=1,
            xanchor='right',
            # y=1.1,
            yanchor='top'
        ),
    ])
    layout['updatemenus'] = updatemenus

    fig = dict(data=data, layout=layout)

    config = {'scrollZoom': True}

    return plot(fig, include_plotlyjs=False, output_type='div', config=config)


def generate_dates_histogram(solves_data, group_date_str, tickformat, dtick, day_end_hour):
    solves_grouped = solves_data[['Puzzle', 'Category', 'single']].groupby(
        [solves_data.SolveDatetime.dropna().apply(lambda x: x - timedelta(hours=day_end_hour)).dt.strftime(
            group_date_str), solves_data.Puzzle, solves_data.Category])[
        'Puzzle'].count().rename('#')

    annotations = []
    count_null_dates = solves_data['SolveDatetime'].isnull().sum()
    if count_null_dates > 0:
        annotations.append(dict(x=0,
                                xref='paper',
                                y=1.05,
                                yref='paper',
                                showarrow=False,
                                text=f'Not showing {count_null_dates} solves that lack dates'))

    renamed_puzzles = [rename_puzzle(puz) for puz in solves_grouped.index.levels[1]]
    solves_grouped.index = solves_grouped.index.set_levels(renamed_puzzles, level=1)
    plot_data = solves_grouped.unstack([1, 2])

    # noinspection PyUnresolvedReferences
    if plot_data.columns.levels[0][0] == 'Sessions':
        # noinspection PyUnresolvedReferences
        plot_data.columns = [str(col[1]).strip() for col in plot_data.columns.values]
    else:
        # noinspection PyUnresolvedReferences
        plot_data.columns = [' '.join(str(i).strip() for i in col) for col in plot_data.columns.values]

    data = []
    for col in sorted(plot_data.columns, key=lambda s: str(s).casefold()):
        data.append(go.Bar(
            x=plot_data.index,
            y=plot_data[col],
            name=col,
            showlegend=True
        ))

    layout = go.Layout(
        barmode='stack',
        margin={'l': 50,
                'r': 50,
                'b': 75,
                't': 50,
                'pad': 4
                },
        xaxis={'tickformat': tickformat, 'dtick': dtick},
        legend={'traceorder': 'normal'},
        annotations=annotations
    )
    fig = dict(data=data, layout=layout)

    config = {'scrollZoom': True}

    return plot(fig, include_plotlyjs=False, output_type='div', config=config)


def get_solves_by_dates(solves_data, day_end_hour):
    resdict = OrderedDict()

    groups = (('Day', '%Y-%m-%d', None, None),
              ('Month', '%Y-%m', '%b %Y', 'M1'),
              ('Year', '%Y', 'd', None))

    for group_name, group_date_str, tickformat, dtick in groups:
        resdict[group_name] = generate_dates_histogram(solves_data, group_date_str, tickformat, dtick, day_end_hour)

    return resdict


def get_time_ms_from_string(s):
    solve_time_parts = s.split(':')
    if len(solve_time_parts) == 3:
        solve_time_ms = (float(solve_time_parts[0]) * 60 * 60 + float(solve_time_parts[1]) * 60 + float(
            solve_time_parts[2])) * 1000
    elif len(solve_time_parts) == 2:
        solve_time_ms = (float(solve_time_parts[0]) * 60 + float(solve_time_parts[1])) * 1000
    else:
        solve_time_ms = float(solve_time_parts[0]) * 1000
    return solve_time_ms


def parse_cstimer_csv_result(s):
    s = str(s)
    if s.startswith('DNF'):
        solve_time = s[4:-1]  # DNF(0.99)
        penalty = 2
    elif s[-1] == '+':
        solve_time = s[:-1]  # cstimer already adds the +2
        penalty = 1
    else:
        solve_time = s
        penalty = 0

    solve_time_ms = get_time_ms_from_string(solve_time)

    return solve_time_ms, penalty


def parse_zyxtimer_result(s):
    if s[-3:] == 'DNF':
        solve_time = s[:-3]
        penalty = 2
    elif s[-1] == '+':
        solve_time = s[:-1]
        penalty = 1
    else:
        solve_time = s
        penalty = 0

    solve_time_ms = get_time_ms_from_string(solve_time)

    if penalty == 1:
        solve_time_ms += 2000

    return solve_time_ms, penalty


def parse_chaotimer_result(s):
    s = s.strip("()")

    if s == 'DNF':
        solve_time = '0'
        penalty = 2
    elif s[-1] == '+':
        solve_time = s[:-1]  # chaotimer already adds the +2
        penalty = 1
    else:
        solve_time = s
        penalty = 0

    solve_time_ms = get_time_ms_from_string(solve_time)

    return solve_time_ms, penalty


def parse_qqtimer_result(s):
    if s[-1] == '-':
        solve_time_ms = int(s[:-1])
        penalty = 2
    elif s[-1] == '+':
        solve_time_ms = int(s[:-1]) + 2000
        penalty = 1
    else:
        solve_time_ms = int(s)
        penalty = 0

    return solve_time_ms, penalty


class WCAIDValueError(ValueError):
    pass


# penalty: 0: none, 1: +2 (time should already include the +2), 2: DNF
def create_dataframe(file, timezone):
    file.seek(0)
    headers = file.readline().decode()
    file.seek(0)

    if headers.startswith('Puzzle,Category,Time(millis),Date(millis),Scramble,Penalty,Comment'):
        # TwistyTimer
        timer_type = 'TwistyTimer'
        df = read_csv(file, sep=';', skiprows=1, header=None)
        df.columns = headers.strip().split(sep=',')
        has_dates = True

        df = df.rename_axis('MyIdx').sort_values(['Date(millis)', 'MyIdx'])

        df['SolveDatetime'] = to_datetime(df['Date(millis)'], unit='ms').astype('datetime64[s]').dt.tz_localize(
            'UTC').dt.tz_convert(timezone).dt.tz_localize(None)

        return df, has_dates, timer_type
    elif headers.startswith('{"session1"'):
        # cstimer
        timer_type = 'cstimer'
        data = json.load(file)

        solves = []
        has_dates = False  # correct later to True if dates found
        for session_id, session_values in json.loads(json.loads(data['properties'])['sessionData']).items():
            for solve_data in json.loads(data.get('session' + session_id, '[]')):
                if len(solve_data) == 3:
                    # old version, without dates
                    times, scramble, notes = solve_data
                    date = None
                else:
                    # new version, with dates (starting December 2018)
                    times, scramble, notes, date = solve_data
                    has_dates = True

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

                # somehow the time can be null in cstimer's full export
                if not times[1]:
                    final_time = 0
                    penalty = 2

                solves.append([session_values['name'], date, final_time, penalty])

        df = DataFrame(data=solves, columns=['Category', 'Date(millis)', 'Time(millis)', 'Penalty'])

        df['SolveDatetime'] = to_datetime(df['Date(millis)'], unit='s').astype('datetime64[s]').dt.tz_localize(
            'UTC').dt.tz_convert(timezone).dt.tz_localize(None)

        df['Puzzle'] = 'Sessions'

        return df, has_dates, timer_type
    elif headers.startswith('No.;Time;Comment;Scramble;Date;'):
        # cstimer CSV export of a single session / part of a session
        timer_type = 'cstimer_csv'

        df = read_csv(file, delimiter=';')

        parsed = df['Time'].apply(parse_cstimer_csv_result)
        df = concat([df, parsed.apply(Series, index=['Time(millis)', 'Penalty'])], axis=1)

        df['SolveDatetime'] = to_datetime(df['Date']).astype('datetime64[s]')  # already in local timezone

        if not df['SolveDatetime'].isnull().all():
            has_dates = True
        else:
            has_dates = False

        df['Puzzle'] = 'Sessions'
        df['Category'] = 'cstimer'

        return df, has_dates, timer_type
    elif headers.startswith('{"puzzles":[{"name":'):
        # BlockKeeper
        timer_type = 'BlockKeeper'
        data = json.load(file)

        solves = []
        for puzzle in data['puzzles']:
            for category in puzzle['sessions']:
                for solve in category['records']:
                    if solve['result'] == 'OK':
                        penalty = 0
                    elif solve['result'] == 'DNF':
                        penalty = 2
                    else:
                        penalty = 1  # +2 penalty (marked in the file as "result": "+2"

                    time_millis = solve['time'] * 1000
                    if penalty == 1:
                        time_millis += 2000

                    # some timers like Block Keeper only added dates at some point in the middle of their existence
                    solves.append([puzzle['name'], category['name'], time_millis, solve.get('date', None), penalty])

        df = DataFrame(data=solves, columns=['Puzzle', 'Category', 'Time(millis)', 'Date(millis)', 'Penalty'])
        df['SolveDatetime'] = to_datetime(df['Date(millis)'], unit='ms').astype('datetime64[s]').dt.tz_localize(
            'UTC').dt.tz_convert(timezone).dt.tz_localize(None)
        has_dates = True

        return df, has_dates, timer_type
    elif headers.startswith('Session: '):
        # ZYXTimer
        timer_type = 'ZYXTimer'

        all_solves = []
        session = None  # avoiding "before assignment" warning
        for line in file:
            line_str = line.decode()
            if line_str.startswith('Session:'):
                session = line_str[9:].strip()
            elif len(line_str.strip()) > 0:
                # remove comments in [], although as the note could include commas and [], it's never fully safe
                line_clean = re.sub(r'\[.*\]', '', line_str.strip())

                solves = line_clean.split(sep=', ')
                session_solves = [[session] + list(parse_zyxtimer_result(solve)) for solve in solves]
                all_solves.extend(session_solves)
        df = DataFrame(data=all_solves, columns=['Category', 'Time(millis)', 'Penalty'])

        has_dates = False
        df['Puzzle'] = 'Sessions'

        return df, has_dates, timer_type
    elif headers.strip('"')[-1] == '>':
        # qqtimer
        timer_type = 'qqtimer'

        # comments are in the form "time|comment", + at end (after comment...), last char >
        line_clean = re.sub(r'\|.*?([>,+])', r'\1', headers.strip('"'))[:-1]
        solves = [parse_qqtimer_result(solve) for solve in line_clean.split(sep=',')]

        df = DataFrame(data=solves, columns=['Time(millis)', 'Penalty'])

        has_dates = False
        df['Puzzle'] = 'Sessions'
        df['Category'] = 'qqtimer'

        return df, has_dates, timer_type
    elif headers.count('\t') == 4:
        # Prisma Puzzle Timer
        timer_type = 'Prisma'

        df = read_csv(file, delimiter='\t', names=['SolveNum', 'DateStr', 'ResultStr', 'PenaltyStr', 'Scramble'],
                      dtype={'PenaltyStr': str}, na_filter=False)
        # prisma displays latest first, reverse it
        df.iloc[:] = df.iloc[::-1].values
        df['Time(millis)'] = df['ResultStr'].apply(get_time_ms_from_string)
        df['Penalty'] = select(
            [
                df['PenaltyStr'] == '+2',
                df['PenaltyStr'] == 'DNF'
            ],
            [
                1,
                2
            ],
            default=0
        )
        df.loc[df['Penalty'] == 1, 'Time(millis)'] += 2000

        # prisma is using a localized medium date and time string
        # try to parse it naively with pandas and fallback to dateparser (much slower)
        try:
            df['SolveDatetime'] = to_datetime(df['DateStr']).astype('datetime64[s]')
        except ValueError:
            ddp = dateparser.date.DateDataParser(try_previous_locales=True)
            df['SolveDatetime'] = df['DateStr'].apply(lambda x: ddp.get_date_data(x)['date_obj']).astype(
                'datetime64[s]')

        has_dates = True
        df['Puzzle'] = 'Sessions'
        df['Category'] = 'Prisma'

        return df, has_dates, timer_type
    elif headers.startswith('Generated By ChaoTimer'):
        # ChaoTimer
        timer_type = 'ChaoTimer'

        all_solves = []
        session = None  # avoiding "before assignment" warning
        for i, line in enumerate(file):
            if i == 2:
                session = line.decode().strip()
            elif i == 10:
                solves = line.decode().strip().split(sep=', ')
                session_solves = [[session] + list(parse_chaotimer_result(solve)) for solve in solves]
                all_solves.extend(session_solves)
        df = DataFrame(data=all_solves, columns=['Category', 'Time(millis)', 'Penalty'])

        has_dates = False
        df['Puzzle'] = 'Sessions'

        return df, has_dates, timer_type
    elif len(headers) == 10:
        # WCA ID
        timer_type = 'WCAID'

        with zipfile.ZipFile(os.path.join(WCA_DATA_FOLDER, 'WCA_export.tsv.zip')) as z:
            filtered = BytesIO()
            with z.open('WCA_export_Results.tsv') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        filtered.write(line)
                    if '\t' + headers.upper() + '\t' in line.decode():
                        filtered.write(line)
            filtered.seek(0)
            results = read_csv(filtered, sep='\t', dtype={'eventId': object})
            filtered.close()

            with z.open('WCA_export_Competitions.tsv') as f:
                comps = read_csv(f, sep='\t')

            with z.open('WCA_export_Events.tsv') as f:
                events = read_csv(f, sep='\t')

        if len(results) == 0:
            raise WCAIDValueError

        results.reset_index(inplace=True)
        results.rename(inplace=True, columns={'index': 'resultRowId'})
        comps['SolveDatetime'] = to_datetime(comps[['year', 'month', 'day']]).astype('datetime64[s]')
        events_timed = events[events['format'] == 'time']
        rescomp = merge(results, comps, how='inner', left_on=['competitionId'], right_on=['id'])
        all_joined = merge(rescomp[['eventId', 'personName', 'personId',
                                    'SolveDatetime', 'resultRowId', 'value1', 'value2', 'value3', 'value4', 'value5']],
                           events_timed, how='inner', left_on=['eventId'], right_on=['id'])
        melted = melt(all_joined, id_vars=['name', 'personName', 'personId', 'SolveDatetime', 'resultRowId'],
                      value_vars=['value1', 'value2', 'value3', 'value4', 'value5'], var_name='result_id',
                      value_name='result').sort_values(
            ['SolveDatetime', 'name', 'resultRowId', 'result_id'])
        melted = melted[(melted['result'] != 0) & (melted['result'] != -2)]  # 0=no result; -2=DNS
        melted['Penalty'] = 0
        melted.loc[melted['result'] <= 0, 'Penalty'] = 2  # -1=DNF, others?
        melted['Time(millis)'] = melted['result'] * 10
        melted.rename(inplace=True, columns={'name': 'Category'})
        melted['Puzzle'] = melted['personName'] + ' (' + melted['personId'] + ')'

        has_dates = True
        return melted[['Puzzle', 'Category', 'SolveDatetime', 'Time(millis)', 'Penalty']], has_dates, timer_type
    else:
        raise NotImplementedError('Unrecognized file type')


def drop_all_dnf_categories(solves_data):
    solves_grouped = solves_data.groupby(['Puzzle', 'Category'])['single']
    non_dnf = solves_grouped.any()
    all_dnf = non_dnf[~non_dnf].reset_index()[['Puzzle', 'Category']]
    solves_grouped_index = solves_data.set_index(['Puzzle', 'Category']).index
    all_dnf_index = all_dnf.set_index(['Puzzle', 'Category']).index
    return solves_data[~solves_grouped_index.isin(all_dnf_index)]


def process_data(file, chart_by, secondary_y_axis, subx_threshold_mode, subx_override, day_end_hour, timezone, trim_percentage):
    set_option('display.max_colwidth', -1)

    # timezone could be a tz name string, or an offset in minutes
    if represents_int(timezone):
        timezone = int(timezone) * 60  # need it in seconds for tz_convert

    # limit to 40 to keep at least one solve for ao5
    if trim_percentage < 0 or trim_percentage > 40:
        trim_percentage = 5

    solves_data, has_dates, timer_type = create_dataframe(file, timezone)

    if not has_dates:
        solves_data['SolveDatetime'] = NaN

    solves_data['TimeCentiSec'] = (solves_data['Time(millis)'] / 10).astype(int)

    solves_data['single'] = [NaN if row.Penalty == 2 else (row.TimeCentiSec / 100)
                             for row in solves_data[['Penalty', 'TimeCentiSec']].itertuples()]

    solves_data['IsDNF'] = [1 if row == 2 else 0 for row in solves_data['Penalty']]

    solves_data = drop_all_dnf_categories(solves_data)

    subx_thresholds = dict()
    for idx, puzzle, category in solves_data[['Puzzle', 'Category']].drop_duplicates().itertuples():
        for ao_len in (3, 5, 12, 50, 100, 1000):
            calculate_and_store_running_ao(solves_data, puzzle, category, ao_len, trim_percentage)
        if secondary_y_axis == 'subx':
            if subx_threshold_mode == 'auto':
                subx_threshold = get_subx_threshold(solves_data, puzzle, category)
            else:
                try:
                    subx_threshold = float(subx_override)
                except ValueError:
                    subx_threshold = get_subx_threshold(solves_data, puzzle, category)
            subx_thresholds[(puzzle, category)] = subx_threshold
            for ao_len in (3, 5, 12, 50, 100, 1000):
                calculate_and_store_running_subx(solves_data, puzzle, category, ao_len, subx_threshold)

    for series in 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000':
        solves_data[series + '_str'] = solves_data[series][notnull(solves_data[series])].apply(sec2dtstr)
        solves_data[series + '_dt'] = solves_data[series].apply(sec2dt)

    solves_data[
        ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
         'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']] = \
        solves_data.groupby(['Puzzle', 'Category']).cummin()[
            ['single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000']]

    solves_data[
        ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
         'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']] = \
        solves_data.groupby(['Puzzle', 'Category'])[
            ['single_cummin', 'mo3_cummin', 'ao5_cummin', 'ao12_cummin',
             'ao50_cummin', 'ao100_cummin', 'ao1000_cummin']].fillna(method='ffill')

    solves_details = get_all_solves_details(solves_data, has_dates, timezone, chart_by, secondary_y_axis,
                                            subx_thresholds)
    overall_pbs = get_overall_pbs(solves_data)
    if has_dates:
        solves_by_dates = get_solves_by_dates(solves_data, day_end_hour)
    else:
        solves_by_dates = None

    return solves_details, overall_pbs, solves_by_dates, timer_type, len(solves_data)
