from datetime import datetime, timedelta
from math import floor, ceil, isnan

from numba import jit
from numpy import sort, repeat, NaN, array, all, select
from pandas import read_csv, to_datetime, concat, notnull, DataFrame, Series, cut

from plotly.offline import plot
import plotly.graph_objs as go

import json
import re
import dateparser

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


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_pb_progression(solves_data, puzzle, category, col_name, has_dates, timezone):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    # create a column for solve num
    solves_data_part.reset_index(inplace=True, drop=True)
    solves_data_part.index += 1
    solves_data_part.reset_index(inplace=True)

    if has_dates:
        solves_data_pb = solves_data_part[(solves_data_part[col_name + '_cummin'].diff() != 0)
                                          & (solves_data_part[col_name + '_cummin'].notnull())][
            ['index', 'SolveDatetime', col_name]]

        solves_data_pb['PB For Time'] = solves_data_pb['SolveDatetime'].diff(periods=-1) * -1

        last_date = solves_data_pb.tail(1)['SolveDatetime']
        last_date_value = Series(datetime.utcnow().replace(microsecond=0)).dt.tz_localize(
            'UTC').dt.tz_convert(timezone).dt.tz_localize(None).iloc[0]
        last_date_diff = last_date_value - last_date.iloc[0]
        if notnull(last_date_diff):
            solves_data_pb.loc[last_date.index, 'PB For Time'] = str(last_date_diff) + ' and counting'

        solves_data_pb.rename(inplace=True, columns={"SolveDatetime": "Date & Time"})
    else:
        solves_data_pb = solves_data_part[(solves_data_part[col_name + '_cummin'].diff() != 0)
                                          & (solves_data_part[col_name + '_cummin'].notnull())][
            ['index', col_name]]

    solves_data_pb['PB For # Solves'] = (solves_data_pb['index'].diff(periods=-1) * -1).dropna().apply(
        lambda x: str(int(x)))

    last_index = solves_data_pb.tail(1)['index']
    solves_data_pb.loc[last_index.index, 'PB For # Solves'] = str(
        solves_data_part.iloc[-1]['index'] - last_index.iloc[0]) + ' and counting'

    solves_data_pb['PB ' + col_name] = solves_data_pb[col_name].apply(sec2dtstr)
    solves_data_pb.rename(inplace=True, columns={"index": "Solve #"})

    return solves_data_pb


def get_all_pb_progressions(solves_data, puzzle, category, has_dates, timezone):
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    res = OrderedDict()
    for series in 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000':
        if not solves_data_part[series].isnull().all():
            res[series] = get_pb_progression(solves_data, puzzle, category, series, has_dates, timezone)
    return res


def generate_pbs_display(pb_progressions, has_dates):
    res = OrderedDict()

    for series, pbs in pb_progressions.items():
        pbs = pbs.iloc[::-1]
        if has_dates:
            pbs_display = pbs[['PB ' + series, 'Date & Time', 'PB For Time', 'Solve #',
                               'PB For # Solves']][:50]

            if pbs_display['Date & Time'].isnull().all():
                pbs_display.drop(labels='Date & Time', axis='columns', inplace=True)
            else:
                pbs_display['Date & Time'] = pbs_display['Date & Time'].fillna(value='--')

            if pbs_display['PB For Time'].isnull().all():
                pbs_display.drop(labels='PB For Time', axis='columns', inplace=True)
            else:
                pbs_display['PB For Time'] = pbs_display['PB For Time'].fillna(value='--')
        else:
            pbs_display = pbs[['PB ' + series, 'Solve #', 'PB For # Solves']][:50]

        res[series] = pbs_display

    return res


def get_top_solves(solves_data_part, col_name, top_n, has_dates):
    if has_dates and not solves_data_part['Date & Time'].isnull().all():
        column_list = [col_name, 'Date & Time', 'Solve #']
    else:
        column_list = [col_name, 'Solve #']

    top_solves = solves_data_part[(solves_data_part['Penalty'] != 2) & (solves_data_part[col_name].notnull())][
        column_list].sort_values([col_name, 'Solve #']).head(top_n)
    top_solves[col_name] = top_solves[col_name].apply(sec2dtstr)

    if ('Date & Time' in column_list) and top_solves['Date & Time'].isnull().all():
        top_solves.drop(labels='Date & Time', axis='columns', inplace=True)
    top_solves.fillna(value='--', inplace=True)

    return top_solves


def get_all_top_solves(solves_data, puzzle, category, has_dates):
    top_n = 50

    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)].copy(
        deep=True)
    # create a column for solve num
    solves_data_part.reset_index(inplace=True, drop=True)
    solves_data_part.index += 1
    solves_data_part.reset_index(inplace=True, )
    solves_data_part.rename(inplace=True, columns={"index": "Solve #"})
    if has_dates:
        solves_data_part.rename(inplace=True, columns={"SolveDatetime": "Date & Time"})

    res = OrderedDict()
    for series in 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000':
        if not solves_data_part[series].isnull().all():
            res[series] = get_top_solves(solves_data_part, series, top_n, has_dates)
    return res


def rename_puzzle(puz):
    puz = str(puz)
    if puz in ('222', '333', '444', '555', '666', '777'):
        return puz[0] + 'x' + puz[0]
    else:
        return puz


def get_all_solves_details(solves_data, has_dates, timezone, chart_by):
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

            solves_plot = get_solves_plot(solves_data, puz, cat, has_dates, chart_by, pb_progressions)
            histograms_plot = get_histograms_plot(solves_data, puz, cat)

            catdict[cat] = pbs_display, solves_plot, histograms_plot, top_solves

        renamed_puz = rename_puzzle(puz)
        resdict[renamed_puz] = catdict

    return resdict


def get_overall_pbs(solves_data):
    pbs = solves_data[solves_data['Penalty'] != 2][
        ['Puzzle', 'Category', 'single', 'mo3', 'ao5', 'ao12', 'ao50', 'ao100', 'ao1000']]. \
        groupby(['Puzzle', 'Category']).min().applymap(sec2dtstr)

    counts = solves_data.groupby(['Puzzle', 'Category']).size().rename('#')

    pbs_with_count = concat([counts, pbs], axis=1)

    pbs_with_count = pbs_with_count.reindex(
        sorted(pbs_with_count.index, key=lambda s: (str(s[0]).casefold(), str(s[1]).casefold())))

    if all(pbs_with_count.index.levels[0] == 'Sessions'):
        # no need to display if no puzzle data
        pbs_with_count.index = pbs_with_count.index.droplevel(level=0)
        pbs_with_count.rename_axis(None, inplace=True)
    else:
        # rename NNN to NxN for TwistyTimer
        renamed_puzzles = [rename_puzzle(puz) for puz in pbs_with_count.index.levels[0]]
        pbs_with_count.index = pbs_with_count.index.set_levels(renamed_puzzles, level=0)

    return pbs_with_count


def get_solves_plot(solves_data, puzzle, category, has_dates, chart_by, pb_progressions):
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
              'ao1000': 'blue'}

    over_60 = False  # for deciding on tickformat
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

            pbs = pb_progressions[series]

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

    layout = go.Layout(margin={'l': 50,
                               'r': 50,
                               'b': 50,
                               't': 50,
                               'pad': 4
                               },
                       yaxis={'tickformat': yaxis_tickformat},
                       annotations=annotations)

    fig = dict(data=data, layout=layout)

    # display only the singles and the two largest aoX
    fig['data'][0].visible = True
    if len(fig['data']) >= 3:
        fig['data'][-2].visible = True
    if len(fig['data']) >= 5:
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
    ao = dict()
    solves_data_part = solves_data[(solves_data['Puzzle'] == puzzle) & (solves_data['Category'] == category)]
    solves_count = len(solves_data_part)

    plot_data_raw = solves_data_part['single']
    data.append(generate_histogram(plot_data_raw, 'all'))
    ao['all'] = 'all ' + str(solves_count) + ' solves'

    if solves_count > 100:
        plot_data_raw = solves_data_part[['single', 'ao100']][-100:]
        data.append(generate_histogram(plot_data_raw['single'], 'last 100'))
        ao['last 100'] = 'last ao100: ' + str(plot_data_raw['ao100'].iloc[-1])

    part_reindexed = solves_data_part[['single', 'ao100', 'ao1000']].reset_index()
    idxmin = part_reindexed['ao100'].idxmin()
    if notnull(idxmin):
        data.append(generate_histogram(part_reindexed['single'][idxmin + 1 - 100: idxmin + 1], 'PB ao100'))
        ao['PB ao100'] = 'PB ao100: ' + str(part_reindexed['ao100'][idxmin])

    if solves_count > 1000:
        plot_data_raw = solves_data_part[['single', 'ao1000']][-1000:]
        data.append(generate_histogram(plot_data_raw['single'], 'last 1000'))
        ao['last 1000'] = 'last ao1000: ' + str(plot_data_raw['ao1000'].iloc[-1])

    idxmin = part_reindexed['ao1000'].idxmin()
    if notnull(idxmin):
        data.append(generate_histogram(part_reindexed['single'][idxmin + 1 - 1000: idxmin + 1], 'PB ao1000'))
        ao['PB ao1000'] = 'PB ao1000: ' + str(part_reindexed['ao1000'][idxmin])

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
                                                        text=ao.get(bar['name'], ''))]}],
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
                                         text=ao.get('all'))]
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


def generate_dates_histogram(solves_data, group_date_str, tickformat, dtick):
    solves_grouped = solves_data[['Puzzle', 'Category', 'single']].groupby(
        [solves_data.SolveDatetime.dropna().dt.strftime(group_date_str), solves_data.Puzzle, solves_data.Category])[
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


def get_solves_by_dates(solves_data):
    resdict = OrderedDict()

    groups = (('Day', '%Y-%m-%d', None, None),
              ('Month', '%Y-%m', '%b %Y', 'M1'),
              ('Year', '%Y', 'd', None))

    for group_name, group_date_str, tickformat, dtick in groups:
        resdict[group_name] = generate_dates_histogram(solves_data, group_date_str, tickformat, dtick)

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
        has_dates = True

        parsed = df['Time'].apply(parse_cstimer_csv_result)
        df = concat([df, parsed.apply(Series, index=['Time(millis)', 'Penalty'])], axis=1)

        df['SolveDatetime'] = to_datetime(df['Date']).astype('datetime64[s]')  # already in local timezone

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
    else:
        raise NotImplementedError('Unrecognized file type')


def drop_all_dnf_categories(solves_data):
    solves_grouped = solves_data.groupby(['Puzzle', 'Category'])['single']
    non_dnf = solves_grouped.any()
    all_dnf = non_dnf[~non_dnf].reset_index()[['Puzzle', 'Category']]
    solves_grouped_index = solves_data.set_index(['Puzzle', 'Category']).index
    all_dnf_index = all_dnf.set_index(['Puzzle', 'Category']).index
    return solves_data[~solves_grouped_index.isin(all_dnf_index)]


def process_data(file, chart_by, timezone):
    # timezone could be a tz name string, or an offset in minutes
    if represents_int(timezone):
        timezone = int(timezone) * 60  # need it in seconds for tz_convert

    solves_data, has_dates, timer_type = create_dataframe(file, timezone)

    solves_data['TimeCentiSec'] = (solves_data['Time(millis)'] / 10).astype(int)

    solves_data['single'] = [NaN if row.Penalty == 2 else (row.TimeCentiSec / 100)
                             for row in solves_data[['Penalty', 'TimeCentiSec']].itertuples()]

    solves_data['IsDNF'] = [1 if row == 2 else 0 for row in solves_data['Penalty']]

    solves_data = drop_all_dnf_categories(solves_data)

    for idx, puzzle, category in solves_data[['Puzzle', 'Category']].drop_duplicates().itertuples():
        for ao_len in (3, 5, 12, 50, 100, 1000):
            calculate_and_store_running_ao(solves_data, puzzle, category, ao_len)

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

    solves_details = get_all_solves_details(solves_data, has_dates, timezone, chart_by)
    overall_pbs = get_overall_pbs(solves_data)
    if has_dates:
        solves_by_dates = get_solves_by_dates(solves_data)
    else:
        solves_by_dates = None

    return solves_details, overall_pbs, solves_by_dates, timer_type, len(solves_data)
