
import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots


def fetch_options(cursor, symbol, expiry):
    query = '''
        SELECT * 
        FROM ohlcv_options_per_minute oopm
        WHERE symbol = %s
        AND expiry_type = 'I'
        AND expiry = %s
        ORDER BY date_timestamp;
        '''
    cursor.execute(query,(symbol,expiry))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df


def fetch_futures(cursor, symbol, x):
    query = '''
        SELECT *
        FROM ohlcv_future_per_minute ofpm 
        WHERE ofpm.symbol = %s
        AND ofpm.expiry_type = 'I'
        AND DATE(ofpm.expiry) = %s
        ORDER BY date_timestamp ASC
    '''
    # Execute the query with parameters as a tuple
    cursor.execute(query, (symbol, x))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df

# Define the fetch_expiries function
def fetch_expiries(cursor, symbol):
    query = f'''
        SELECT DISTINCT ofpem.expiry 
        FROM ohlcv_future_per_minute ofpem 
        WHERE ofpem.symbol = '{symbol}'
        AND ofpem.expiry_type = 'I'
        GROUP BY ofpem.expiry 
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    return df



def fill_missings(futures):
    # Generate the full range of timestamps for each day
    start_time = '09:15:00'
    end_time = '15:30:00'

    all_timestamps = pd.date_range(start=start_time, end=end_time, freq='min').time

    # Generate a DataFrame with all timestamps for each day in the data
    all_dates = futures['date_timestamp'].dt.date.unique()
    all_date_times = [pd.Timestamp.combine(date, time) for date in all_dates for time in all_timestamps]
    all_date_times_df = pd.DataFrame(all_date_times, columns=['date_timestamp'])


    # Merge with the original DataFrame to include all timestamps
    futures = pd.merge(all_date_times_df, futures, on='date_timestamp', how='left')

    # Forward and backword fill missing values
    futures = futures.ffill()

    # Sort by date_timestamp to maintain order
    futures = futures.sort_values('date_timestamp').reset_index(drop=True)


def genrate_signals(futures , long_window = 26, short_window = 9):
    futures['Short_EMA'] = futures['close'].ewm(span=short_window).mean()
    futures['Long_EMA'] = futures['close'].ewm(span=long_window).mean()

    futures['Signal'] = 0 

    for i in range(26, len(futures)):
        if futures['Short_EMA'].iloc[i] > futures['Long_EMA'].iloc[i] and futures['Short_EMA'].iloc[i-1] <= futures['Long_EMA'].iloc[i-1]:
            futures.at[futures.index[i], 'Signal'] = 1  # Buy signal
        elif futures['Short_EMA'].iloc[i] < futures['Long_EMA'].iloc[i] and futures['Short_EMA'].iloc[i-1] >= futures['Long_EMA'].iloc[i-1]:
            futures.at[futures.index[i], 'Signal'] = -1  # Sell signal




def plot_signals(futures):
    
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    candlestick = go.Candlestick(
        x=futures.index,
        open=futures['open'],
        high=futures['high'],
        low=futures['low'],
        close=futures['close'],
        name='Candlesticks'
    )
    fig.add_trace(candlestick)

    # Short EMA
    short_ema = go.Scatter(x=futures.index, y=futures['Short_EMA'], mode='lines', name='Short EMA')
    fig.add_trace(short_ema)

    # Long EMA
    long_ema = go.Scatter(x=futures.index, y=futures['Long_EMA'], mode='lines', name='Long EMA')
    fig.add_trace(long_ema)

    # Buy signals
    buy_signals = go.Scatter(
        x=futures[futures['Signal'] == 1].index,
        y=futures['Short_EMA'][futures['Signal'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-up', color='yellow', size=12),
        name='Buy Signal'
    )
    fig.add_trace(buy_signals)

    # Sell signals
    sell_signals = go.Scatter(
        x=futures[futures['Signal'] == -1].index,
        y=futures['Short_EMA'][futures['Signal'] == -1],
        mode='markers',
        marker=dict(symbol='triangle-down', color='black', size=12),
        name='Sell Signal'
    )
    fig.add_trace(sell_signals)

    fig.update_layout(
        title=f'{futures[0]['Symbol']} EMA Crossover Strategy',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        width=2000,
        height=1000
    )

    return fig


def _plot(futures ):

    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    candlestick = go.Candlestick(
        x=futures['date_timestamp'],
        open=futures['open'],
        high=futures['high'],
        low=futures['low'],
        close=futures['close'],
        name='Candlesticks'
    )
    fig.add_trace(candlestick)

    # Short EMA
    short_ema = go.Scatter(x=futures['date_timestamp'], y=futures['Short_EMA'], mode='lines', name='Short EMA')
    fig.add_trace(short_ema)

    # Long EMA
    long_ema = go.Scatter(x=futures['date_timestamp'], y=futures['Long_EMA'], mode='lines', name='Long EMA')
    fig.add_trace(long_ema)

    # Buy signals
    buy_signals = go.Scatter(
        x=futures[futures['Signal'] == 1]['date_timestamp'],
        y=futures['Short_EMA'][futures['Signal'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-up', color='yellow', size=12),
        name='Buy Signal'
    )
    fig.add_trace(buy_signals)

    # Sell signals
    sell_signals = go.Scatter(
        x=futures[futures['Signal'] == -1]['date_timestamp'],
        y=futures['Short_EMA'][futures['Signal'] == -1],
        mode='markers',
        marker=dict(symbol='triangle-down', color='black', size=12),
        name='Sell Signal'
    )
    fig.add_trace(sell_signals)

    fig.update_layout(
        title=f'{futures.iloc[0]["Symbol"]} EMA Crossover Strategy',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        width=2000,
        height=1000
    )

    return fig   


import pandas as pd

def fetch_call_put(options):

    if 'id' in options.columns:
        options.drop(columns=['id'], inplace=True)
        
    options.drop_duplicates(inplace=True)
    options['date_timestamp'] = pd.to_datetime(options['date_timestamp'])
    options['liquidity'] = 0

    copt = options[options['opt_type'] == 'CE'].copy()
    popt = options[options['opt_type'] == 'PE'].copy()

    copt.drop_duplicates(inplace=True)
    popt.drop_duplicates(inplace=True)

    strike_values = options['strike'].unique()
    all_timestamps = options['date_timestamp'].unique()

    d2c = [[j, i] for j in all_timestamps for i in strike_values]
    d2c_df = pd.DataFrame(d2c, columns=['date_timestamp', 'strike'])

    ce = pd.merge(d2c_df, copt, on=['date_timestamp', 'strike'], how='left')
    pe = pd.merge(d2c_df, popt, on=['date_timestamp', 'strike'], how='left')

    ce['liquidity'] = 0
    pe['liquidity'] = 0

    return ce.ffill(), pe.ffill()





def lower_bound(arr, target):
    low , high = 0, len(arr)
    while low < high:
        mid = low + (high -low)//2
        if arr['strike'].iloc[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def drawdown_trade(arr):
    max_val = -10000000000
    max_index , i = 0 , 0
    drawdown_val = -10000000000
    while i < len(arr):
        if arr[i] > max_val:
            max_val = arr[i]
            while i < len(arr) and arr[i]<=max_val:
                drawdown_val = max(drawdown_val, max_val - arr[i])
                i += 1

    return drawdown_val


def generate_trades(futures, ce, pe, io=0, buying='open', selling='close' ,intial_capital=1000000):
    # Define the structures
    put = {'pos': 0, 'movment': [], 'sharpie_ratio': None, 'drawdown': None, 'signal_time': None, 'long_ema': None, 
           'short_ema': None, 'entry_time': None, 'strike': None, 'entry_price': None, 'exit_time': None, 
           'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0, 'price': None, 'logs': []}
    
    call = {'pos': 0, 'movment': [], 'sharpie_ratio': None, 'drawdown': None, 'signal_time': None, 'long_ema': None, 
            'short_ema': None, 'entry_time': None, 'strike': None, 'entry_price': None, 'exit_time': None, 
            'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0, 'price': None, 'logs': []}

    # Initialize trades list
    trades = []
    risk_free_per_min = 0.0
    capital = intial_capital
    rollover = 0
    for i in range(len(futures) - 1):
        row = futures.iloc[i]
        signal = row['Signal']
        price = futures.iloc[i + 1]['close']
        dt = futures.iloc[i + 1]['date_timestamp']

        if signal == 1:
            if call['pos'] == 0:
                strikes = ce[ce['date_timestamp'] == dt][['date_timestamp', 'open', 'close', 'strike', 'liquidity']]
                if not strikes.empty:
                    strikes.sort_values(by='strike', inplace=True)
                    var = price
                    strikes.reset_index(drop=True, inplace=True)
                    ind = np.argmin(np.abs(strikes['strike'] - var))

                    buy_val = strikes[buying].iloc[ind]

                    if pd.isna(buy_val):
                        ce.at[i + 1, 'liquidity'] = ce.at[i, 'liquidity'] + 1
                    else:
                        ce.at[i + 1, 'liquidity'] = ce.at[i, 'liquidity']

                    shift, t_io = (1 if io > 0 else -1), io
                    while t_io != 0:
                        if pd.notna(strikes['strike'].iloc[ind - t_io]):
                            buy_val = strikes['close'].iloc[ind - t_io]
                            break
                        t_io -= shift

                    if pd.isna(buy_val):
                        call['logs'].append(f"stock is illiquid at {i} and {dt}")
                    else:
                        if buy_val > capital:
                            call['logs'].append(f"capital not sufficient at {i} and {dt}")
                        else:
                            capital -= buy_val
                            call.update({'pos': 1, 'price': var, 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                     'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[ind],
                                     'entry_time': dt, 'entry_price': buy_val , 'movment': [buy_val]})
                else:
                    call['logs'].append(f"No strikes on call option at {i}, {dt} and strike: {call['strike']}")

            if put['pos'] == 1:
                strikes = pe[(pe['date_timestamp'] == dt) & (pe['strike'] == put['strike'])]
                if not strikes.empty:
                    sell_val = strikes[selling].iloc[0]
                    capital += sell_val
                    if capital > intial_capital:
                        rollover += (capital - intial_capital)
                        capital = intial_capital
                    put.update({'pos': 0, 'exit_time': dt, 'exit_price': sell_val, 'pnl': var - put['entry_price'], 'type': 'PE'})
                    put['movment'].append(sell_val)
                    if len(put['movment']) > 1 and np.std(put['movment']) != 0:
                        sharpie_ratio = (sell_val - put['entry_price'] - risk_free_per_min * (len(put['movment']) - 1)) / (np.std(put['movment']) + 1)
                    else:
                        sharpie_ratio = np.nan
                    drawdown_val = drawdown_trade(put['movment'])
                    put.update({'sharpie_ratio': sharpie_ratio, 'drawdown': drawdown_val})
                    trades.append(put.copy())
                    put.update({'logs': [], 'movment': []})
                else:
                    put['logs'].append(f"No closing on put option at {i}, {dt}, and strike: {put['strike']}")

        elif signal == -1:
            if put['pos'] == 0:
                strikes = pe[pe['date_timestamp'] == dt][['date_timestamp', 'open', 'close', 'strike', 'liquidity']]
                if not strikes.empty:
                    strikes.sort_values(by='strike', inplace=True)
                    var = price
                    strikes.reset_index(drop=True, inplace=True)
                    ind = np.argmin(np.abs(strikes['strike'] - var))
                    buy_val = strikes[buying].iloc[ind]

                    if pd.isna(buy_val):
                        pe.at[i + 1, 'liquidity'] = pe.at[i, 'liquidity'] + 1
                    else:
                        pe.at[i + 1, 'liquidity'] = pe.at[i, 'liquidity']

                    shift, t_io = (1 if io > 0 else -1), io
                    while t_io != 0:
                        if pd.notna(strikes['strike'].iloc[ind + t_io]):
                            buy_val = strikes['close'].iloc[ind + t_io]
                            break
                        t_io -= shift

                    if pd.isna(buy_val):
                        put['logs'].append(f"stock is illiquid at {i} and {dt}")
                    else:
                        if buy_val > capital:
                            put['logs'].append(f"capital not sufficient at {i} and {dt}")
                        else:
                            capital -= buy_val
                            put.update({'pos': 1, 'price': var, 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                    'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[ind],
                                    'entry_time': dt, 'entry_price': buy_val , 'movment': [buy_val]})
                else:
                    put['logs'].append(f"No strikes available on put option at {i}, {dt}, {put['strike']}")

            if call['pos'] == 1:
                strikes = ce[(ce['date_timestamp'] == dt) & (ce['strike'] == call['strike'])]
                if not strikes.empty:
                    sell_val = strikes[selling].iloc[0]
                    capital += sell_val
                    if capital > intial_capital:
                        rollover += (capital - intial_capital)
                        capital = intial_capital
                    call.update({'pos': 0, 'exit_time': dt, 'exit_price': sell_val, 'pnl': var - call['entry_price'], 'type': 'CE'})
                    call['movment'].append(sell_val)
                    if len(call['movment']) > 1 and np.std(call['movment']) != 0:
                        sharpie_ratio = (sell_val - call['entry_price'] - risk_free_per_min * (len(call['movment']) - 1)) / (np.std(call['movment']) + 1)
                    else:
                        sharpie_ratio = np.nan
                    drawdown_val = drawdown_trade(call['movment'])
                    call.update({'sharpie_ratio': sharpie_ratio, 'drawdown': drawdown_val})
                    trades.append(call.copy())
                    call.update({'logs': [], 'movment': []})
                else:
                    call['logs'].append(f"No closing on call option at {i}, {dt}, {call['strike']}")
        else:
            pe.at[i + 1, 'liquidity'] = pe.at[i, 'liquidity']
            ce.at[i + 1, 'liquidity'] = ce.at[i, 'liquidity']
            if call['pos'] == 1:
                call['movment'].append(ce[(ce['date_timestamp'] == dt) & (ce['strike'] == call['strike'])].iloc[0]['close'])
            if put['pos'] == 1:
                put['movment'].append(pe[(pe['date_timestamp'] == dt) & (pe['strike'] == put['strike'])].iloc[0]['close'])

    # Final closing for any open positions at the end of the dataset
    if call['pos'] == 1:
        call_short = ce[(ce['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (ce['strike'] == call['strike'])]
        if not call_short.empty:
            sell_val = call_short.iloc[0][selling]
            capital += sell_val
            if capital > intial_capital:
                rollover += (capital - intial_capital)
                capital = intial_capital
            call.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': sell_val, 'pnl': sell_val - call['entry_price']})
            call['movment'].append(sell_val)
            if len(call['movment']) > 1 and np.std(call['movment']) != 0:
                sharpie_ratio = (sell_val - call['entry_price'] - risk_free_per_min * (len(call['movment']) - 1)) / (np.std(call['movment']) + 1)
            else:
                sharpie_ratio = np.nan
            drawdown_val = drawdown_trade(call['movment'])
            call.update({'sharpie_ratio': sharpie_ratio, 'drawdown': drawdown_val})
            trades.append(call.copy())
            call.update({'logs': [], 'movment': []})
        else:
            call['logs'].append(f"No closing on call option at {futures.index[-1]}, {futures.iloc[-1]['date_timestamp']}, {call['strike']}")

    if put['pos'] == 1:
        put_short = pe[(pe['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (pe['strike'] == put['strike'])]
        if not put_short.empty:
            sell_val = put_short.iloc[0][selling]
            capital += sell_val
            if capital > intial_capital:
                rollover += (capital - intial_capital)
                capital = intial_capital
            put.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': sell_val, 'pnl': sell_val - put['entry_price']})
            put['movment'].append(sell_val)
            if len(put['movment']) > 1 and np.std(put['movment']) != 0:
                sharpie_ratio = (sell_val - put['entry_price'] - risk_free_per_min * (len(put['movment']) - 1)) / (np.std(put['movment']) + 1)
            else:
                sharpie_ratio = np.nan
            drawdown_val = drawdown_trade(put['movment'])
            put.update({'sharpie_ratio': sharpie_ratio, 'drawdown': drawdown_val})
            trades.append(put.copy())
            put.update({'logs': [], 'movment': []})
        else:
            put['logs'].append(f"No closing on put option at {futures.index[-1]}, {futures.iloc[-1]['date_timestamp']}, {put['strike']}")

    if trades:
        trades[0]['pnl_sum'] = trades[0]['pnl']
        for i in range(1, len(trades)):
            trades[i]['pnl_sum'] = trades[i - 1]['pnl_sum'] + trades[i]['pnl']

    return trades , capital, rollover

def generate_trades_(futures,ce, pe, initial_capital):

    # Define the structures
    put = {'pos': 0, 'signal_time': None, 'long_ema': None, 'short_ema': None, 'entry_time': None, 'strike': None,
           'entry_price': None, 'exit_time': None, 'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0, 'price': None, 'rollover': 0}
    call = {'pos': 0, 'signal_time': None, 'long_ema': None, 'short_ema': None, 'entry_time': None, 'strike': None,
            'entry_price': None, 'exit_time': None, 'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0, 'price': None, 'rollover': 0}

    # Initialize trades list and capital
    trades = []
    capital = initial_capital

    for i in range(len(futures) - 1):
        row = futures.iloc[i]
        signal = row['Signal']
        price_selector = 1
        price = futures.iloc[i + 1]['close'] if price_selector == 1 else futures.iloc[i + 1]['open']
        dt = futures.iloc[i + 1]['date_timestamp']

        if signal == 1:
            if call['pos'] == 0:
                strikes = ce[ce['date_timestamp'] == dt][['date_timestamp', 'open', 'strike']]
                if not strikes.empty:
                    strikes.sort_values(by='strike', inplace=True)
                    var = price
                    strikes.reset_index(drop=True, inplace=True)
                    index = lower_bound(strikes, var)
                    if index >= len(strikes): index -= 1
                    if index > 0: index -= 1
                    buy_val = strikes['open'].iloc[index]
                    if buy_val <= capital:
                        capital -= buy_val
                        call.update({'pos': 1, 'price': row['close'], 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                     'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[index],
                                     'entry_time': dt, 'entry_price': buy_val})
                    else:
                        print(f'Not enough capital to buy call option at {i} and {dt}')
                else:
                    print(f'No strikes on call option at {i} and {dt}')

            if put['pos'] == 1:
                strikes = pe[(pe['date_timestamp'] == dt) & (pe['strike'] == put['strike'])]
                if not strikes.empty:
                    var = strikes['open'].iloc[0]
                    capital += var
                    if capital > initial_capital:
                        rollover = capital - initial_capital
                        capital = initial_capital
                    else:
                        rollover = 0
                    put.update({'pos': 0, 'exit_time': dt, 'exit_price': var, 'pnl': var - put['entry_price'], 'type': 'PE', 'rollover': rollover})
                    trades.append(put.copy())
                else:
                    print(f'No closing on put option at {i}, {dt}, {put["strike"]}')

        elif signal == -1:
            if put['pos'] == 0:
                strikes = pe[pe['date_timestamp'] == dt][['date_timestamp', 'open', 'strike']]
                if not strikes.empty:
                    strikes.sort_values(by='strike', inplace=True)
                    var = price
                    strikes.reset_index(drop=True, inplace=True)
                    index = lower_bound(strikes, var)
                    if index >= len(strikes): index -= 1
                    if index < len(strikes): index += 1
                    sell_val = strikes['open'].iloc[index]
                    if sell_val <= capital:
                        capital -= sell_val
                        put.update({'pos': 1, 'price': row['close'], 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                    'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[index],
                                    'entry_time': dt, 'entry_price': sell_val})
                    else:
                        print(f'Not enough capital to buy put option at {i} and {dt}')
                else:
                    print(f'No strikes on put option at {i} and {dt}')
            if call['pos'] == 1:
                strikes = ce[(ce['date_timestamp'] == dt) & (ce['strike'] == call['strike'])]
                if not strikes.empty:
                    var = strikes['open'].iloc[0]
                    capital += var
                    if capital > initial_capital:
                        rollover = capital - initial_capital
                        capital = initial_capital
                    else:
                        rollover = 0
                    call.update({'pos': 0, 'exit_time': dt, 'exit_price': var, 'pnl': var - call['entry_price'], 'type': 'CE', 'rollover': rollover})
                    trades.append(call.copy())
                else:
                    print(f'No closing on call option at {i}, {dt}, {call["strike"]}')

    # Final closing for any open positions at the end of the dataset
    if call['pos'] == 1:
        call_short = ce[(ce['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (ce['strike'] == call['strike'])]
        if not call_short.empty:
            var = call_short.iloc[0]['close']
            capital += var
            if capital > initial_capital:
                rollover = capital - initial_capital
                capital = initial_capital
            else:
                rollover = 0
            call.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': var, 'pnl': var - call['entry_price'], 'rollover': rollover})
            trades.append(call.copy())
        else:
            print('No closing on call option')

    if put['pos'] == 1:
        put_short = pe[(pe['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (pe['strike'] == put['strike'])]
        if not put_short.empty:
            var = put_short.iloc[0]['close']
            capital += var
            if capital > initial_capital:
                rollover = capital - initial_capital
                capital = initial_capital
            else:
                rollover = 0
            put.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': var, 'pnl': var - put['entry_price'], 'rollover': rollover})
            trades.append(put.copy())
        else:
            print('No closing on put option')
    trades[0]['pnl_sum'] = trades[0]['pnl']
    for i in range(1,len(trades)):
        trades[i]['pnl_sum'] = trades[i-1]['pnl_sum'] + trades[i]['pnl']
    
    return trades

def grpah(x,y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        name='PnL'
    ))

    fig.show()



def drawdown(arr):
    max_val = -10000000000
    max_index , i = 0 , 0
    drawdown_val = -10000000000
    while i < len(arr):
        if arr[i]['pnl_sum'] > max_val:
            max_val = arr[i]['pnl_sum']
            while i < len(arr) and arr[i]['pnl_sum']<=max_val:
                drawdown_val = max(drawdown_val, max_val - arr[i]['pnl_sum'])
                i += 1

    return drawdown_val

def sharpie(trades):

    entry_per_day , exit_per_day , percentage_pnl = {} , {} , {}

    for trade in trades:
        if trade['entry_time'] not in entry_per_day: entry_per_day[trade['entry_time']] = 0
        if trade['exit_time'] not in exit_per_day: exit_per_day[trade['exit_time']] = 0
        entry_per_day[trade['entry_time']] += trade['entry_price']
        exit_per_day[trade['exit_time']] += trade['exit_price']

    for day in entry_per_day:
        if day not in exit_per_day: exit_per_day[day] = 0
        if day not in entry_per_day: entry_per_day[day] = 0
        percentage_pnl[day] = ((exit_per_day[day]- entry_per_day[day]) / entry_per_day[day]) * 100
    arr = np.array(list(percentage_pnl.values()))
    mean , std_dev , risk_free_rate = np.mean(arr) , np.std(arr) ,0.0

    sharpie_ratio = (mean - risk_free_rate) / std_dev

    return sharpie_ratio




def print_trades(trades, columns_to_print):
    for trade in trades:
        output = []
        if columns_to_print.get('pos', 0):
            output.append(f"pos {trade['pos']}")
        if columns_to_print.get('signal_time', 0):
            output.append(f"signal time {trade['signal_time']}")
        if columns_to_print.get('long_ema', 0):
            output.append(f"long_ema {trade['long_ema']}")
        if columns_to_print.get('short_ema', 0):
            output.append(f"short_ema {trade['short_ema']}")
        if columns_to_print.get('entry_time', 0):
            output.append(f"entry time {trade['entry_time']}")
        if columns_to_print.get('strike', 0):
            output.append(f"strike {trade['strike']}")
        if columns_to_print.get('entry_price', 0):
            output.append(f"entry price {trade['entry_price']}")
        if columns_to_print.get('exit_time', 0):
            output.append(f"exit time {trade['exit_time']}")
        if columns_to_print.get('exit_price', 0):
            output.append(f"exit price {trade['exit_price']}")
        if columns_to_print.get('pnl', 0):
            output.append(f"pnl {trade['pnl']}")
        if columns_to_print.get('type', 0):
            output.append(f"type {trade['type']}")
        if columns_to_print.get('pnl_sum', 0):
            output.append(f"pnl_sum {trade['pnl_sum']}")
        if columns_to_print.get('price', 0):
            output.append(f"price {trade['price']}")
        if columns_to_print.get('logs', 0):
            output.append(f"logs {trade['logs']}")
        if columns_to_print.get('movment', 0):
            output.append(f"movment {trade['movment']}")
        if columns_to_print.get('sharpie_ratio', 0):
            output.append(f"sharpie_ratio {trade['sharpie_ratio']}")
        if columns_to_print.get('drawdown', 0):
            output.append(f"drawdown {trade['drawdown']}")
        
        print(', '.join(output))