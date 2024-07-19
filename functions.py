
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



def fill_missings(futures , long_window = 26, short_window = 9):
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
    futures = futures.ffill().bfill()

    # Sort by date_timestamp to maintain order
    futures = futures.sort_values('date_timestamp').reset_index(drop=True)


def genrate_signals(futures , long_window = 26, short_window = 9):
    futures['Short_EMA'] = futures['close'].ewm(span=short_window, adjust=False).mean()
    futures['Long_EMA'] = futures['close'].ewm(span=long_window, adjust=False).mean()

    futures['Signal'] = 0 

    for i in range(1, len(futures)):
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
        title=f'{symbol} EMA Crossover Strategy',
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


def fetch_call_put(options):

    options.drop(columns=['id'],inplace = True)
    options.drop_duplicates(inplace=True)

    options['date_timestamp'] = pd.to_datetime(options['date_timestamp'])
    copt = options[options['opt_type']=='CE'].copy()
    popt = options[options['opt_type']=='PE'].copy()

    copt.drop_duplicates(inplace = True)

    popt.drop_duplicates(inplace = True)

    strike_values = options['strike'].unique()
    all_timestamps = options['date_timestamp'].unique()

    d2c = [[j, i]  for j in all_timestamps for i in strike_values]

    d2c_df = pd.DataFrame(d2c , columns = ['date_timestamp' , 'strike'])

    ce = pd.merge(d2c_df , copt,on=['date_timestamp','strike'] , how='left')
    pe = pd.merge(d2c_df , popt,on=['date_timestamp','strike'] , how='left')

    return ce.ffill().bfill() , pe.ffill().bfill() 



def lower_bound(arr, target):
    low , high = 0, len(arr)
    while low < high:
        mid = low + (high -low)//2
        if arr['strike'].iloc[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low



def generate_trades(futures,ce,pe):

    # Define the structures
    put = {'pos': 0, 'signal_time': None, 'long_ema': None, 'short_ema': None, 'entry_time': None, 'strike': None,
            'entry_price': None, 'exit_time': None, 'exit_price': None , 'pnl': 0 ,'type':None, 'pnl_sum':0 , 'price':None}
    call = {'pos': 0, 'signal_time': None, 'long_ema': None, 'short_ema': None, 'entry_time': None, 'strike': None,
            'entry_price': None, 'exit_time': None, 'exit_price': None , 'pnl': 0 ,'type':None, 'pnl_sum':0 , 'price':None}

    # Initialize trades list
    trades = []

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
                    if index >= len(strikes):  index -= 1
                    if index > 0:  index -= 1
                    buy_val = strikes['open'].iloc[index]
                    call.update({'pos': 1, 'price': row['close'], 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[index],
                                'entry_time': dt, 'entry_price': buy_val})
                else:
                    print(f'No strikes on call option at {i} and {dt}')

            if put['pos'] == 1:
                strikes = pe[(pe['date_timestamp'] == dt) & (pe['strike'] == put['strike'])]
                if not strikes.empty:
                    var = strikes['open'].iloc[0]
                    put.update({'pos': 0, 'exit_time': dt, 'exit_price': var, 'pnl': var - put['entry_price'], 'type': 'PE'})
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
                    if index >= len(strikes):   index -= 1
                    if index < len(strikes):   index += 1
                    sell_val = strikes['open'].iloc[index]
                    put.update({'pos': 1, 'price': row['close'], 'signal_time': row['date_timestamp'], 'long_ema': row['Long_EMA'],
                                'short_ema': row['Short_EMA'], 'strike': strikes['strike'].iloc[index],
                                'entry_time': dt, 'entry_price': sell_val})
                else:
                    print(f'No strikes on put option at {i} and {dt}')
            if call['pos'] == 1:
                strikes = ce[(ce['date_timestamp'] == dt) & (ce['strike'] == call['strike'])]
                if not strikes.empty:
                    var = strikes['open'].iloc[0]
                    call.update({'pos': 0, 'exit_time': dt, 'exit_price': var, 'pnl': var - call['entry_price'], 'type': 'CE'})
                    trades.append(call.copy())
                else:
                    print(f'No closing on call option at {i}, {dt}, {call["strike"]}')

    # Final closing for any open positions at the end of the dataset
    if call['pos'] == 1:
        call_short = ce[(ce['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (ce['strike'] == call['strike'])]
        if not call_short.empty:
            var = call_short.iloc[0]['close']
            call.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': var, 'pnl': var - call['entry_price']})
            trades.append(call.copy())
        else:
            print('No closing on call option')

    if put['pos'] == 1:
        put_short = pe[(pe['date_timestamp'] == futures.iloc[-1]['date_timestamp']) & (pe['strike'] == put['strike'])]
        if not put_short.empty:
            var = put_short.iloc[0]['close']
            put.update({'pos': 0, 'exit_time': futures.iloc[-1]['date_timestamp'], 'exit_price': var, 'pnl': var - put['entry_price']})
            trades.append(put.copy())
        else:
            print('No closing on put option')

    trades[0]['pnl_sum'] = trades[0]['pnl']
    for i in range(1,len(trades)):
        trades[i]['pnl_sum'] = trades[i-1]['pnl_sum'] + trades[i]['pnl']
    
    return trades




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

    sharpe_ratio = (mean - risk_free_rate) / std_dev

    return sharpe_ratio