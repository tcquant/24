import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
def fetch_options(cursor, symbol, expiry):
    cursor.execute(
        f'''
        SELECT * 
        FROM ohlcv_options_per_minute oopm
        WHERE symbol = '{symbol}'
        AND expiry_type = 'I'
        AND expiry = '{expiry}'
        ORDER BY date_timestamp;
        '''
    )
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df


def fetch_futures(cursor, symbol, x=12):
    query = f'''
        SELECT *
        FROM ohlcv_future_per_minute ofpm 
        WHERE ofpm.symbol = '{symbol}'
        AND ofpm.expiry_type = 'I'
        AND ofpm.expiry = (
            SELECT ofpem.expiry 
            FROM ohlcv_future_per_minute ofpem 
            WHERE ofpem.symbol = '{symbol}'
            AND ofpem.expiry_type = 'I'
            GROUP BY ofpem.expiry 
            OFFSET {x} 
            LIMIT 1
        )
        ORDER BY date_timestamp ASC
    '''
    cursor.execute(query, (symbol, symbol, x))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
conn = psycopg2.connect(
           dbname="qdap_test",
           user="amt",
           password="your_password",
           host="192.168.2.23",
           port="5432"
       )

# Create a cursor object using the cursor() method
cursor = conn.cursor()
symbol = 'BANKNIFTY'
df = fetch_futures(cursor, symbol)
df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
df.set_index('date_timestamp', inplace=True)
print(df.columns)
df.head()
symbol = 'BANKNIFTY'
nft = df[df['expiry']=='2023-05-25 14:30:00']

nft = nft.drop(columns=['id'])
nft.drop_duplicates()
nft
short_window = 9
long_window = 26
nft['Short_EMA'] = nft['close'].ewm(span=short_window, adjust=False).mean()
nft['Long_EMA'] = nft['close'].ewm(span=long_window, adjust=False).mean()

nft['Signal'] = 0
nft['Position'] = 0
for i in range(1, len(nft)):
    if nft['Short_EMA'].iloc[i] > nft['Long_EMA'].iloc[i] and nft['Short_EMA'].iloc[i-1] <= nft['Long_EMA'].iloc[i-1]:
        nft.at[nft.index[i], 'Signal'] = 1  # Buy signal
    elif nft['Short_EMA'].iloc[i] < nft['Long_EMA'].iloc[i] and nft['Short_EMA'].iloc[i-1] >= nft['Long_EMA'].iloc[i-1]:
        nft.at[nft.index[i], 'Signal'] = -1  # Sell signal

nft.head()
nft.head()
fig = make_subplots(rows=1, cols=1)

# Candlestick chart
candlestick = go.Candlestick(x=nft.index,
                             open=nft['open'],
                             high=nft['high'],
                             low=nft['low'],
                             close=nft['close'],
                             name='Candlesticks')
fig.add_trace(candlestick)

# Short EMA
short_ema = go.Scatter(x=nft.index, y=nft['Short_EMA'], mode='lines', name='Short EMA')
fig.add_trace(short_ema)

# Long EMA
long_ema = go.Scatter(x=nft.index, y=nft['Long_EMA'], mode='lines', name='Long EMA')
fig.add_trace(long_ema)

# Buy signals
buy_signals = go.Scatter(x=nft[nft['Signal'] == 1].index, 
                         y=nft['Short_EMA'][nft['Signal'] == 1], 
                         mode='markers', 
                         marker=dict(symbol='triangle-up', color='blue', size=8), 
                         name='Buy Signal')
fig.add_trace(buy_signals)

# Sell signals
sell_signals = go.Scatter(x=nft[nft['Signal'] == -1].index, 
                          y=nft['Short_EMA'][nft['Signal'] == -1], 
                          mode='markers', 
                          marker=dict(symbol='triangle-down', color='black', size=8), 
                          name='Sell Signal')
fig.add_trace(sell_signals)


fig.update_layout(title=f'{symbol} EMA Crossover Strategy',
                  yaxis_title='Price',
                  xaxis_title='Date',
                  xaxis_rangeslider_visible=False,
                  width=1440,  # Adjust the width as needed
                  height=400)  # Adjust the height as needed

# Show plot
fig.show()
position_size = 1  
current_position = 0  # Track current position (1 for long, -1 for short, 0 for neutral)
entry_price = 0  # Track entry price
total_pnl = 0  # Total P&L

# Iterate through each row in the DataFrame
for index, row in nft.iterrows():
    if row['Signal'] == 1 and current_position == 0:  # Buy signal and no current position
        current_position = 1
        # print("entry")
        entry_price = row['close']
    elif row['Signal'] == -1 and current_position == 1:  # Sell signal and long position
        pnl = (row['close'] - entry_price) * position_size
        total_pnl += pnl
        # ###print("close")
        current_position = 0
        entry_price = 0
    elif row['Signal'] == -1 and current_position == 0:  # Sell signal but no current long position
        continue  # Can't sell if not long
        
# Print total P&L
print(f'Total P&L: {total_pnl}')
Total P&L: 51180
ex = df['expiry'].iloc[0]  # Use iloc to access the first element by position

# Assuming fetch_options is defined to fetch data based on 'symbol' and 'expiry'
cursor1 = conn.cursor()
options = fetch_options(cursor1, symbol=symbol, expiry=ex)
copt = options[options['opt_type']=='CE'].copy()
popt = options[options['opt_type']=='PE'].copy()

# options.head()
copt['date_timestamp'] = pd.to_datetime(copt['date_timestamp'])
popt['date_timestamp'] = pd.to_datetime(popt['date_timestamp'])
print(nft.columns)
dts = nft.index
put = {'pos': 0, 'dts': None, 'strike': None}
call = {'pos': 0, 'dts': None, 'strike': None}
trades = []

for i in range(len(nft) - 1):
    row = nft.iloc[i]  # Add this line to define 'row'
    signal = row['Signal']
    close = nft.iloc[i + 1]['close']
    dt = dts[i + 1]

    if signal == 1:
        print('s1',end='-> ')
        atm_call = copt[(copt['date_timestamp'] == dt) & (abs(copt['strike'] - close) == abs(copt['strike'] - close).min())]

        if not atm_call.empty: 
            print('long on call',end=' ')
            trades.append({
                'position_type': 'long',
                'opt_type': 'CE',
                'price': atm_call.iloc[0]['close']
            })

            call['pos'] = 1
            call['strike'] = atm_call.iloc[0]['strike']

        if put['pos'] == 1:
            
            put_short = popt[(popt['date_timestamp'] == dt) & (popt['strike'] == put['strike'])]
            if not put_short.empty: 
                print(' => short on put',end=' ')
                trades.append({
                    'position_type': 'short',
                    'opt_type': 'PE',
                    'price': put_short.iloc[0]['close']
                })
                put['pos'] = 0
            else:
                print('no closing on put opt ' , end=' ')
        print('',end='\n')
    elif signal == -1:
        print('s2',end='-> ')
        atm_put = popt[(popt['date_timestamp'] == dt) & (abs(popt['strike'] - close) == abs(popt['strike'] - close).min())]

        if not atm_put.empty:  
            print('long on put ',end=' ')
            trades.append({
                'position_type': 'long',
                'opt_type': 'PE',
                'price': atm_put.iloc[0]['close']
            })

            put['pos'] = 1
            put['strike'] = atm_put.iloc[0]['strike']

        if call['pos'] == 1:
            call_short = copt[(copt['date_timestamp'] == dt) & (copt['strike'] == call['strike'])]
            if not call_short.empty:  
                print('short on call opt',end=' ')
                trades.append({
                    'position_type': 'short',
                    'opt_type': 'CE',  # This should be 'CE' not 'PE'
                    'price': call_short.iloc[0]['close']
                })
                call['pos'] = 0
            else:
                print('no closing on call opt ' , end=' ')
        print('',end='\n')
pnl = 0

for it in trades:
    print(f"{it['position_type']} on {it['opt_type']} with strike price = {it['price']}")
    if it['position_type']=='long':
        pnl -= it['price']
    else:
        pnl += it['price']
# Ensure trades_df is sorted by date
trades_df = trades_df.sort_values(by='date')

# Calculate daily returns
trades_df['daily_return'] = trades_df['price'].pct_change()

# Fill NaN values with 0 for the first trade
trades_df['daily_return'] = trades_df['daily_return'].fillna(0)

# Calculate cumulative returns
trades_df['cumulative_return'] = (1 + trades_df['daily_return']).cumprod() - 1

# Assume risk-free rate is 0 for simplicity
risk_free_rate = 0

# Calculate the mean and standard deviation of daily returns
mean_daily_return = trades_df['daily_return'].mean()
std_daily_return = trades_df['daily_return'].std()

# Calculate the Sharpe ratio
sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return

# Calculate the total PnL
total_pnl = pnl

# Assume initial investment is the sum of all buy trades
initial_investment = trades_df[trades_df['action'].isin(['buy', 'short'])]['price'].sum()

# Calculate percentage PnL
percentage_pnl = (total_pnl / initial_investment) * 100

# Display the results
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Percentage PnL: {percentage_pnl}%")
Sharpe Ratio: 0.019993927283579172
Percentage PnL: 0.33622513921916514%
