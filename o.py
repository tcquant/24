import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
def run_query(query):
    try:
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(
            dbname="qdap_test",
            user="amt",
            #password="your_password",
            host="192.168.2.23",
            port="5432"
        )

        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()


    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)

    finally:
        return rows
        # Close the cursor and connection
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
def future_data(stock,start,end):
    query = f"""
            select * from ohlcv_future_per_minute
            WHERE symbol = '{stock}'
            AND expiry_type = 'I'
            AND date_timestamp >= TIMESTAMP '{start} 09:15:00.000'
            AND date_timestamp <= TIMESTAMP '{end} 15:29:00.000'
            order by date_timestamp asc;
        """
    row = run_query(query)
    columns = """ SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'ohlcv_future_per_minute' 
    ORDER BY ordinal_position """
    columns = run_query(columns)
    cleaned_columns = []
    for column in columns:
        cleaned_column = column[0].replace('(', '').replace(',', '').replace("'", '')
        cleaned_columns.append(cleaned_column)
    row =  pd.DataFrame(row,columns = cleaned_columns)
    return row

def cm_data(stock,start,end):
    query = f"""
            select * from ohlcv_cm_per_minute
            WHERE symbol = '{stock}'
            AND date_timestamp >= TIMESTAMP '{start} 09:15:00.000'
            AND date_timestamp <= TIMESTAMP '{end} 15:29:00.000'
            order by date_timestamp asc;
        """
    rows = run_query(query)

    
    columns = """ SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'ohlcv_cm_per_minute' 
    ORDER BY ordinal_position """
    columns = run_query(columns)
    cleaned_columns = []
    for column in columns:
        cleaned_column = column[0].replace('(', '').replace(',', '').replace("'", '')
        cleaned_columns.append(cleaned_column)
    rows =  pd.DataFrame(rows,columns = cleaned_columns)
    return rows

def option_data(stock,start,end):
    query = f"""
            select * from ohlcv_options_per_minute
            WHERE symbol = '{stock}'
            AND expiry_type = 'I'
            AND date_timestamp >= TIMESTAMP '{start}'
            AND date_timestamp <= TIMESTAMP '{end}'
            order by date_timestamp asc;
        """
    row = run_query(query)
    columns = """ SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'ohlcv_options_per_minute' 
    ORDER BY ordinal_position """
    columns = run_query(columns)
    cleaned_columns = []
    for column in columns:
        cleaned_column = column[0].replace('(', '').replace(',', '').replace("'", '')
        cleaned_columns.append(cleaned_column)
    row =  pd.DataFrame(row,columns = cleaned_columns)
    return row
### Fetching data and removing dublicates
stock = 'POWERGRID'
start = "2024-01-01"
end = "2024-07-01"
df_fut = future_data(stock,start,end)
df_cm = cm_data(stock,start,end)
df_fut.set_index('date_timestamp',inplace = True)
df_cm.set_index('date_timestamp',inplace = True)
df_fut = df_fut[~df_fut.index.duplicated(keep='first')]
df_cm = df_cm[~df_cm.index.duplicated(keep='first')]
##### Close price of futures
df_fut_close = pd.merge(df_cm, df_fut, how='outer', left_index=True, right_index=True).ffill()
df_fut_close.rename(columns={'close_y': 'close'}, inplace=True)
df_fut_close = df_fut_close[['close']]
#(df_fut_close) = (df_fut_close)[~(df_fut_close).index.duplicated(keep='first')]
df_fut_close
close
date_timestamp	
2024-01-01 09:15:00	23865.0
2024-01-01 09:16:00	23900.0
2024-01-01 09:17:00	23845.0
2024-01-01 09:18:00	23865.0
2024-01-01 09:19:00	23870.0
...	...
2024-07-01 15:25:00	33050.0
2024-07-01 15:26:00	33070.0
2024-07-01 15:27:00	33070.0
2024-07-01 15:28:00	33070.0
2024-07-01 15:29:00	33060.0
45750 rows × 1 columns

plt.figure(figsize = (20,10))
plt.plot(df_fut_close['close'].values, color = 'orange', label = 'future market')
plt.plot(df_cm['close'].values, color = 'green', label = 'cash market')
plt.legend()
plt.show()
No description has been provided for this image
## Signal Geneartion
fast_window = 10
slow_window = 40
df_fut_close['fast'] = df_fut_close['close'].ewm(span=fast_window).mean()
df_fut_close['slow'] = df_fut_close['close'].ewm(span=slow_window).mean()
df_fut_close['diff'] = df_fut_close['slow'] - df_fut_close['fast']
df_fut_close['signal'] = 0
prev = 0
for index, row in df_fut_close.iterrows():
    if prev < 0 and row['diff'] > 0:
         df_fut_close.loc[index, 'signal'] = -1
    if prev > 0 and row['diff'] < 0:
         df_fut_close.loc[index, 'signal'] = 1
    prev = row['diff']

signals = df_fut_close[['signal','close']]
signals = signals[signals['signal'] != 0]
signals
signal	close
date_timestamp		
2024-01-01 09:17:00	-1	23845.0
2024-01-01 09:20:00	1	23905.0
2024-01-01 09:43:00	-1	23880.0
2024-01-01 09:58:00	1	23935.0
2024-01-01 10:10:00	-1	23880.0
...	...	...
2024-07-01 13:14:00	-1	33035.0
2024-07-01 13:46:00	1	33065.0
2024-07-01 14:48:00	-1	33065.0
2024-07-01 15:03:00	1	33125.0
2024-07-01 15:10:00	-1	33050.0
1439 rows × 2 columns

import plotly.express as px
import pandas as pd
fig = px.line(df_fut_close, x=df_fut_close.index, y='close', title='Close Prices with Signals')

# Add scatter markers for buy (signal == 1) and sell (signal == -1)
fig.add_scatter(x=df_fut_close[df_fut_close['signal'] == 1].index,
                y=df_fut_close[df_fut_close['signal'] == 1]['close'],
                mode='markers', marker=dict(color='green', size=10), name='Buy Signal')

fig.add_scatter(x=df_fut_close[df_fut_close['signal'] == -1].index,
                y=df_fut_close[df_fut_close['signal'] == -1]['close'],
                mode='markers', marker=dict(color='red', size=10), name='Sell Signal')

# Update layout
fig.update_layout(
    xaxis_title='Index',
    yaxis_title='Close Price',
    legend_title='Signals',
    showlegend=True,
    height=600,
    width=1000,
)

# Show plot
fig.show()
THE BACKTESTING PART
stock = 'POWERGRID'
start = "2023-12-25"
end = "2024-07-07"
opt_main = option_data(stock,start,end).set_index('date_timestamp')
opt_main
symbol	open	high	low	close	volume	opt_type	strike	expiry_type	id	expiry
date_timestamp											
2023-12-26 09:15:00	POWERGRID	5	5	5	5	36000	PE	19000	I	156822423	2023-12-28
2023-12-26 09:15:00	POWERGRID	50	50	40	40	10800	PE	22500	I	156822816	2023-12-28
2023-12-26 09:15:00	POWERGRID	360	370	285	325	75600	CE	23000	I	156823012	2023-12-28
2023-12-26 09:15:00	POWERGRID	155	175	135	135	21600	PE	23000	I	156823215	2023-12-28
2023-12-26 09:15:00	POWERGRID	225	230	205	205	21600	CE	23250	I	156823415	2023-12-28
...	...	...	...	...	...	...	...	...	...	...	...
2024-07-05 15:29:00	POWERGRID	550	555	550	555	25200	CE	34500	I	307779607	2024-07-25
2024-07-05 15:29:00	POWERGRID	375	385	375	375	25200	CE	35000	I	307779943	2024-07-25
2024-07-05 15:29:00	POWERGRID	255	255	255	255	86400	CE	35500	I	307780166	2024-07-25
2024-07-05 15:29:00	POWERGRID	180	180	180	180	28800	CE	36000	I	307780331	2024-07-25
2024-07-05 15:29:00	POWERGRID	55	55	55	55	21600	CE	38000	I	307780597	2024-07-25
643923 rows × 11 columns

# opt = opt_main.copy()
# CE_main = (opt[opt['opt_type'] == 'CE' ]).copy()
# PE_main = (opt[opt['opt_type'] == 'PE' ]).copy()
opt = opt_main.copy()
CE_main = (opt[opt['opt_type'] == 'CE' ]).copy()
PE_main = (opt[opt['opt_type'] == 'PE' ]).copy()
#opt_main = option_data(stock,start,end).set_index('date_timestamp')
positions = pd.DataFrame({
    'option_type':[],
    'singnal_time': [],
    'strike':[],
    'entry_time':[],
    'entry_price':[],
    'exit_time':[],
    'exit_price':[],
})
def close_position(df, strike, time):
    while(df[time:time].empty):
        print(time)
        time = time + timedelta(minutes=1)
    close = df.loc[time, strike]
    positions.loc[positions.index[-1], 'exit_time'] = time
    positions.loc[positions.index[-1], 'exit_price'] = close

# opt = opt_main.copy()
# CE = (opt[opt['opt_type'] == 'CE' ])
# PE = (opt[opt['opt_type'] == 'PE' ])
# CE = (CE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())
# PE = (PE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())


i = 0
for time, row in signals.iterrows():
        entry = time
        signal = row['signal']
        close = row['close']
    
        
        start = entry - timedelta(days=3)
        start = (pd.Timestamp(start.date()) + pd.Timedelta(hours=9, minutes=15)) #.strftime('%Y-%m-%d %H:%M:%S')
        end = (pd.Timestamp(entry.date()) + pd.Timedelta(hours=15, minutes=29)) #.strftime('%Y-%m-%d %H:%M:%S')
    
        # opt = option_data(stock,start,end).set_index('date_timestamp')
        # opt = opt_main.copy()
        # CE = (opt[opt['opt_type'] == 'CE' ])[start:end]
        # PE = (opt[opt['opt_type'] == 'PE' ])[start:end]
        CE = CE_main[start:end].copy()
        PE = PE_main[start:end].copy()
        CE = (CE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())
        PE = (PE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())
    
    
        if(signal == 1 and (time.time() <= pd.to_datetime('15:00:00').time())):
            if(not positions.empty and (time.time() <= pd.to_datetime('15:20:00').time())):
                option_type = positions.iloc[-1]['option_type']
                strike = positions.iloc[-1]['strike']
                time_ev = time + timedelta(minutes=1)
                # print(f"option type = {option_type} and strike = {strike} and time = {time_ev}")
            
                if(option_type == 'CE'):
                    close_position(CE, strike, time_ev)
                if(option_type == 'PE'):
                    close_position(PE, strike, time_ev)
            if((time.time() <= pd.to_datetime('15:00:00').time())): 
                # CE = (CE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())
                CE = CE[entry + timedelta(minutes=1):]
                price_difference = abs(CE.columns - close)
                price_difference = price_difference.tolist()
                min_index = price_difference.index(min(price_difference))
        
                strike = CE.columns[min_index]
                entry_time = CE[strike].first_valid_index()
                entry_price = CE.loc[entry_time, strike]
        
                # print(f'strike {strike} and entry time = {entry_time} entry_price {entry_price}')
        
                data = pd.DataFrame({
                    'option_type':['CE'],
                    'singnal_time': [time],
                    'strike':[strike],
                    'entry_time':[entry_time],
                    'entry_price':[entry_price],
                    'exit_time':[np.nan],
                    'exit_price':[np.nan],
                })
                positions = pd.concat([positions, data], ignore_index=True)
                i = i + 1
                print(f'{i}th iteration is done')
            
        if(signal == -1 and (time.time() <= pd.to_datetime('15:00:00').time())):
            if(not positions.empty and (time.time() <= pd.to_datetime('15:20:00').time())):
                option_type = positions.iloc[-1]['option_type']
                strike = positions.iloc[-1]['strike']
                time_ev = time + timedelta(minutes=1)
                #print(f"option type = {option_type} and strike = {strike} and time = {time_ev}")
            
                if(option_type == 'CE'):
                    close_position(CE, strike, time_ev)
                if(option_type == 'PE'):
                    close_position(PE, strike, time_ev)
            if((time.time() <= pd.to_datetime('15:00:00').time())):
                # PE = (PE.pivot_table(index='date_timestamp', columns='strike', values='close', aggfunc='first').ffill())
                PE = PE[entry + timedelta(minutes=1):]
                price_difference = abs(PE.columns - close)
                price_difference = price_difference.tolist()
                min_index = price_difference.index(min(price_difference))
        
                strike = PE.columns[min_index]
                entry_time = PE[strike].first_valid_index()
                entry_price = PE.loc[entry_time, strike]
        
                data = pd.DataFrame({
                    'option_type':['PE'],
                    'singnal_time': [time],
                    'strike':[strike],
                    'entry_time':[entry_time],
                    'entry_price':[entry_price],
                    'exit_time':[np.nan],
                    'exit_price':[np.nan],
                })
                positions = pd.concat([positions, data], ignore_index=True) 
        i = i + 1
        print(f'{i}th iteration is done')
print("DONE!!!")
