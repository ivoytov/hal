# %%
import pandas as pd
import glob

# Use glob to match the pattern and return a list of file paths
all_csvs = glob.glob('../docs/trades/' + "*.csv")

# Use a list comprehension to read each csv into a DataFrame and store all the DataFrames in a list
dataframes = [pd.read_csv(f) for f in all_csvs]

# Use pd.concat to concatenate all the dataframes in the list into one DataFrame
trades = pd.concat(dataframes, ignore_index=True)

tickers = trades.ticker.unique()

# %%

# Open the HDF5 file
data = pd.read_hdf('~/data/bond_data.h5', mode='r', key='prices')

# Load the data in the 'prices' table
data = data[data.index.get_level_values('ticker').isin(tickers)]

# Export the data to a CSV file
data['price'].to_csv('../docs/px_data.csv')

# %%
