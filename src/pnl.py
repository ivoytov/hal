import pandas as pd
import numpy as np
import sys

filename = sys.argv[0]
df = pd.read_csv(filename, index_col=0, header=None)

returns = df[7]/(df[7].shift() + df[9])
returns
returns.dropna(inplace=True)
log_returns = np.log(returns)
print(log_returns.cumsum().apply(np.exp))
