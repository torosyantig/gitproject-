import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter("ignore")
%matplotlib inline

sns.set(style="darkgrid")


data = pd.read_csv("exam_data.csv")
data.head()

plt.figure(figsize=(60, 50))
sns.heatmap(data.corr(),
cmap = 'BrBG',
fmt = '.2f',
linewidths = 2,
annot = True)
