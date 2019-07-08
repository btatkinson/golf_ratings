import numpy as np
import pandas as pd
import sys
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns
import math
import cartopy.crs as ccrs
from tqdm import tqdm
import random

yby = pd.read_csv('./data/yby_data.csv')

fig, ax = plt.subplots(figsize=(15,7))

elo_y = yby.Elo.values
relo_y = yby.rElo.values
glicko_y = yby.Glicko.values
l5_y = yby.L5.values

x = yby.Season.values

plt.plot(x, elo_y, 'b-', label="Improved Elo")
plt.plot(x, relo_y, 'c-', label="Elo")
plt.plot(x, glicko_y, 'r-', label="Glicko")
plt.plot(x, l5_y, 'k-', label="Log_5")

plt.xticks([2000,2003,2006,2009,2012,2015,2018])
plt.xlabel("Year")
plt.ylabel("Error")
plt.legend(loc='upper left')
plt.title('Error Comparison')

plt.show()







#end
