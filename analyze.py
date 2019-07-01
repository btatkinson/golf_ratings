import numpy as np
import pandas as pd
import sys
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns
import math
import gc
import cartopy.crs as ccrs
from tqdm import tqdm
import random

sys.path.insert(0, './classes')
from map import Map

######### World Map Projection #########
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.stock_img()
#
# tournaments = pd.read_csv('../golf_scraper/sched.csv')
# saved = pd.read_csv('../golf_scraper/has_saved.csv')
# saved_ids = saved.TID.values
#
# trns = tournaments.loc[tournaments['tid'].isin(saved_ids)]
# Map = Map()
#
# pga_lats = []
# pga_lngs = []
# euro_lats = []
# euro_lngs = []
# for index, row in trns.iterrows():
#     location = row['location']
#     trn_lat = Map.loc_dict[location]['Latitude']
#     trn_lng = Map.loc_dict[location]['Longitude']
#     tour = row['tour']
#     if row['tour'] == 'PGA':
#         pga_lats.append(trn_lat)
#         pga_lngs.append(trn_lng)
#     elif row['tour'] == 'Euro':
#         euro_lats.append(trn_lat)
#         euro_lngs.append(trn_lng)
# # lon, lat = [-75,77.23], [43,28.61]
#
# ax.scatter(euro_lngs, euro_lats,color='k', s=3, label='Euro')
# ax.scatter(pga_lngs, pga_lats,color='b',s=3, label='PGA')
#
# plt.legend(loc='upper left')
#
# plt.show()
#
# raise ValueError('Done Plotting')

######### Elo vs SG ###############

# Direction of travel
east = ['NE','E','SE']
west = ['NW','W','SW']
vert = ['N','S']

# are they on short week?
def osw(row):
    on_short_week = False
    if (row['Days_Since'] <= 8
    and row['PR4']==True
    # and row['Bearing_From_Last'] in east
    and row['Dist_From_Last'] > 1000
    and row['Round'] in ['R1']):
        on_short_week = True
    return on_short_week

df = pd.read_csv('./data/elo_vs_sg.csv')

print("Before dropping results from golfers with low rounds played", len(df))
# df = df.loc[df['Rnds_Played']>=40]
print(len(df))

df['Exp_SG'] = 0.01485372967616837*df['Elo_Gained']
df['Diff'] = df['Strokes_Gained']-df['Exp_SG']

# create mask for on short week
# df['OSW'] = df.apply(lambda row: osw(row),axis=1)
#
# osw = df.loc[df['OSW']==True]
# nosw = df.loc[df['OSW']==False]
#
# print(len(osw),osw.Diff.mean(),osw.Diff.std())
# print(len(nosw),nosw.Diff.mean(),nosw.Diff.std())

# plt.hist(df.Days_Since)
# plt.show()

df = df.dropna(subset=['Dist_From_Last'])
# Dropping more than a week long off
print("Before narrowing time",len(df))
# df = df.loc[df['Days_Since']<=8]
# df = df.loc[df['Days_Since']>=6]
print(len(df))
#
# # Did they play in round 4 of the previous tournament?
# print('Before PR4', len(df))
np4df = df.loc[df['PR4']==False]
df = df.loc[df['PR4']==True]

# print(len(df))
#
# print("Before direction",len(df))
# # df = df.loc[df['Bearing_From_Last'].isin(east)]
# print(len(df))
#
# # Drop non round 1 values
# print('Before Narrowing Rounds',len(df))
df = df.loc[df['Round'].isin(['R1'])]
# print(len(df))

# Drop close tournaments
# df = df.loc[df['Dist_From_Last']>=1000]
# plt.hist(df.Days_Since)
# plt.show()

# plt.show()

fig, ax = plt.subplots(figsize=(15,7))

np4x = np4df['Distance Gained'].values
np4y = np4df.Exp_SG.values

x = df['Distance Gained'].values
y = df.Exp_SG.values

# new_xs = []
# for _x in x:
#     new_x = math.log10(_x+1)
#     new_xs.append(new_x)
#
# x = new_xs
#
# print(np.percentile(x, 1))
# print(np.percentile(y, 95))

# # result in no skew and kurtosis 3
# epsilon = -0.031
# delta = 0.92
# #
# new_x = []
# for _x in x:
#     new_x.append(np.sinh(delta*np.arcsinh(_x)-epsilon)*delta*np.cosh(delta*np.arcsinh(_x)-epsilon)/np.sqrt(1+_x**2))
#
# epsilon = -0.056
# delta = 0.9362
# new_y = []
# for _y in y:
#     new_y.append(np.sinh(delta*np.arcsinh(_y)-epsilon)*delta*np.cosh(delta*np.arcsinh(_y)-epsilon)/np.sqrt(1+_y**2))
#
# y = new_y

# _x = stats.yeojohnson(x)
# _x = list(_x[0])
# x = _x

# _y = stats.yeojohnson(y)
# _y = list(_y[0])
# y = _y

# log past 12
# new_ys = []
# for _y in y:
#     if _y < -9.5:
#         new_y = - 9.5 - 3.25*math.log10((-1*_y)-9.5)
#     else:
#         new_y = _y
#     new_ys.append(new_y)
#
# y = new_ys

# print("Normal Test")
# print(stats.normaltest(x))
# print(stats.normaltest(y))
#
# print("Kurtosis")
# print(stats.kurtosis(x))
# print(stats.kurtosis(y))
#
# print("Skew")
# print(stats.skew(x))
# print(stats.skew(y))
#
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
print("slope: ", slope)
print("y int", intercept)
print("r value", r_value)
print("p value", p_value)
print("std error", std_err)

line_y = []
for _x in x:
    line_y.append(slope * _x + intercept)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np4x,np4y)
print("slope: ", slope)
print("y int", intercept)
print("r value", r_value)
print("p value", p_value)
print("std error", std_err)

line_np4y = []
for _x in np4x:
    line_np4y.append(slope * _x + intercept)
plt.scatter(np4x,np4y,s=3)
plt.plot(np4x, line_np4y, label="Did Not Play R4")
plt.scatter(x,y,s=3)
plt.plot(x, line_y, '-r', label="Did Play")


# stats.probplot(y, plot=plt)
plt.xlabel("Distance From Last Tournament (Miles)")
plt.ylabel("Strokes Gained Difference")
plt.legend(loc='upper left')
# plt.title('Q-Q Plot of Strokes Gained')

plt.show()

######## Correlation of ranking systems ##################

# rnks = pd.read_csv('./data/rating_comp.csv')
#
# sans_plyr = rnks.drop(columns='Player')
#
# corr = sans_plyr.corr()
# sns.heatmap(corr, cbar=True, annot=True, cmap='inferno', square=True);
# # plt.imshow(corr, cmap='viridis', interpolation='nearest')
# plt.show()

#################### Season by Season Error #######################

# yby = pd.read_csv('./data/sea_by_sea.csv')

# pga_yby = yby.loc[yby['Tour']=='PGA']
# euro_yby = yby.loc[yby['Tour']=='Euro']
#
# pga_x = pga_yby.Season.unique()
# euro_x = euro_yby.Season.unique()
#
# pga_elo_y = pga_yby.loc[pga_yby['System']=='elo'].Error.values
# pga_glicko_y = pga_yby.loc[pga_yby['System']=='glicko'].Error.values
# pga_l5_y = pga_yby.loc[pga_yby['System']=='l5'].Error.values
#
# euro_elo_y = euro_yby.loc[euro_yby['System']=='elo'].Error.values
# euro_glicko_y = euro_yby.loc[euro_yby['System']=='glicko'].Error.values
# euro_l5_y = euro_yby.loc[euro_yby['System']=='l5'].Error.values
#
# fig, ax = plt.subplots(1, 2)
#
# plt.subplot(1,2,1)
# plt.plot(pga_x, pga_elo_y, 'b-', label="Improved Elo")
# plt.plot(pga_x, pga_glicko_y, 'r-', label="Glicko")
# plt.plot(pga_x, pga_l5_y, 'k-', label="Log_5")
#
# plt.xticks([2000,2003,2006,2009,2012,2015,2018])
# plt.xlabel("Year")
# plt.ylabel("Error")
# plt.legend(loc='upper left')
# plt.title('PGA Errors')
#
# plt.subplot(1,2,2)
# plt.plot(euro_x, euro_elo_y, 'b-', label="Improved Elo")
# plt.plot(euro_x, euro_glicko_y, 'r-', label="Glicko")
# plt.plot(euro_x, euro_l5_y, 'k-', label="Log_5")
#
# plt.xlabel("Year")
# plt.ylabel("Error")
#
# plt.xticks([2000,2003,2006,2009,2012,2015,2018])
# # xint = range(0, math.ceil(17)+1)
# # plt.xticks(xint)
# # ax.set_xscale('log')
# plt.legend(loc='upper left')
# plt.title('Euro Errors')
#
# plt.show()
############# Round by Round error #########################

# l5_rbr = pd.read_csv('./data/rbr/l5_brp.csv')
# elo_rbr = pd.read_csv('./data/rbr/elo_brp.csv')
# glicko_rbr = pd.read_csv('./data/rbr/glicko_brp.csv')

# elo_rbr = elo_rbr.loc[elo_rbr['Rnds_Played'] >= 25]
# glicko_rbr = glicko_rbr.loc[glicko_rbr['Rnds_Played'] >= 25]
# elo_rbr = elo_rbr.loc[elo_rbr['Rnds_Played'] <= 300]
# glicko_rbr = glicko_rbr.loc[glicko_rbr['Rnds_Played'] <= 300]

#
# l5_x = l5_rbr.Rnds_Played.values
# l5_y = l5_rbr.Error.values
#
# elo_x = elo_rbr.Rnds_Played.values
# elo_y = elo_rbr.Error.values
# #
# glicko_x = glicko_rbr.Rnds_Played.values
# glicko_y = glicko_rbr.Error.values
#
# # l5_x = [math.log(x+1,10) for x in l5_x]
# # elo_x = [math.log(x+1,10) for x in elo_x]
# # glicko_x = [math.log(x+1,10) for x in glicko_x]


# b1, m1 = polyfit(l5_x, l5_y, 1)
# b2, m2 = polyfit(elo_x, elo_y, 1)
# b3, m3 = polyfit(glicko_x, glicko_y, 1)

# print('Elo Slope: ', m2*100000)
# print('Glicko Slope: ', m3*100000)
# #
# fig, ax = plt.subplots(figsize=(15,7))
#
# plt.scatter(l5_x, l5_y, color="black", label=None)
# plt.scatter(elo_x, elo_y, color="blue", label=None)
# plt.scatter(glicko_x, glicko_y, color="red", label=None)
#
# l5_line_y = []
# for x in l5_x:
#     l5_line_y.append(x * m1 + b1)
# elo_line_y = []
# for x in elo_x:
#     elo_line_y.append(x * m2 + b2)
# glicko_line_y = []
# for x in glicko_x:
#     glicko_line_y.append(x * m3 + b3)
# #
# # plt.plot(l5_x, l5_line_y, 'k-', label="Log5")
# plt.plot(elo_x, elo_line_y, 'b-', label="Elo")
# plt.plot(glicko_x, glicko_line_y, 'r-', label="Glicko")
#
# plt.xlabel("Rounds Played")
# plt.ylabel("Error")
# plt.legend(loc='upper left')
# plt.title('Error by Rounds Played')
#
# plt.show()


# print(A,K)

# exp_decay = []
# for x in l5_x:
#     # exp_decay.append(x * K + A_log)
#     new_y = A * np.exp(K * x) - C0
#     exp_decay.append(new_y)
# exp_decay[1] = exp_decay[2]
# exp_decay[0] = exp_decay[1]
# plt.plot(l5_x, exp_decay, 'k-')

# def fit_exp_linear(t, y, C=0):
#     y = y - C
#     y = np.log(y)
#     K, A_log = np.polyfit(t, y, 1)
#     A = np.exp(A_log)
#     return A, K
#
# def model_func(t, A, K, C):
#     return A * np.exp(K * t) + C


# A, K = fit_exp_linear(l5_x, l5_y, C0)
# fit_y = [model_func(y,A,K,C0) for y in l5_y]
# plt.plot(l5_x, fit_y, 'b-')

# print("Slope of Log5 Error", m1)
# print("Slope of Elo Error", m2)
# print("Slope of Glicko Error", m3)
# print("Y-Int of Log5 Error", b1)
# print("Y-Int of Elo Error", b2)
# print("Y-Int of Glicko Error", b3)

# plt.xlabel("Rounds Played")
# plt.ylabel("Error")
#
# plt.xticks([0,1,2,3], ["1","10","100","1000"])
# xint = range(0, math.ceil(17)+1)
# plt.xticks(xint)
# ax.set_xscale('log')
# plt.legend(loc='upper left')
#
# plt.show()









# end
