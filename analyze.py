import numpy as np
import pandas as pd
import sys
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns
import math
import gc
from tqdm import tqdm
import random

# https://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def model_func(t, A, K, C):
    return A * np.exp(K * t) + C

gc.collect()

sched = pd.read_csv('../golf_scraper/sched.csv')
seasons = sched.season.unique()
seasons = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]


# finding adjustment based on time off

all_sea = pd.DataFrame()
for season in seasons:
    print("loading " + str(season))
    path = './data/seasons/'+str(season)+'.csv'
    sea_df = pd.read_csv(path)
    all_sea = pd.concat([all_sea,sea_df],sort=False)

print(len(all_sea))
print("dropping weird days since")
all_sea = all_sea.loc[all_sea['DS']>=7]
print(len(all_sea))

# print("dropping players that play once or twice")
# all_sea = all_sea.loc[all_sea['RndsPlayed']>=10]
# print(len(all_sea))

def apply_epoly(x):
    return -0.001012 * x**2 + 0.4611 * x - 46.4

def apply_gpoly(x):
    return 5.305e-05 * math.pow(x,3) - 0.02604 * x**2 + 4.391*x - 251.2

# convert glicko and elo to predicted SG
all_sea['Elo10'] = all_sea['Elo']/10
all_sea['EloSG'] = all_sea.Elo10.apply(lambda x: apply_epoly(x))
all_sea = all_sea.drop(columns=['Elo10'])
#
all_sea['Glicko10'] = all_sea['Glicko']/10
all_sea['GlickoSG'] = all_sea.Glicko10.apply(lambda x: apply_gpoly(x))

def apply_gdo(x,ds):
    return x + (0.34837 * np.exp(-0.20651 * (ds/7)) - 0.25)

all_sea['GlickoSG'] = all_sea.apply(lambda row: apply_gdo(row['GlickoSG'],row['DS']), axis=1)
all_sea = all_sea.drop(columns=['Glicko10'])

# x = all_sea.Glicko.values
# y = all_sea.SG.values
# print("Glicko Vs SG")
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
# print("slope: ", slope)
# print("y int", intercept)
# print("r value", r_value)
# print("p value", p_value)
# print("std error", std_err)
# raise ValueError

# acc_check = all_sea[['Elo','SG','EloSG','GlickoSG','ASG']]
# print(all_sea.SG.mean())
# print(all_sea.EloSG.mean())
# print(all_sea.GlickoSG.mean())
# from sklearn.metrics import mean_squared_error
# print("ASG Error",mean_squared_error(acc_check.SG.values, acc_check.ASG.values))
# print("Elo Error",mean_squared_error(acc_check.SG.values, acc_check.EloSG.values))
# print("Glicko Error",mean_squared_error(acc_check.SG.values, acc_check.GlickoSG.values))
# raise ValueError

all_sea['GDiff'] = all_sea['SG'] - all_sea['GlickoSG']

# tens of rounds played
all_sea['TRP'] = all_sea['RndsPlayed']/10
all_sea['TRP'] = all_sea['TRP'].astype(float)
all_sea['TRP'] = all_sea['TRP'].round(0)
all_sea['TRP'] = all_sea['TRP'].astype(int)

# find min threshold (35 rounds in 100 player fields)
min_t = 100*25

gb = all_sea.groupby(['TRP']).count()
gb = gb.reset_index()
gb = gb.loc[gb['GDiff']>=min_t]
allowed_rps = gb.TRP.unique()
# print('ALLOWED RP',allowed_rps)

gb = all_sea.groupby(['TRP']).mean()
gb = gb.reset_index()
gb = gb.loc[gb['TRP'].isin(allowed_rps)]
x = gb.TRP.values
y = gb.GDiff.values

# C0=-0.25
# A, K = fit_exp_linear(x, y, C0)
# print("A",A,"K",K)
# fit_y = model_func(x, A, K, C0)

fig, ax = plt.subplots(figsize=(15,7))
plt.title('Effect of RP')
plt.xlabel("RP")
plt.ylabel("Actual vs. Expected SG")
plt.plot(x,y)
# plt.plot(x,fit_y)
plt.show()

######### Glicko Weeks Since Last ####################
# all_sea['GDiff'] = all_sea['SG'] - all_sea['GlickoSG']
# #
# all_sea['WS'] = all_sea['DS']/7
# all_sea['WS'] = all_sea['WS'].astype(float)
# all_sea['WS'] = all_sea['WS'].round(0)
# all_sea['WS'] = all_sea['WS'].astype(int)
# print(all_sea.head())
#
# # find min threshold (35 rounds in 100 player fields)
# min_t = 100*10
# #
# gb = all_sea.groupby(['WS']).count()
# gb = gb.reset_index()
# gb = gb.loc[gb['GDiff']>=min_t]
# allowed_wks = gb.WS.unique()
# # print('ALLOWED WEEKS',allowed_wks)
# #
# gb = all_sea.groupby(['WS']).mean()
# gb = gb.reset_index()
# gb = gb.loc[gb['WS']<=12]
# x = gb.WS.values
# y = gb.GDiff.values

# C0=-0.25
# A, K = fit_exp_linear(x, y, C0)
# print("A",A,"K",K)
# fit_y = model_func(x, A, K, C0)

# fig, ax = plt.subplots(figsize=(15,7))
# plt.title('Effect of Layoff Time')
# plt.xlabel("Weeks Since Last Tournament")
# plt.ylabel("Actual vs. Expected SG")
# plt.plot(x,y)
# # plt.plot(x,fit_y)
# plt.show()
raise ValueError

######### Adj Elo and Glicko ######

# create buckets of elo ratings
all_sea['EloB'] = all_sea['Elo']/10
all_sea['EloB'] = all_sea['EloB'].astype(float)
all_sea['EloB'] = all_sea['EloB'].round(0)
all_sea['EloB'] = all_sea['EloB'].astype(int)

gb = all_sea.groupby(['EloB']).count()
print(gb)
gb = gb.reset_index()
gb = gb.loc[gb['Elo']>=75]
allowed_rps = gb.EloB.unique()
print('ALLOWED RP',allowed_rps)
#
gb = all_sea.groupby(['EloB']).mean()
gb = gb.reset_index()
gb = gb.loc[gb['EloB'].isin(allowed_rps)]
print(gb)
x = gb.EloB.values
y = gb.SG.values

# x = all_sea.Elo.values
# y = all_sea.SG.values

# def apply_poly(x):
#     return -9.768e-07 * math.pow(x,4) + 0.0006565 * math.pow(x,3) - 0.1654 * x**2 + 18.64*x - 795.5
#
# all_sea['Elo10'] = all_sea['Elo']/10
# all_sea['EloSG'] = all_sea.Elo10.apply(lambda x: apply_poly(x))
# all_sea = all_sea.drop(columns=['Elo10'])
# poly_pred = all_sea.EloSG.values

# print("Elo Vs SG")
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
# print("slope: ", slope)
# print("y int", intercept)
# print("r value", r_value)
# print("p value", p_value)
# print("std error", std_err)
# line_y = []
# for _x in x:
#     line_y.append(slope * _x + intercept)

poly2 = np.poly1d(np.polyfit(x,y,2))
# poly3 = np.poly1d(np.polyfit(x,y,3))
# poly4 = np.poly1d(np.polyfit(x,y,4))

# fig, ax = plt.subplots(figsize=(15,7))

# plt.title('Effect of Layoff Time')
plt.xlabel("Elo Buckets")
plt.ylabel("SG")
plt.scatter(x,y)
# plt.plot(x, line_y, '-r')
plt.plot(x, poly2(x),'-k')
plt.show()

print(poly2)
raise ValueError

######################################

# create buckets of glicko ratings
all_sea['GlickoB'] = all_sea['Glicko']/10
all_sea['GlickoB'] = all_sea['GlickoB'].astype(float)
all_sea['GlickoB'] = all_sea['GlickoB'].round(0)
all_sea['GlickoB'] = all_sea['GlickoB'].astype(int)
#
gb = all_sea.groupby(['GlickoB']).count()
print(gb)
gb = gb.reset_index()
gb = gb.loc[gb['Glicko']>=75]
allowed_rps = gb.GlickoB.unique()
print('ALLOWED RP',allowed_rps)
# #
gb = all_sea.groupby(['GlickoB']).mean()
gb = gb.reset_index()
gb = gb.loc[gb['GlickoB'].isin(allowed_rps)]
print(gb)
x = gb.GlickoB.values
y = gb.SG.values


def apply_poly(x):
    return -3.235e-06 * math.pow(x,4) + 0.002052 * math.pow(x,3) - 0.4879 * x**2 + 51.66*x - 2060
#
all_sea['Glicko10'] = all_sea['Glicko']/10
all_sea['GlickoSG'] = all_sea.Glicko10.apply(lambda x: apply_poly(x))
all_sea = all_sea.drop(columns=['Glicko10'])

print("Glicko Vs SG")
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
print("slope: ", slope)
print("y int", intercept)
print("r value", r_value)
print("p value", p_value)
print("std error", std_err)
line_y = []
for _x in x:
    line_y.append(slope * _x + intercept)

poly = np.poly1d(np.polyfit(x,y,2))
poly3 = np.poly1d(np.polyfit(x,y,3))
poly4 = np.poly1d(np.polyfit(x,y,4))

# fig, ax = plt.subplots(figsize=(15,7))
plt.title('Effect of Layoff Time')
plt.xlabel("Glicko Buckets")
plt.ylabel("SG")
plt.scatter(x,y)
plt.plot(x, line_y, '-r')
plt.plot(x, poly3(x),'-y')
plt.plot(x, poly4(x),'-k')
plt.show()

print(poly3)

######### Adjust SG based on rounds played ###############

# tens of rounds played
# all_sea['TRP'] = all_sea['RndsPlayed']/10
# all_sea['TRP'] = all_sea['TRP'].astype(float)
# all_sea['TRP'] = all_sea['TRP'].round(0)
# all_sea['TRP'] = all_sea['TRP'].astype(int)

# find min threshold (35 rounds in 100 player fields)
# min_t = 100*35
#
# gb = all_sea.groupby(['TRP']).count()
# print(gb)
# gb = gb.reset_index()
# gb = gb.loc[gb['Diff']>=min_t]
# allowed_rps = gb.TRP.unique()
# print('ALLOWED RP',allowed_rps)
#
# gb = all_sea.groupby(['TRP']).mean()
# gb = gb.reset_index()
# gb = gb.loc[gb['TRP'].isin(allowed_rps)]
# t = gb.TRP.values
# y = gb.Diff.values
#


#
# C0=-0.14
# A, K = fit_exp_linear(t, y, C0)
# print("A",A,"K",K)
# fit_y = model_func(t, A, K, C0)

# y = [math.log10(_x + 1) for _x in x]

# fig, ax = plt.subplots(figsize=(15,7))
#
# plt.title('Effect of Rounds Played')
# plt.xlabel("Rounds Played")
# plt.ylabel("Actual vs. Expected SG")
# #
# plt.plot(t,y)
# plt.plot(t,fit_y)
# plt.show()

###########  Adj based on layoff ##################

# all_sea['WS'] = all_sea['DS']/7
# all_sea['WS'] = all_sea['WS'].astype(float)
# all_sea['WS'] = all_sea['WS'].round(0)
# all_sea['WS'] = all_sea['WS'].astype(int)
# print(all_sea.head())

# find min threshold (35 rounds in 100 player fields)
# min_t = 100*30
#
# gb = all_sea.groupby(['WS']).count()
# gb = gb.reset_index()
# gb = gb.loc[gb['Diff']>=min_t]
# allowed_wks = gb.WS.unique()
# print('ALLOWED WEEKS',allowed_wks)
#
# gb = all_sea.groupby(['WS']).mean()
# gb = gb.reset_index()
# gb = gb.loc[gb['WS']<=12]
# x = gb.WS.values
# y = gb.Diff.values
#
# C0=-0.124
# A, K = fit_exp_linear(x, y, C0)
# print("A",A,"K",K)
# fit_y = model_func(x, A, K, C0)
#
# fig, ax = plt.subplots(figsize=(15,7))
#
# plt.title('Effect of Layoff Time')
# plt.xlabel("Weeks Since Last Tournament")
# plt.ylabel("Actual vs. Expected SG")
#
# plt.plot(x,y)
# plt.plot(x,fit_y)
# plt.show()

## show year by year data ##

# yby = pd.read_csv('./data/yby_data.csv')
#
# fig, ax = plt.subplots(figsize=(15,7))
#
# elo_y = yby.Elo.values
# relo_y = yby.rElo.values
# glicko_y = yby.Glicko.values
# l5_y = yby.L5.values
#
# x = yby.Season.values
#
# plt.plot(x, elo_y, 'b-', label="Improved Elo")
# plt.plot(x, relo_y, 'c-', label="Elo")
# plt.plot(x, glicko_y, 'r-', label="Glicko")
# plt.plot(x, l5_y, 'k-', label="Log_5")
#
# plt.xticks([2000,2003,2006,2009,2012,2015,2018])
# plt.xlabel("Year")
# plt.ylabel("Error")
# plt.legend(loc='upper left')
# plt.title('Error Comparison')
#
# plt.show()







#end
