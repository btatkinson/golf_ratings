import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import seaborn as sns
import math
import gc

######## Correlation of ranking systems ##################

rnks = pd.read_csv('./data/rating_comp.csv')

sans_plyr = rnks.drop(columns='Player')

corr = sans_plyr.corr()
sns.heatmap(corr, cbar=True, annot=True, cmap='inferno', square=True);
# plt.imshow(corr, cmap='viridis', interpolation='nearest')
plt.show()

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

# l5_rbr = pd.read_csv('./data/rbr/log_5.csv')
# elo_rbr = pd.read_csv('./data/rbr/elo.csv')
# glicko_rbr = pd.read_csv('./data/rbr/glicko.csv')
#
# l5_x = l5_rbr.Rnds_Played.values
# l5_y = l5_rbr.Error.values
#
# elo_x = elo_rbr.Rnds_Played.values
# elo_y = elo_rbr.Error.values
#
# glicko_x = glicko_rbr.Rnds_Played.values
# glicko_y = glicko_rbr.Error.values
#
# # l5_x = [math.log(x+1,10) for x in l5_x]
# # elo_x = [math.log(x+1,10) for x in elo_x]
# # glicko_x = [math.log(x+1,10) for x in glicko_x]
#
# b1, m1 = polyfit(l5_x, l5_y, 1)
# b2, m2 = polyfit(elo_x, elo_y, 1)
# b3, m3 = polyfit(glicko_x, glicko_y, 1)
#
# fig, ax = plt.subplots(figsize=(15,7))
#
# plt.scatter(l5_x, l5_y, color="black", label='Log5')
# plt.scatter(elo_x, elo_y, color="blue", label=None)
# plt.scatter(glicko_x, glicko_y, color="red", label=None)
#
# elo_line_y = []
# for x in elo_x:
#     elo_line_y.append(x * m2 + b2)
# glicko_line_y = []
# for x in glicko_x:
#     glicko_line_y.append(x * m3 + b3)
#
# # plt.plot(l5_x, l5_line_y, '-', label="Log5")
# plt.plot(elo_x, elo_line_y, 'b-', label="Elo")
# plt.plot(glicko_x, glicko_line_y, 'r-', label="Glicko")

# C0 = -.43
# l5_y_log = l5_y - C0
# l5_y_log = [math.log(y) for y in l5_y_log]
# # l5_y_log = [math.log(y) for y in l5_y_log]
# K, A_log = np.polyfit(l5_x, l5_y_log, 1)
# A = np.exp(A_log)
# print(A,K)
# A = 5
# K = -3
# plt.scatter(l5_x, l5_y_log, label=None)


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
