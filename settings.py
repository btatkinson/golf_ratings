import numpy as np

euro_init = {
    'elo':1455,
    'glicko':1397,
}

asg_set = {
    'alpha':.023, #0.25
    'max_window_len':200,
    'min_ewm':100,
    'init_len':15,
    'pvar_sms':9.043475601688755,
    'pvar_lgs':8.347564763611148,
    'pct_sms':1,
    'pct_lgs':0.65,
    'init_pga':np.array([]),
    'init_euro':np.array([]),
}

# basic elo
relo_set = {
    'K':0.23 #0.25 - 0.29137
}

ielo_set = {
    'init':1500,
    'K':0.2215, #0.221 & 0.2225 - 29.137
    'beta':400,
    'ACP':0.006,
    'C':3
}

glicko_set = {
    'init':1442,
    'phi':8.9, # 9 - 29.305, 8.75-29.306, < 8.75 is worse
    'sigma':0.002,
    'tau':0.2,
    'epsilon':0.000001,
    'ratio':173.7178,
    # how fast does uncertainty increase?
    'u_i':1.5
}








# end
