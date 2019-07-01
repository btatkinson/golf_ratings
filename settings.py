euro_init = {
    'elo':1455,
    'glicko':1455,
}

asg_set = {
    'alpha':.03,
    'max_window_len':200,
    'min_ewm':100,
    'init_len':15,
    'pga': -1.5,
    'euro':-1.65,
    'web':-2
}

ielo_set = {
    'init':1500,
    'K':0.225,
    'beta':400,
    'ACP':0.006,
    'C':3.55
}

glicko_set = {
    'init':1500,
    'phi':20.0,
    'sigma':0.0285,
    'tau':0.009,
    'epsilon':0.000001,
    'ratio':173.7178,
    # how fast does uncertainty increase?
    'u_i':1.30
}








# end
