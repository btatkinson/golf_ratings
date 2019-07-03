import numpy as np
import pandas as pd
import math
import scipy.stats
import ast


ratings = pd.read_csv('./data/current_player_ratings.csv')

def is_team(x):
    if ' / ' in x:
        to_drop = True
    else:
        to_drop = False
    return to_drop
# preprocess
ratings['to_drop'] = ratings.name.apply(lambda x: is_team(x))

print(len(ratings))
ratings = ratings.loc[ratings['to_drop']==False]
print(len(ratings))
i = 0

def find_var(x):
    try:
        x = [float(i) for i in x]
    except:
        x = ast.literal_eval(x)
        if len(x) <=0:
            return 9
    return float(np.var(x))

ratings['var'] = ratings.prev_sgs.apply(lambda x: find_var(x))

# ratings for high sample size
rhs = ratings.loc[ratings.rnds_played >= 100]
rls = ratings.loc[ratings.rnds_played < 100]
rls = rls.loc[rls.rnds_played >= 5]

print("high sample var = ", rhs['var'].mean())
print("low sample var = ", rls['var'].mean())
# end
