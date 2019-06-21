import pandas as pd
import numpy as np
import sys

from helpers import *
from settings import *

sys.path.insert(0, './classes')
from player import Player
from glicko import Glicko
from elo import Elo

MU = glicko_set['init']
ratio = glicko_set['ratio']

travs = pd.read_csv('./data/travelersfield.csv')
pr = pd.read_csv('./data/current_player_ratings.csv')

field = list(travs.Name.unique())
field.append('Bryson DeChambeau')
# field player ratings
fpr = pr.loc[pr['name'].isin(field)]

print(fpr)

raise ValueError()

def get_expected(pairs):
    global Elo
    global Glicko

    p1_name = pairs[0]
    p2_name = pairs[1]

    if p2_name=='Charlie Hoffman':
        p2_name = 'Charley Hoffman'

    p1_row = fpr.loc[pr['name']==p1_name]
    p2_row = fpr.loc[pr['name']==p2_name]

    p1_elo = p1_row['ielo'].values[0]
    p2_elo = p2_row['ielo'].values[0]

    elo_expected = Elo.x(p1_elo,p2_elo)

    impact = Glicko.reduce_impact(p2_row['gvar'].values[0])
    mu = (p1_row['glicko'].values[0] - MU)/ratio
    opp_mu = (p2_row['glicko'].values[0] - MU)/ratio
    expected_result = Glicko.get_expected(mu, opp_mu, impact)

    row = [p1_name, p2_name, elo_expected, expected_result]
    return row

pairs = [
['Ryan Moore','Byeong Hun An'],
['C.T. Pan','Brian Harman'],
['Brandt Snedeker','Patrick Reed'],
['Francesco Molinari','Justin Thomas'],
['Brooks Koepka','Patrick Cantlay'],
['Tony Finau','Bubba Watson'],
['Viktor Hovland','Charlie Hoffman'],
['Viktor Hovland','Daniel Berger'],
['Russell Knox','Phil Mickelson'],
['Bryson DeChambeau','Tommy Fleetwood'],
['Daniel Berger','Adam Hadwin'],
['Kevin Kisner','Keegan Bradley'],
['Jason Day','Paul Casey'],
['Jordan Spieth','Marc Leishman'],
['Kevin Streelman','Louis Oosthuizen'],
['Bubba Watson', 'Brooks Koepka'],
['Matthew Wolff','Kevin Tway']
]

Elo = Elo()
Glicko = Glicko()

all_expected = []
for pair in pairs:
    expected = get_expected(pair)
    all_expected.append(expected)

mdf = pd.DataFrame(all_expected, columns=['Player 1', 'Player 2', 'Elo X', 'Glicko X'])

mdf['Model'] = 0.35*mdf['Elo X'] + 0.65*mdf['Glicko X']
mdf['Model'] = mdf['Model'].round(4)


def pct_to_odds(x):
    x *= 100
    if x >= 50:
        y = 0 - (x/(100 - x)) * 100
    else:
        y = ((100-x)/x) * 100
        y = '+' + str(np.round(y,3))
        # 53 / 47
    return y
mdf['Line P1'] = mdf['Model'].apply(lambda x: pct_to_odds(x))
mdf['Line P2'] = mdf['Model'].apply(lambda x: pct_to_odds(1-x))

print(mdf)
# print(mdf[['Player 1', 'Player 2', 'Model', 'Line P1', 'Line P2']])
# end
