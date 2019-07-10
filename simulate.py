import pandas as pd
import numpy as np
import math
import sys
from helpers import *
from itertools import combinations
from tqdm import tqdm

sys.path.insert(0, './classes')
from player import Player
from glicko import Glicko
from elo import Elo
from map import Map

# initialize ratings objects
Glicko = Glicko()
Elo = Elo()

# import field
field_name = 'ScottishOpen'
start_date = 'Jul 11 2019'
tour = 'Euro'
CUT_RULE = 60
field = pd.read_csv('./data/fields/'+field_name+'.csv')
golfers = list(field.Name.values)

# import ratings
pdf = pd.read_csv('./data/current_player_ratings.csv')
rated_players = (pdf.name.values)

def adjust_names(golfer):
    adj_dict = {
    'Hao-Tong Li':'Haotong Li',
    'Mike Lorenzo-vera':'Michael Lorenzo-vera',
    'Jordan L. Smith':'Jordan Smith',
    'Billy Hurley':'Billy Hurley III',
    'Fredrik Jacobson':'Freddie Jacobson',
    'Mike Lorenzo-Vera':'Michael Lorenzo-vera',
    'Sang-hyun Park':'Sanghyun Park',
    'Alexander Bjork':'Alexander Björk'
    }

    if golfer in adj_dict:
        golfer = adj_dict[golfer]
    return golfer

golfers = [adjust_names(x) for x in golfers]

# clean names
def find_unusual(names):
    print("Golfers not found",list(set(names)-set(rated_players)))
    return

print("\n")
find_unusual(golfers)
print("\n")
print("Top 10", pdf[:9])

pdf = pdf.set_index('name')
pdf = pdf.to_dict('index')


plist = []

for player in golfers:
    if player in pdf:
        dict = pdf[player]
        PObj = Player(
            name=player,
            elo=dict['ielo'],
            rnds_played=dict['rnds_played'],
            glicko = dict['glicko'],
            gvar = dict['gvar'],
            gsig = dict['gsig'],
            ldate = dict['last_date'],
            cdate = start_date,
            prev_sgs = dict['prev_sgs'],
            pvar = dict['pvar'],
        )
    else:
        if tour == 'PGA':
            elo = ielo_set['init']
            glicko = glicko_set['init']
            # prev_sgs = asg_set['init_pga']
        elif tour == 'Euro':
            elo = euro_init['elo']
            glicko = euro_init['glicko']
            # prev_sgs = asg_set['init_euro']
        PObj = Player(
            name=player,
            tour=tour,
            elo=elo,
            # relo=elo,
            glicko=glicko,
            ldate=start_date,
            cdate=start_date,
            # prev_sgs=prev_sgs
        )
    plist.append(PObj)

def get_field_str(plist):
    field_str = []
    for p in plist:
        field_str.append(0.9*p.esg+0.1*p.gsg)
    return np.mean(field_str)

def sim_round(rnd_num,plist):
    field_str = get_field_str(plist)
    round_lb = []
    for p in plist:
        p.rnds_played += 1
        p.gsg = get_gsg(p.glicko,p.days_since,p.rnds_played)
        p.esg = get_esg(p.elo,p.days_since,p.rnds_played)
        exp_sg = (0.9 * p.esg + 0.1 * p.gsg) - field_str
        p.calc_var()
        round_score = int(np.round(71 - np.random.normal(exp_sg,math.sqrt(p.pvar)),0))
        setattr(p, rnd_num, round_score)
        round_lb.append([p.name,getattr(p,rnd_num)])
    return pd.DataFrame(round_lb,columns=['Name',rnd_num])

def update_elo(plist, round):
    combos = [c for c in combinations(plist,2)]
    num_opps = len(plist) - 1
    change_dict = {}

    for p in plist:
        change_dict[p.name] = 0

    for combo in combos:
        p1 = combo[0]
        p2 = combo[1]
        p1_score = getattr(p1, round)
        p2_score = getattr(p2, round)
        margin = abs(p2_score - p1_score)

        elo_x = Elo.x(p1.elo, p2.elo)
        glicko_x = Glicko.x(p1, p2)
        if p1_score <= p2_score:
            p1_change, p2_change = Elo.get_ielo_delta(elo_x, margin, p1, p2, num_opps)
            if p1_score == p2_score:
                result = 0.5
            else:
                result = 1
        else:
            p1_change, p2_change = Elo.get_ielo_delta(elo_x, (-1*margin), p1, p2, num_opps)
            result = 0

        change_dict[p1.name] += p1_change
        change_dict[p2.name] += p2_change

    for p in plist:
        p.elo += change_dict[p.name]
    return plist

def update_glicko(plist,round):
    new_pobjs = []
    for pobj in plist:
        # add all opponents and their round scores for glicko calc
        opps = [[p, getattr(p, round)] for p in plist if p != pobj]

        results = []
        player_round_score = getattr(pobj, round)
        # append result vs opponent
        for opponent, opp_score in opps:
            if opp_score == player_round_score:
                result = 0.5
            elif opp_score < player_round_score:
                result = 0
            else:
                result = 1
            results.append([opponent, result])
        # Glicko class edits glicko rating of player
        new_pobj = Glicko.update(pobj, results)
        new_pobjs.append(new_pobj)
    return new_pobjs

def update_ratings(plist,rlb,round):

    # dict with round scores
    # rlb = rlb.set_index('Name')
    # rdict = rlb.to_dict('index')
    # update elo
    plist = update_elo(plist,round)

    plist = update_glicko(plist,round)

    return plist

def make_cut(tlb):
    # total after round 2
    tlb['TAR2'] = tlb['R1'] + tlb['R2']
    tlb = tlb.sort_values(by='TAR2',ascending=True)
    tlb = tlb.reset_index(drop=True)
    # add rank
    prev_score = 0
    prev_pos = 0
    i = 0
    positions = []
    for index, row in tlb.iterrows():
        i += 1
        score = row['TAR2']
        if score > prev_score:
            pos = i
            prev_score = score
            prev_pos = i
        else:
            pos = prev_pos
        positions.append(pos)
    tlb['Pos'] = pd.Series(positions,name="Pos")

    wknd_lb = tlb.loc[tlb['Pos']<=CUT_RULE]
    wknders = list(wknd_lb.Name.values)
    made_cut = [x for x in plist if x.name in wknders]
    mc = tlb.loc[tlb['Pos']>CUT_RULE]
    mc = mc.set_index('Name')
    mc = mc.to_dict('index')
    return wknd_lb,made_cut, mc

def sim_tournament(plist):

    tlb = sim_round('R1',plist)
    plist = update_ratings(plist, tlb, 'R1')

    round_lb = sim_round('R2',plist)
    plist = update_ratings(plist, round_lb, 'R2')
    tlb = pd.merge(left=tlb,right=round_lb,how='left',on=['Name','Name'])

    tlb,plist,mc = make_cut(tlb)

    round_lb = sim_round('R3',plist)
    plist = update_ratings(plist, round_lb, 'R3')
    tlb = pd.merge(left=tlb,right=round_lb,how='left',on=['Name','Name'])

    round_lb = sim_round('R4',plist)
    plist = update_ratings(plist, round_lb, 'R4')
    tlb = pd.merge(left=tlb,right=round_lb,how='left',on=['Name','Name'])

    tlb = tlb.fillna(0)
    tlb['Final'] = tlb['R1'] + tlb['R2'] + tlb['R3'] + tlb['R4']
    tlb = tlb.set_index('Name')
    tlb = tlb.to_dict('index')
    return tlb,mc

matchups = [
['Alexander Björk','Lucas Bjerregaard',-135,105],
['Graeme McDowell','Joost Luiten',-120,-110],
['Rory McIlroy','Matt Kuchar',-205,165],
['Andrew Putnam','Branden Grace',-130,100],
['Henrik Stenson','Justin Thomas',100,-130],
['Matt Wallace','Eddie Pepperell',-115,-115],
['Christiaan Bezuidenhout','Victor Dubuisson',-110,-120],
['Matt Wallace','Rafa Cabrera Bello',-110,-120],
['Erik Van Rooyen','Ian Poulter',100,-130],
['Rafa Cabrera Bello','Eddie Pepperell',-135,105],
['Bernd Wiesberger','Haotong Li', 105,-125],
['Jorge Campillo','Jordan Smith',-130,100],
['Henrik Stenson','Matthew Fitzpatrick',-140,110],
['Rory McIlroy','Rickie Fowler',-195,165],
['Robert Rock','Andrea Pavan',-115,-115],
['Gavin Green','Robert Macintyre',-120,-110],
]

# matchup dataframe
mdf = pd.DataFrame(matchups,columns=['P1','P2','P1 Line','P2 Line'])
mdf['P1 Wins'] = 0
mdf['P2 Wins'] = 0
mdf['Ties'] = 0

num_sims = 10000
for i in tqdm(range(num_sims)):
    tlb,mc = sim_tournament(plist)
    new_mdf = []
    for index,row in mdf.iterrows():
        p1 = row['P1']
        p2 = row['P2']
        prev_p1w = row['P1 Wins']
        prev_p2w = row['P2 Wins']
        prev_ties = row['Ties']
        if (p1 in tlb) and (p2 in tlb):
            p1_score = tlb[p1]['Final']
            p2_score = tlb[p2]['Final']
            if p2_score > p1_score:
                prev_p1w +=1
            elif p1_score > p2_score:
                prev_p2w +=1
            else:
                prev_ties +=1
        elif (p1 in tlb.keys()) and (p2 not in tlb.keys()):
            prev_p1w += 1
        elif (p1 not in tlb.keys()) and (p2 in tlb.keys()):
            prev_p2w += 1
        else:
            p1_score = mc[p1]['TAR2']
            p2_score = mc[p2]['TAR2']
            if p2_score > p1_score:
                prev_p1w +=1
            elif p1_score > p2_score:
                prev_p2w +=1
            else:
                prev_ties +=1
        new_mdf.append([p1,p2,row['P1 Line'],row['P2 Line'],prev_p1w,prev_p2w,prev_ties])
    mdf = pd.DataFrame(new_mdf,columns=['P1','P2','P1 Line','P2 Line','P1 Wins','P2 Wins','Ties'])


p1_returns = []
p2_returns = []
for index, row in mdf.iterrows():
    p1_line = row['P1 Line']
    p2_line = row['P2 Line']
    p1_wins = row['P1 Wins']
    p2_wins = row['P2 Wins']
    ties = row['Ties']
    p1_dec = 0
    if p1_line < 0:
        p1_dec = (100/(-1*p1_line)) + 1
    else:
        p1_dec = p1_line/100 + 1
    p2_dec = 0
    if p2_line < 0:
        p2_dec = (100/(-1*p2_line)) + 1
    else:
        p2_dec = p2_line/100 + 1
    p1_return = (p1_dec * p1_wins + ties)/(p1_wins + p2_wins + ties)
    p2_return = (p2_dec * p2_wins + ties)/(p1_wins + p2_wins + ties)
    p1_returns.append(p1_return)
    p2_returns.append(p2_return)

mdf['P1_Return'] = pd.Series(p1_returns,name='P1_Return')
mdf['P2_Return'] = pd.Series(p2_returns,name='P2_Return')
print(mdf.head(50))









# end
