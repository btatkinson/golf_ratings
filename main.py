
import sys
import numpy as np
import pandas as pd
import gc
import re

from settings import *
from helpers import *
from itertools import combinations

sys.path.insert(0, './classes')
from player import Player
from glicko import Glicko
from elo import Elo

player_database = pd.DataFrame(columns=['name','ielo','num_played','glicko','gvar','last_date','last_loc'])

# load player_database
player_database_path = './data/players.csv'
pdf = pd.read_csv(player_database_path)
pdf = pdf.set_index('name')

# change to dictionary format bc it's quicker
pdf = pdf.to_dict('index')

# load schedule
schedule_path = '../golf_scraper/sched.csv'
sdf = pd.read_csv(schedule_path)

# load list of tournaments that were scraped
collected_path = '../golf_scraper/has_saved.csv'
cdf = pd.read_csv(collected_path)
collected_ids = list(cdf.TID.unique())

# only need tournaments that have been scraped
sdf = sdf.loc[sdf['tid'].isin(collected_ids)]

del cdf
gc.collect()

# iterate tournaments
sdf = sdf[2:3]
print(sdf.head())

for index, row in sdf.iterrows():

#   load tournament leaderboard based on inferred path
    name = row['name']
    name = name.strip()
    name = name.replace(" ","")
    tour = row['tour']
    season = row['season']
    start_date = row['start_date']
    tournament_leaderboard_path = '../golf_scraper/leaderboards/'+season+'/'+tour+'/'+name+'.csv'
    tdf = pd.read_csv(tournament_leaderboard_path)

    # list to contain player objects
    plist = []
    for i, r in tdf.iterrows():
        player = r['name']
        # if in dict, initialize player class with data
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
                lloc = dict['last_loc'],
                cloc = None,
                R1 = r['R1'],
                R2 = r['R2'],
                R3 = r['R3'],
                R4 = r['R4']
            )
        # if not in dict, initialize player class with new player settings
        else:
            PObj = Player(
                name=player,
                cdate=start_date,
                R1 = r['R1'],
                R2 = r['R2'],
                R3 = r['R3'],
                R4 = r['R4']
            )

        plist.append(PObj)

    # separate round cols using regex
    col_text = ' '.join(list(tdf))
    rounds = re.findall(r"(\bR[1-4]{1}\b)", col_text)

    Glicko = Glicko()
    Elo = Elo()

    #############
    ##  TEST   ##
    #############
    # p1 = Player(name='Blake',glicko=1500, gvar=200, gsig=0.06)
    # p2 = Player(name='Kelly',glicko=1400, gvar=30)
    # p3 = Player(name='Eric',glicko=1550, gvar=100)
    # p4 = Player(name='Cailynn',glicko=1700, gvar=300)
    #
    # results = [[p2, 1],[p3, 0], [p4,0]]
    #
    # p1 = Glicko.update(p1, results)
    # print(p1.glicko, p1.gvar, p1.gsig)
    #################
    ##  END TEST   ##
    #################

    # find unique player combinations for elo calc
    for round in rounds:
        # throw out any player that doesn't have a round score
        # still keep them to update after tournament
        good_plist = []
        bad_plist = []
        for p in plist:
            round_score = getattr(p, round)
            include = validate(round_score)
            if include:
                good_plist.append(p)
            else:
                bad_plist.append(p)

        # all combinations of players not cut or withdrawn
        combos = [c for c in combinations(good_plist,2)]

        # track number of opponents for Elo K val
        num_opps = len(good_plist) - 1

        ## elo calc ##

        # track elo changes for everyone
        change_dict = {}
        for p in good_plist:
            change_dict[p.name] = 0

        for combo in combos:
            p1 = combo[0]
            p2 = combo[1]
            p1_score = getattr(p1, round)
            p2_score = getattr(p2, round)
            margin = abs(p2_score - p1_score)
            if p1_score <= p2_score:
                # player 1 is winner/draw
                expected_result = Elo.x(p1.elo, p2.elo)
                p1_change, p2_change = Elo.get_ielo_delta(expected_result, margin, p1, p2, num_opps)
            else:
                # player 2 is winner
                # order of arguments matters
                expected_result = Elo.x(p2.elo, p1.elo)
                p2_change, p1_change = Elo.get_ielo_delta(expected_result, margin, p2, p1, num_opps)

            change_dict[p1.name] += p1_change
            change_dict[p2.name] += p2_change

        # apply changes to player elo after round
        for p in good_plist:
            p.elo += change_dict[p.name]

        ## Glicko Calc ##

        # create list for player objects with update ratings after each round
        new_pobjs = []
        for pobj in good_plist:
            # add all opponents and their round scores for glicko calc
            opps = [[p, getattr(p, round)] for p in good_plist if p != pobj]

            results = []
            player_round_score = getattr(pobj, round)
            for opponent, opp_score in opps:
                if opp_score == player_round_score:
                    result = 0.5
                elif opp_score < player_round_score:
                    result = 0
                else:
                    result = 1
                results.append([opponent, result])
            # Glicko class edits glicko rating of
            new_pobj = Glicko.update(pobj, results)
            new_pobjs.append(new_pobj)
        # reset all the player objects with the new ratings
        good_plist = new_pobjs

        # add rounds played
        for gp in good_plist:
            gp.rnds_played += 1

        # recombine good_plist and bad_plist
        plist = good_plist + bad_plist


for p in plist:
    print(p.name, p.elo, p.glicko)



#
# for np in new_pobjs:
#     print(np.name, np.glicko)


    #       glicko calculation
    #       ielo calc
    #       update ratings

#   update ratings post tournament

# update ratings post all tournaments

# save player player_database







# end
