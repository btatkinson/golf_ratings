
import sys
import numpy as np
import pandas as pd
import gc
import re

from settings import *

sys.path.insert(0, './classes')
from player import Player
from glicko import Glicko

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
sdf = sdf[:1]
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
    # divide into new players
    players = list(tdf.name.unique())
    # list to contain player objects
    plist = []
    for player in players:
        # if in dict, initialize player class with data
        if player in pdf:
            dict = pdf[player]
            PObj = Player(
                name=player,
                ielo=dict['ielo'],
                rnds_played=dict['rnds_played'],
                glicko = dict['glicko'],
                gvar = dict['gvar'],
                gsig = dict['gsig'],
                ldate = dict['last_date'],
                cdate = start_date,
                lloc = dict['last_loc'],
                cloc = None
            )
        # if not in dict, initialize player class with new player settings
        else:
            PObj = Player(name=player, cdate=start_date)
            # will use the following to store round scores
            pdf[player] = {
                    'ielo':ielo_set['init'],
                    'rnds_played':0,
                    'glicko':glicko_set['init'],
                    'gvar':glicko_set['phi'],
                    'gsig':glicko_set['sigma'],
                    'ldate': None,
                    'cdate': None,
                    'lloc': None,
                    'cloc': None
                    }

        plist.append(PObj)

    # separate round cols using regex
    col_text = ' '.join(list(tdf))
    rounds = re.findall(r"(\bR[1-4]{1}\b)", col_text)

    # assign round scores to player keys in player dictionary
    for index, row in tdf.iterrows():
        for round in rounds:
            pdf[row['name']][round] = row[round]

    Glicko = Glicko()
    for round in rounds:
        for pobj in plist:
            # add all opponents for glicko calc
            opps = [[p, pdf[p.name][round]] for p in plist if p != pobj]
            print(opps)


    #       glicko calculation
    #       ielo calc
    #       update ratings

#   update ratings post tournament

# update ratings post all tournaments

# save player player_database







# end
