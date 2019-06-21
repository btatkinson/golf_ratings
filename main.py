
import sys
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import gc
import re
import datetime
import matplotlib.pyplot as plt

from settings import *
from helpers import *
from operator import itemgetter
from itertools import combinations
from sklearn.metrics import log_loss
from tqdm import tqdm

sys.path.insert(0, './classes')
from player import Player
from glicko import Glicko
from elo import Elo

def get_tournament_leaderboard(row):
    name = row['name']
    name = name.strip()
    name = name.replace(" ","")
    tour = row['tour']
    season = row['season']
    season = str(season)
    tournament_leaderboard_path = '../golf_scraper/leaderboards/'+season+'/'+tour+'/'+name+'.csv'
    return pd.read_csv(tournament_leaderboard_path)

player_database = pd.DataFrame(columns=['name','ielo','num_played','glicko','gvar','last_date','last_loc'])

# load player_database
player_database_path = './data/players.csv'
pdf = pd.read_csv(player_database_path)
# change to dictionary format bc it's quicker
pdf = pdf.set_index('name')
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

# sort by oldest first
sdf['end_date'] = pd.to_datetime(sdf['end_date'], format='%b %d %Y')
sdf = sdf.sort_values(by='end_date', ascending=True)

del cdf
gc.collect()

# iterate tournaments
START = 0
END = 2000
sdf = sdf[START:END]
print(sdf.head())

# initialize ratings objects
Glicko = Glicko()
Elo = Elo()

all_elo_loss = []
all_glicko_loss = []
all_log5_loss = []

# elo & glicko by rounds played


# track error round by round
# all_r1e_errors = []
# all_r2e_errors = []
# all_r3e_errors = []
# all_r4e_errors = []
#
# all_r1g_errors = []
# all_r2g_errors = []
# all_r3g_errors = []
# all_r4g_errors = []


# all_num_opps = []

for index, row in tqdm(sdf.iterrows()):

    not_valid = validate_tournament(row)
    if not_valid:
        continue

    # load tournament leaderboard based on inferred path
    tlb = get_tournament_leaderboard(row)
    start_date = tlb['start_date'].mode()[0]

    ## for testing ##
    dt_start = datetime.datetime.strptime(str(start_date), '%b %d %Y').date()
    if dt_start <= datetime.datetime.strptime('Jan 01 2019', '%b %d %Y').date():
        continue

    print(row['name'], row['season'])

    # list to contain player objects
    plist = []

    # error tracking
    telo_err = []
    tglicko_err = []
    tl5_err = []

    # by rounds played error tracking
    elo_brp = []
    glicko_brp = []

    # possible options to speed up here
    # could try subtracting sets, then multiple key lookup using pydash or itemgetter
    # note: tried itemgetter and pydash and neither were faster (bc they returned errors on missing)

    for i, r in tlb.iterrows():
        player = r['name']

        # name preprocessing
        player = name_pp(player)
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
                R4 = r['R4'],
                wins=dict['wins'],
                losses=dict['losses'],
                ties=dict['ties'],
                wl=dict['wl']
            )
        # if not in dict, initialize player class with new player settings
        else:
            PObj = Player(
                name=player,
                tour=r['tour'],
                ldate=start_date,
                cdate=start_date,
                R1 = r['R1'],
                R2 = r['R2'],
                R3 = r['R3'],
                R4 = r['R4']
            )

        plist.append(PObj)

    # separate round cols using regex
    col_text = ' '.join(list(tlb))
    rounds = re.findall(r"(\bR[1-4]{1}\b)", col_text)

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

        # add uncertainty if it's been awhile since they've played
        # if round == 'R1':
        #     for p in good_plist:
        #         if p.days_since is not None:
        #             if p.days_since >= 15:
        #                 p.gvar = add_uncertainty(p.gvar, p.days_since)

        # all combinations of players not cut or withdrawn
        combos = [c for c in combinations(good_plist,2)]

        # track number of opponents for Elo K val
        num_opps = len(good_plist) - 1

        # all_num_opps.append(num_opps)

        ## elo / log5 calc ##

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
                log5_x = l5_x(p1.wl,p2.wl)
                if p1_score == p2_score:
                    p1.add_tie()
                    p2.add_tie()
                else:
                    p1.add_win()
                    p2.add_loss()
                expected_result = Elo.x(p1.elo, p2.elo)
                p1_change, p2_change = Elo.get_ielo_delta(expected_result, margin, p1, p2, num_opps)
                # used for error tracking
                x = expected_result
                if p1_score == p2_score:
                    result = 0.5
                else:
                    result = 1
            else:
                # player 2 is winner
                log5_x = l5_x(p1.wl,p2.wl)
                p2.add_win()
                p1.add_loss()
                # order of arguments matters
                expected_result = Elo.x(p2.elo, p1.elo)
                p2_change, p1_change = Elo.get_ielo_delta(expected_result, margin, p2, p1, num_opps)
                # used for error tracking
                x = 1-expected_result
                result = 0

            elo_error = cross_entropy(x, result)
            l5_error = cross_entropy(log5_x, result)
            telo_err.append(elo_error)

            elo_brp.append([p1.rnds_played, elo_error])
            elo_brp.append([p2.rnds_played, elo_error])
            # if round == 'R1':
            #     all_r1e_errors.append(elo_error)
            # if round == 'R2':
            #     all_r2e_errors.append(elo_error)
            # if round == 'R3':
            #     all_r3e_errors.append(elo_error)
            # if round == 'R4':
            #     all_r4e_errors.append(elo_error)
            tl5_err.append(l5_error)
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
            # append result vs opponent
            for opponent, opp_score in opps:
                if opp_score == player_round_score:
                    result = 0.5
                elif opp_score < player_round_score:
                    result = 0
                else:
                    result = 1
                results.append([opponent, result])
            # Glicko class edits glicko rating of
            new_pobj, glicko_error, brp = Glicko.update(pobj, results)

            # for tracking round by round error
            # if round == 'R1':
            #     all_r1g_errors.append(glicko_error)
            # if round == 'R2':
            #     all_r2g_errors.append(glicko_error)
            # if round == 'R3':
            #     all_r3g_errors.append(glicko_error)
            # if round == 'R4':
            #     all_r4g_errors.append(glicko_error)

            tglicko_err.append(glicko_error)
            # adding by rounds played
            for ebrp in brp:
                glicko_brp.append(ebrp)
            new_pobjs.append(new_pobj)
        # reset all the player objects with the new ratings
        good_plist = new_pobjs

        # add rounds played
        for gp in good_plist:
            gp.rnds_played += 1
            gp.calc_win_loss()

        # recombine good_plist and bad_plist
        plist = good_plist + bad_plist

    # update dict
    for p in plist:
        # get stats into dict format
        stats = {'ielo': p.elo, 'rnds_played': p.rnds_played, 'glicko': p.glicko, 'gvar': p.gvar, 'gsig':p.gsig, 'last_date': start_date, 'last_loc': p.lloc,
        'wins':p.wins,'losses':p.losses,'ties':p.ties,'wl':p.wl}
        pdf[p.name] = stats
    # calculate error
    tournament_elo_loss = np.round(sum(telo_err)/len(telo_err),5)
    tournament_glicko_loss = np.round(sum(tglicko_err)/len(tglicko_err),5)
    tournament_log5_loss = np.round(sum(tl5_err)/len(tl5_err),5)
    all_elo_loss.append(tournament_elo_loss)
    all_glicko_loss.append(tournament_glicko_loss)
    all_log5_loss.append(tournament_log5_loss)


player_ratings = pd.DataFrame.from_dict(pdf, orient='index')
player_ratings.index.name=('name')
player_ratings = player_ratings.reset_index('name')
player_ratings = player_ratings.sort_values(by='glicko',ascending=False)
print(player_ratings.head(50))
player_ratings = player_ratings.sort_values(by='ielo',ascending=False)
print(player_ratings.head(50))

player_ratings.to_csv('./data/current_player_ratings.csv',index=False)

print('TOTAL AVERAGE ELO LOSS', str(np.round(sum(all_elo_loss)/len(all_elo_loss),5)))
print('TOTAL AVERAGE GLICKO LOSS', str(np.round(sum(all_glicko_loss)/len(all_glicko_loss),5)))
print('TOTAL AVERAGE LOG5 LOSS', str(np.round(sum(all_log5_loss)/len(all_log5_loss),5)))

print('LENGTH OF ELO BRP', len(elo_brp))
print('LENGTH OF Glicko BRP', len(glicko_brp))

elo_brp_df = pd.DataFrame(elo_brp,columns=['Rnds_Played', 'Error'])
elo_brp_gb = elo_brp_df.groupby(['Rnds_Played']).mean()
elo_brp_gb = elo_brp_gb.reset_index()

glicko_brp_df = pd.DataFrame(glicko_brp,columns=['Rnds_Played', 'Error'])
glicko_brp_gb = glicko_brp_df.groupby(['Rnds_Played']).mean()
glicko_brp_gb = glicko_brp_gb.reset_index()

elo_x = elo_brp_gb.Rnds_Played
elo_y = elo_brp_gb.Error

glicko_x = glicko_brp_gb.Rnds_Played
glicko_y = glicko_brp_gb.Error

fig, ax = plt.subplots(figsize=(15,7))
plt.scatter(elo_x, elo_y)
plt.scatter(glicko_x, glicko_y)

# fit line to them
b1, m1 = polyfit(elo_x, elo_y, 1)
b2, m2 = polyfit(glicko_x, glicko_y, 1)

plt.plot(elo_x, b1 + m1 * elo_x, '-', label="Elo")
plt.plot(glicko_x, b2 + m2 * glicko_x, '-', label="Glicko")

plt.xlabel("Rounds Played")
plt.ylabel("Error")
# xint = range(0, math.ceil(17)+1)
# plt.xticks(xint)
plt.legend(loc='upper left')

plt.show()

# print('TOTAL R1 ELO LOSS', str(np.round(sum(all_r1e_errors)/len(all_r1e_errors),5)))
# print('TOTAL R2 ELO LOSS', str(np.round(sum(all_r2e_errors)/len(all_r2e_errors),5)))
# print('TOTAL R3 ELO LOSS', str(np.round(sum(all_r3e_errors)/len(all_r3e_errors),5)))
# print('TOTAL R4 ELO LOSS', str(np.round(sum(all_r4e_errors)/len(all_r4e_errors),5)))
#
# print('TOTAL R1 ELO LOSS', str(np.round(sum(all_r1g_errors)/len(all_r1g_errors),5)))
# print('TOTAL R2 ELO LOSS', str(np.round(sum(all_r2g_errors)/len(all_r2g_errors),5)))
# print('TOTAL R3 ELO LOSS', str(np.round(sum(all_r3g_errors)/len(all_r3g_errors),5)))
# print('TOTAL R4 ELO LOSS', str(np.round(sum(all_r4g_errors)/len(all_r4g_errors),5)))

# print('AVERAGE NUM OPPONENTS', str(np.round(sum(all_num_opps)/len(all_num_opps),5)))







# end
