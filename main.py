
import sys
import numpy as np
import pandas as pd
import gc
import re
import datetime
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize

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

gc.collect()
CALC_LOG_5=False

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
all_l5_loss = []

# elo & glicko by rounds played

# track error by season
# season_errors = {}
# season_errors['errors'] = {}
# season_errors['counts'] = {}
#
# tours = ['PGA','Euro']
#
# seasons = list(sdf.season.unique())
# for season in seasons:
#     str_season = str(season)
#     season_errors['errors'][str_season] = {}
#     season_errors['counts'][str_season] = {}
#     for tour in tours:
#         season_errors['errors'][str_season][tour] = {}
#         season_errors['counts'][str_season][tour] = {}
#
#         season_errors['errors'][str_season][tour]['elo'] = 0
#         season_errors['errors'][str_season][tour]['glicko'] = 0
#         season_errors['errors'][str_season][tour]['l5'] = 0
#
#         season_errors['counts'][str_season][tour]['elo'] = .0001
#         season_errors['counts'][str_season][tour]['glicko'] = .0001
#         season_errors['counts'][str_season][tour]['l5'] = .0001

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

elo_brp = []
glicko_brp = []
l5_brp = []

for index, row in tqdm(sdf.iterrows()):

    not_valid = validate_tournament(row)
    if not_valid:
        continue

    # load tournament leaderboard based on inferred path
    tlb = get_tournament_leaderboard(row)
    start_date = tlb['start_date'].mode()[0]

    season = str(row['season'])

    # for testing ##
    dt_start = datetime.datetime.strptime(str(start_date), '%b %d %Y').date()
    if dt_start <= datetime.datetime.strptime('Aug 01 2015', '%b %d %Y').date():
        continue

    # print(row['name'], row['season'])

    # list to contain player objects
    plist = []

    # error tracking
    telo_err = []
    tglicko_err = []
    tl5_err = []

    # track round error
    relo_errors = []

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
                matches=dict['matches'],
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
                p.rnds_played += 1
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
                if CALC_LOG_5:
                    log5_x = l5_x(p1.wl,p2.wl)
                if p1_score == p2_score:
                    p1.add_tie()
                    p2.add_tie()
                else:
                    p1.add_win()
                    p2.add_loss()
                expected_result = Elo.x(p1.elo, p2.elo)
                p1_change, p2_change = Elo.get_ielo_delta(expected_result, margin, p1, p2, num_opps, round)
                # used for error tracking
                x = expected_result
                if p1_score == p2_score:
                    result = 0.5
                else:
                    result = 1
            else:
                # player 2 is winner
                if CALC_LOG_5:
                    log5_x = l5_x(p1.wl,p2.wl)
                p2.add_win()
                p1.add_loss()
                # order of arguments matters
                expected_result = Elo.x(p2.elo, p1.elo)
                p2_change, p1_change = Elo.get_ielo_delta(expected_result, margin, p2, p1, num_opps, round)
                # used for error tracking
                x = 1-expected_result
                result = 0

            elo_error = cross_entropy(x, result)
            elo_brp.append([p1.rnds_played, elo_error])
            elo_brp.append([p2.rnds_played, elo_error])
            telo_err.append(elo_error)
            # if round == 'R1':
            #     all_r1e_errors.append(elo_error)
            # if round == 'R2':
            #     all_r2e_errors.append(elo_error)
            # if round == 'R3':
            #     all_r3e_errors.append(elo_error)
            # if round == 'R4':
            #     all_r4e_errors.append(elo_error)

            if CALC_LOG_5:
                l5_error = cross_entropy(log5_x, result)
                l5_brp.append([p1.rnds_played, l5_error])
                l5_brp.append([p2.rnds_played, l5_error])
                tl5_err.append(l5_error)

            change_dict[p1.name] += p1_change
            change_dict[p2.name] += p2_change

        # apply changes to player elo & log_5 after round
        for p in good_plist:
            p.elo += change_dict[p.name]

        if CALC_LOG_5:
            for p in good_plist:
                p.calc_win_loss()

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
            # Glicko class edits glicko rating of player
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

        # recombine good_plist and bad_plist
        plist = good_plist + bad_plist

    # update dict
    for p in plist:
        # get stats into dict format
        stats = {'ielo': p.elo, 'rnds_played': p.rnds_played, 'glicko': p.glicko, 'gvar': p.gvar, 'gsig':p.gsig, 'last_date': start_date, 'last_loc': p.lloc,
        'wins':p.wins,'losses':p.losses,'ties':p.ties,'matches':p.matches,'wl':p.wl}
        pdf[p.name] = stats
    # calculate error
    tournament_elo_loss = np.round(sum(telo_err)/len(telo_err),5)
    tournament_glicko_loss = np.round(sum(tglicko_err)/len(tglicko_err),5)
    all_elo_loss.append(tournament_elo_loss)
    all_glicko_loss.append(tournament_glicko_loss)

    if CALC_LOG_5:
        tournament_l5_loss = np.round(sum(tl5_err)/len(tl5_err),5)
        all_l5_loss.append(tournament_l5_loss)

    # season_errors['errors'][str(season)][row['tour']]['elo'] += tournament_elo_loss
    #
    # season_errors['errors'][str(season)][row['tour']]['glicko'] += tournament_glicko_loss
    #
    # season_errors['errors'][str(season)][row['tour']]['l5'] += tournament_l5_loss
    #
    # season_errors['counts'][str(season)][row['tour']]['elo'] += 1
    #
    # season_errors['counts'][str(season)][row['tour']]['glicko'] += 1
    #
    # season_errors['counts'][str(season)][row['tour']]['l5'] += 1



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
if CALC_LOG_5:
    print('TOTAL AVERAGE LOG5 LOSS', str(np.round(sum(all_l5_loss)/len(all_l5_loss),5)))
    l5_brp_df = pd.DataFrame(l5_brp,columns=['Rnds_Played', 'Error'])
    l5_brp_gb = l5_brp_df.groupby(['Rnds_Played']).count()
    l5_brp_gb = l5_brp_gb.reset_index()
    l5_brp_gb.to_csv('./data/rbr/log_5_counts.csv',index='False')

elo_brp_df = pd.DataFrame(elo_brp,columns=['Rnds_Played', 'Error'])
print('TOTAL BY ROUNDS PLAYED ERROR COUNT', len(elo_brp_df))
elo_brp_gb = elo_brp_df.groupby(['Rnds_Played']).mean()
elo_brp_gb = elo_brp_gb.reset_index()

elo_brp_count = elo_brp_df.groupby(['Rnds_Played']).count()
elo_brp_count = elo_brp_count.reset_index()
#
glicko_brp_df = pd.DataFrame(glicko_brp,columns=['Rnds_Played', 'Error'])
glicko_brp_gb = glicko_brp_df.groupby(['Rnds_Played']).mean()
glicko_brp_gb = glicko_brp_gb.reset_index()

glicko_brp_count = glicko_brp_df.groupby(['Rnds_Played']).count()
glicko_brp_count = glicko_brp_count.reset_index()

# l5_brp_df = pd.DataFrame(l5_brp,columns=['Rnds_Played', 'Error'])
# l5_brp_gb = l5_brp_df.groupby(['Rnds_Played']).mean()
# l5_brp_gb = l5_brp_gb.reset_index()

# groupby data to csv
elo_brp_gb.to_csv('./data/rbr/elo_brp.csv',index='False')
glicko_brp_gb.to_csv('./data/rbr/glicko_brp.csv',index='False')
# l5_brp_gb.to_csv('./data/rbr/l5_brp.csv',index='False')

elo_brp_count.to_csv('./data/rbr/elo_brp_count.csv',index='False')
glicko_brp_count.to_csv('./data/rbr/glicko_brp_count.csv',index='False')

# print('TOTAL R1 ELO LOSS', str(np.round(sum(all_r1e_errors)/len(all_r1e_errors),5)))
# print('TOTAL R2 ELO LOSS', str(np.round(sum(all_r2e_errors)/len(all_r2e_errors),5)))
# print('TOTAL R3 ELO LOSS', str(np.round(sum(all_r3e_errors)/len(all_r3e_errors),5)))
# print('TOTAL R4 ELO LOSS', str(np.round(sum(all_r4e_errors)/len(all_r4e_errors),5)))
#
# print('TOTAL R1 GLICKO LOSS', str(np.round(sum(all_r1g_errors)/len(all_r1g_errors),5)))
# print('TOTAL R2 GLICKO LOSS', str(np.round(sum(all_r2g_errors)/len(all_r2g_errors),5)))
# print('TOTAL R3 GLICKO LOSS', str(np.round(sum(all_r3g_errors)/len(all_r3g_errors),5)))
# print('TOTAL R4 GLICKO LOSS', str(np.round(sum(all_r4g_errors)/len(all_r4g_errors),5)))

# ses = []
# systems = ['elo','glicko','l5']
# for season in seasons:
#     for tour in tours:
#         for system in systems:
#             er = season_errors['errors'][str(season)][tour][system]
#             count = season_errors['counts'][str(season)][tour][system]
#             er_avg = er/count
#             ses.append([season, tour, system, er_avg])
#
# sea_by_sea = pd.DataFrame(ses, columns=['Season','Tour', 'System', 'Error'])
#
# sea_by_sea.to_csv('./data/sea_by_sea.csv')





# end
