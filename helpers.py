import math
import numpy as np
import pandas as pd

from settings import *

max_gvar = glicko_set['phi']
# uncertainty increase
u_i = glicko_set['u_i']

def cross_entropy(yHat, y):
    delta = .00015
    if yHat >= 1:
        yHat = 1
        yHat -= delta
    if yHat <= 0:
        yHat = 0
        yHat += delta
    if y == 1:
      return -math.log10(yHat)
    else:
      return -math.log10(1 - yHat)

def l5_x(pa, pb):
    if pa == pb:
        return 0.5
    denom = ((pa + pb) - (2 * pa * pb))
    if  denom == 0:
        return 0.5
    return (pa - (pa * pb))/denom

def add_uncertainty(gvar, days_since):
    # convert days to units of two weeks
    time_passed = days_since/14
    return min(math.sqrt((gvar**2)+(time_passed * (u_i **2))),max_gvar)

name_dict = {
    'III Davis Love':'Davis Love III',
    'III Charles Howell':'Charles Howell III',
    'PELT Bo Van':'Bo Van Pelt',
    'Miguel Ángel Jiménez':'Miguel Angel Jiménez',
    'K J Choi':'K.J. Choi',
    'Rory Mcilroy':'Rory McIlroy'
}

def name_pp(name):
    if "(AM) " in name:
        name = name.replace("(AM) ","")
    if name in name_dict:
        name = name_dict[name]
    return name

def validate_tournament(row):
    not_valid= False

    # double scrape majors
    if row['name'] == 'USPGAChampionship':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] == 'USPGACHAMPIONSHIP':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] == 'U.S.OpenChampionship':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] == 'U.S.Open':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] == 'USOpen':
        if row['tour'] == 'Euro':
            not_valid= True
    if 'TheOpenChampionship' in row['name']:
        if row['tour'] == 'PGA':
            not_valid= True
    if row['name'] =='MastersTournament':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] =='THEMASTERS':
        if row['tour'] == 'Euro':
            not_valid= True

    # double scrape WGCs
    if row['name'] =='WGC-AmericanExpressChampionship':
        if row['tour'] == 'Euro':
            not_valid= True
    if row['name'] =='WGC-NECInvitational':
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-EMC2' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-CAChampionship' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-CadillacChampionship' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-BridgestoneInvitational' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-HSBCChampions' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True
    if 'WGC-MexicoChampionship' in row['name']:
        if row['tour'] == 'Euro':
            not_valid= True

    return not_valid

def validate(score):
    isValid = True
    try:
        score = int(score)
    except:
        return False

    if score < 50:
        isValid = False
    if score > 120:
        isValid = False
    return isValid









# end
