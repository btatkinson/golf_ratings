import math
import numpy as np
import pandas as pd
import scipy.stats
# from statistics import NormalDist

from settings import *

max_gvar = glicko_set['phi']
# uncertainty increase
u_i = glicko_set['u_i']
bearings = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"];

def cross_entropy(yHat, y):
    delta = .000015
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

def rmse(pred, actual):
    return math.sqrt((pred-actual)**2)

def get_l5_x(pa, pb):
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

def get_distance(lat1, lng1, lat2, lng2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lng2 - lng1) * p)) / 2
    return 0.6213712 * 12745.6 * np.arcsin(np.sqrt(a))

def get_direction(lat1,lng1,lat2,lng2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lng1 = math.radians(lng1)
    lng2 = math.radians(lng2)
    dLng = lng2 - lng1

    dPhi = math.log(math.tan(lat2/2.0+math.pi/4.0)/math.tan(lat1/2.0+math.pi/4.0))
    if abs(dLng) > math.pi:
         if dLng > 0.0:
             dLng = -(2.0 * math.pi - dLng)
         else:
             dLng = (2.0 * math.pi + dLng)

    return (math.degrees(math.atan2(dLng, dPhi)) + 360.0) % 360.0;

def get_cardinal(dir):
    index = dir - 22.5
    if index < 0:
        index += 360
    intdex = int(np.round(index/45,0))
    try:
        card = bearings[intdex]
    except:
        card = "Other"
    return card

def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def window_size(alpha, sum_proportion):
    # Increases with increased sum_proportion and decreased alpha
    # solve (1-alpha)**window_size = (1-sum_proportion) for window_size
    return int(np.log(1-sum_proportion) / np.log(1-alpha))

def asg_pred(p1m, p1v, p2m, p2v):
    # too slow to use scipy
    return scipy.stats.norm.cdf((p1m-p2m)/math.sqrt(p1v + p2v))
    # return (p1m-p2m)/math.sqrt(p1v + p2v)

def clean_ps(x):
    x = x.tolist()
    print(x)
    return x

name_dict = {
    'III Davis Love':'Davis Love III',
    'III Charles Howell':'Charles Howell III',
    'III Harold Varner':'Harold Varner III',
    'PELT Bo Van':'Bo Van Pelt',
    'JAGER Louis De':'Louis De Jager',
    'Miguel Ángel Jiménez':'Miguel Angel Jiménez',
    'K J Choi':'K.J. Choi',
    'Rory Mcilroy':'Rory McIlroy',
    'BELLO Rafa Cabrera':'Rafa Cabrera Bello',
    'Bryson Dechambeau':'Bryson DeChambeau',
    'Erik van Rooyen':'Erik Van Rooyen',
    'ROOYEN Erik Van':'Erik Van Rooyen',
    'Cheng Tsung Pan':'C.T. Pan',
    'Mike Lorenzo-vera':'Michael Lorenzo-vera',
    'ZYL Jaco Van':'Jaco Van Zyl',
    'Gonzalo Fdez-Castano':'Gonzalo Fernandez-Castano',
    'Gonzalo Fdez-castaño':'Gonzalo Fernandez-Castano',
    '"Ted Potter, Jr."':'Ted Potter, Jr.',
}

def name_pp(name):
    if "(AM) " in name:
        name = name.replace("(AM) ","")
    if name in name_dict:
        name = name_dict[name]
    name = name.replace('"','')
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


def apply_gpoly(x):
    return 5.305e-05 * math.pow(x,3) - 0.02604 * x**2 + 4.391*x - 251.2

def apply_gdo(x,ds):
    return x + (0.34837 * np.exp(-0.20651 * (ds/7)) - 0.25)

def apply_grp(x,rp):
    if rp >= 10:
        return x + (0.5398 * np.exp(-0.03349 * (rp/10)) -0.42)
    else:
        return -0.0007911 *math.pow(x,4) + 0.02037 *math.pow(x,3) - 0.1835 * x**2 + 0.6805 *x - 0.9421

def get_gsg(glicko,days_since,rnds_played):
    glicko = glicko/10
    # #convert to sg
    gsg = apply_gpoly(glicko)
    # #adjust for days off
    gsg = apply_gdo(gsg,days_since)
    # #adjust for rounds played
    gsg = apply_grp(gsg,rnds_played)
    return gsg

def apply_epoly(x):
    return -0.001049 * x**2 + 0.4673 * x - 46.92
def apply_edo(x,ds):
    return x + (0.25603 * np.exp(-0.21482 * (ds/7)) - 0.184)
def apply_erp(x,rp):
    if rp >= 10:
        return x + (0.3583 * np.exp(-0.02573 * (rp/10)) -0.17)
    elif rp >= 1:
        return x
    else:
        return x - 0.33
def get_esg(elo,days_since,rnds_played):
    elo = elo/10
    # #convert to sg
    esg = apply_epoly(elo)
    # #adjust for days off
    esg = apply_edo(esg,days_since)
    # #adjust for rounds played
    esg = apply_erp(esg,rnds_played)
    return esg
# determine variance
# PVar is based on Rounds Played
def apply_pvrp1(rp):
    return 2.000 * np.exp(-0.05430 * (rp/10)) + 7.76
# PVar is also based on Skill Level
def apply_pvrp2(esg):
    return .87129 * np.exp(-0.48229 * esg) + 7.3

def get_var(rp,esg):
    x1 = apply_pvrp1(rp)
    x2 = apply_pvrp2(esg)
    x = .3*x1 + .7*x2
    return 1.611664 * x -4.73032



# end
