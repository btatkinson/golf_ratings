import numpy as np
import pandas as pd
import math
import sys

sys.path.insert(0, '../')
from settings import *

class Elo(object):
    """docstring for Elo."""

    beta = ielo_set['beta']
    K = ielo_set['K']
    rK = relo_set['K']

    ielo_acp = ielo_set['ACP']
    ielo_c = ielo_set['C']

    def __init__(self):
        super(Elo, self).__init__()

    # for reg Elo
    def x(self, p1_elo, p2_elo):
        elo_diff = p1_elo - p2_elo
        return 1/(1 + math.pow(10,(-elo_diff / self.beta)))

    def get_delta(self, prob, tie):
        return self.rK * (1 - prob) if not tie else self.rK * (0.5-prob)

    def get_movm(self, margin):
        return 0.48*np.log(max(abs(margin), 1) + 1.0)
        # return 0.14*margin

    def get_acp(self, elo_diff):
        return (self.ielo_c / ((elo_diff) * self.ielo_acp + self.ielo_c))

    def get_gamma(self, margin, elo_diff):
        movm = self.get_movm(margin)
        acp = self.get_acp(elo_diff)
        return movm * acp

    def get_k(self, rp, ds, num_opps, round):
        # dimish as function of number of rp
        K = self.K

        rpf = 0.96 + math.exp(-.02 * rp + 0.50)

        K *= rpf

        # diminishing returns for larger field
        m = 100/num_opps
        m = (m - 1)/1.5
        m += 1
        K*=m

        # decreasing K during the tournament leads to better round by round errors
        # however, it leads to worse scores at the next tournament round 1

        # if round == 'R1':
        #     K *= 0.96
        # if round == 'R2':
        #     K *= 0.96
        # if round == 'R3':
        #     K *= 0.96
        # if round == 'R4':
        #     K *= 1.05


        # increase K after long layoff
        # convert to week units
        # ws = ds/7

        # 3-5 years (208 weeks since), layoff factor would be around 2.5
        # two weeks, layoff factor would be about 1
        # just trying linear first
        # if round == 'R1':
        #     lf = 0.94 + .02*ws
        # elif round == 'R2':
        #     lf = 0.94 + .015*ws
        # else:
        #     lf = 1
        #
        # K *= lf

        return K

    def get_ielo_delta(self, prob, margin, p1, p2, num_opps, round):
        gamma = self.get_gamma(margin, (p1.elo-p2.elo))

        p1_K = self.get_k(p1.rnds_played, p1.days_since, num_opps, round)
        p2_K = self.get_k(p2.rnds_played, p2.days_since, num_opps, round)

        p1_delta = (p1_K * gamma) * (1 - prob)
        p2_delta = -((p2_K * gamma) * (1 - prob))
        return p1_delta,p2_delta






# end
