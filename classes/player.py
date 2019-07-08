import sys
import datetime
import ast

sys.path.insert(0, '../')
from settings import *
from helpers import *

ALPHA = asg_set['alpha']
MWL = asg_set['max_window_len']
MEWM = asg_set['min_ewm']

pvar_sms = asg_set['pvar_sms']
pvar_lgs = asg_set['pvar_lgs']
pct_sms = asg_set['pct_sms']
pct_lgs = asg_set['pct_lgs']

class Player(object):
    """docstring for Player."""

    def __init__(self,
        name=None,
        tour='PGA',
        asg=0,
        pvar=asg_set['pvar_sms'],
        elo=ielo_set['init'],
        relo=ielo_set['init'],
        rnds_played = 0,
        glicko = glicko_set['init'],
        gvar = glicko_set['phi'],
        gsig = glicko_set['sigma'],
        ldate = None,
        cdate = None,
        pr4 = False,
        R1=None,
        R2=None,
        R3=None,
        R4=None,
        prev_sgs = asg_set['init_pga'],
        wins=0,
        losses=0,
        ties=0,
        matches=0,
        wl=0.5
        ):

        super(Player, self).__init__()
        self.name = name
        self.asg = asg
        self.pvar = pvar
        self.elo = elo
        self.relo = relo
        self.glicko = glicko
        self.rnds_played = rnds_played
        self.gvar = gvar
        self.gsig = gsig
        self.ldate = ldate
        self.cdate = cdate
        self.pr4=pr4
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
        self.prev_sgs = prev_sgs
        self.days_since = self.days_since_last()
        self.calc_var()
        self.wins=wins
        self.losses=losses
        self.ties=ties
        self.matches=matches
        self.wl = wl

    def add_win(self):
        self.wins+=1
        self.matches+=1
        return
    def add_loss(self):
        self.losses+=1
        self.matches+=1
        return
    def add_tie(self):
        self.ties+=1
        self.matches+=1
        return

    def calc_wl(self):
        self.wl = ((0.5*self.ties)+self.wins)/(self.matches)
        return

    def calc_var(self):
        try:
            self.prev_sgs = [float(i) for i in self.prev_sgs]
        except:
            self.prev_sgs = ast.literal_eval(self.prev_sgs)
        if len(self.prev_sgs) <=0:
            self.pvar = pvar_sms
        elif self.rnds_played <= 100:
            self.pvar = pvar_sms
        else:
            self.pvar = (pct_lgs * pvar_lgs) + (1-pct_lgs)* np.var(self.prev_sgs)
        return

    def calc_new_asg(self):
        if len(self.prev_sgs) > MWL:
            self.prev_sgs = np.delete(self.prev_sgs, 0)

        ewm = ewma_vectorized(self.prev_sgs,ALPHA)
        asg = ewm[-1]
        if len(self.prev_sgs) >= MEWM:
            self.asg = asg
        else:
            self.asg= (sum(self.prev_sgs)/len(self.prev_sgs) + asg)/2
        return

    def days_since_last(self):
        dsl = 365
        if self.ldate is None:
            return None
        elif self.cdate is None:
            return None
        else:
            last_date = datetime.datetime.strptime(str(self.ldate), '%b %d %Y').date()
            current_date = datetime.datetime.strptime(str(self.cdate), '%b %d %Y').date()
            dsl = current_date - last_date
            days = dsl.days
        return days
