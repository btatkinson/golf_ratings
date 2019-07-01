import sys
import datetime

sys.path.insert(0, '../')
from settings import *
from helpers import *

ALPHA = asg_set['alpha']
MWL = asg_set['max_window_len']
MEWM = asg_set['min_ewm']

class Player(object):
    """docstring for Player."""

    def __init__(self,
        name=None,
        tour='PGA',
        asg=0,
        elo=ielo_set['init'],
        rnds_played = 0,
        glicko = glicko_set['init'],
        gvar = glicko_set['phi'],
        gsig = glicko_set['sigma'],
        ldate = None,
        cdate = None,
        llat = None,
        llng = None,
        pr4 = False,
        R1=None,
        R2=None,
        R3=None,
        R4=None,
        prev_sgs = np.array([]),
        dist_from_last=0,
        bearing_from_last=0
        ):

        super(Player, self).__init__()
        self.name = name
        self.asg = asg
        if tour == 'PGA':
            self.elo = elo
            self.glicko = glicko
        else:
            self.elo = euro_init['elo']
            self.glicko = euro_init['glicko']
        self.rnds_played = rnds_played
        self.gvar = gvar
        self.gsig = gsig
        self.ldate = ldate
        self.cdate = cdate
        self.llat = llat
        self.llng = llng
        self.pr4=pr4
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
        self.prev_sgs = prev_sgs
        self.days_since = self.days_since_last()
        self.dist_from_last = dist_from_last
        self.bearing_from_last = bearing_from_last

    def calc_new_asg(self):
        temp = None
        if len(self.prev_sgs) > MWL:
            self.prev_sgs = np.delete(self.prev_sgs, 0)
        if len(self.prev_sgs) >= MEWM:
            ewm = ewma_vectorized(self.prev_sgs,ALPHA)
            self.asg = ewm[-1]
        else:
            self.asg=sum(self.prev_sgs)/len(self.prev_sgs)
        return temp

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
