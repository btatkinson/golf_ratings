import sys
import datetime

sys.path.insert(0, '../')
from settings import *

class Player(object):
    """docstring for Player."""

    def __init__(self,
        name=None,
        tour='PGA',
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
        cloc = None,
        R1=None,
        R2=None,
        R3=None,
        R4=None,
        wins=int(0),
        losses=int(0),
        ties=int(0),
        matches=0,
        wl=float(0.5)
        ):

        super(Player, self).__init__()
        self.name = name
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
        self.cloc = cloc
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
        self.wins=wins
        self.losses=losses
        self.ties=ties
        self.matches=matches
        self.wl=wl
        self.days_since = self.days_since_last()
        self.dist_from_last = None
        self.bearing_from_last = None

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

    def calc_win_loss(self):
        if self.rnds_played <= 1:
            self.wl = 0.5
        else:
            self.wl = (self.wins + 0.5*self.ties)/self.matches
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
