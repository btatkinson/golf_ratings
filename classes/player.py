import sys
import datetime

sys.path.insert(0, '../')
from settings import *

class Player(object):
    """docstring for Player."""

    def __init__(self,
        name=None,
        elo=ielo_set['init'],
        rnds_played = 0,
        glicko = glicko_set['init'],
        gvar = glicko_set['phi'],
        gsig = glicko_set['sigma'],
        ldate = None,
        cdate = None,
        lloc = None,
        cloc = None,
        R1=None,
        R2=None,
        R3=None,
        R4=None
        ):

        super(Player, self).__init__()
        self.name = name
        self.elo = elo
        self.rnds_played = rnds_played
        self.glicko = glicko
        self.gvar = gvar
        self.gsig = gsig
        self.ldate = ldate
        self.cdate = cdate
        self.days_since = self.days_since_last()
        self.lloc = lloc
        self.cloc = cloc
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4

    def days_since_last(self):
        dsl = 365
        if self.ldate is None:
            pass
        elif self.cdate is None:
            pass
        else:
            last_date = datetime.datetime.strptime(self.ldate, '%b %d %Y').date()
            current_date = datetime.datetime.strptime(self.cdate, '%b %d %Y').date()
            dsl = current_date - last_date
        return dsl
