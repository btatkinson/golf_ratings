import numpy as np
import random
import time

from operator import itemgetter
from pydash import at


# look up 100 players in 10000 player dict
dict_size = 100000
players = 100
idx = players-1

pdict = {}
for x in range(dict_size):
    name = random.randint(0, 100000)
    # simulate attributes
    a = random.randint(0, 100000)
    b = random.randint(0, 100000)
    c = random.randint(0, 100000)
    d = random.randint(0, 100000)
    e = random.randint(0, 100000)
    v = {'a':a, 'b':b, 'c':c,'d':d,'e':e}
    pdict[name] = v

player_names = list(pdict.keys())
# randomize
random.shuffle(player_names)
# desired_players
dp = player_names[:idx]

t0 = time.time()
for p in dp:
    if p in pdict:
        value = pdict[p]
t1 = time.time()
print("conventional: ", t1-t0)

t0 = time.time()
to_look_up = []
attrs = {}
for p in dp:
    if p in pdict:
        to_look_up.append(p)
list = itemgetter(*to_look_up)(pdict)

t1 = time.time()
print("itemgetter: ", t1-t0)
t0 = time.time()
list = at(pdict, *dp)
t1 = time.time()
print("pydash: ", t1-t0)









# end
