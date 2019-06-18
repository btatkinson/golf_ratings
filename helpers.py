import math
import numpy as np
import pandas as pd


def name_pp(name):
    if "(AM) " in name:
        name = name.replace("(AM) ","")
    return name

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
