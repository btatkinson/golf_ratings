import math
import numpy as np
import pandas as pd


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
