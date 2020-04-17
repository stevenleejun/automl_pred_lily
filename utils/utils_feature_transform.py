# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:18:28 2018
@author: shuyun
"""
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import re
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from matplotlib.ticker import NullFormatter
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'


def log_trans(x, inverse=False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)
