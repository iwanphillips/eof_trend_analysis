import numpy as np
from numpy import meshgrid, deg2rad, gradient, sin, cos, sqrt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib
import matplotlib.cm as cm
from matplotlib import rc
from math import exp, floor
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import linalg as LA
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import gamma, factorial, hyp2f1
import scipy.signal as ss
from scipy.signal import savgol_filter
from scipy.stats import levy_stable, norm, expon, halfnorm
from scipy.stats import gaussian_kde
from scipy.fft import fft, ifft
import scipy as scipy
import scipy.odr as odr
import scipy as sc

import cartopy
from cartopy.crs import PlateCarree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point

from eofs.standard import Eof
from eofs.examples import example_data_path
import xarray as xr
from xarray import DataArray

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
import importlib

#import time
#import statsmodels.api as sm
#from sklearn.preprocessing import StandardScaler
#from numba import jit

from brokenaxes import brokenaxes

'''
# Using Mathematica code in python (for analytical results)

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
session = WolframLanguageSession('/Applications/Mathematica 13.1.app/Contents/MacOS/WolframKernel')
'''

blues, bupu = cm.get_cmap('Blues'), cm.get_cmap('BuPu')
rev_blues, rev_bupu = blues.reversed(), bupu.reversed()

params = {'xtick.labelsize': 22, 'ytick.labelsize': 22,
              'legend.fontsize': 20, 'axes.labelsize': 22,
              'axes.titlesize': 22,  'font.size': 22, 
              'legend.handlelength': 2}

def resize_colobar(event):
    plt.draw()

    posn = ax.get_position()
    colorbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                              0.04, axpos.height])


                            
