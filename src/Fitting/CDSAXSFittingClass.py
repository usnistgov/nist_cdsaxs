# Attempt to rewrite the CDSAXS functions in a class structure
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.special as sp
import matplotlib.patches as mpatches

class CDSAXS_fitter():
    
    def __init__(self,Intensitydata,Qxdata,Qzdata):
        self.Intensitydata=Intensitydata
        self.Qxdata=Qxdata
        self.Qzdata=Qzdata
        
    def importCDSAXS1D(self):
    # imports data from a 1D grating
        self.Intensity = np.loadtxt(self.Intensitydata)
        self.Qx=np.loadtxt(self.Qxdata)
        self.Qz=np.loadtxt(self.Qzdata)
    
        self.Intensity[self.Intensity == 0]=np.nan # replaces 
        self.Qx[self.Qx == 0]=np.nan
        self.Qz[self.Qz == 0]=np.nan
        return self.Intensity,self.Qx,self.Qz
    
    