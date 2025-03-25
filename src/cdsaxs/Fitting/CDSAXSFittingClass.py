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
        #return self.Intensity,self.Qx,self.Qz
        
    def FreeFormTrapezoid(self):
        H1 = self.Coord[0,3]
        H2 = self.Coord[0,3]
        self.form=np.zeros([len(self.Qx[:,1]),len(self.Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
        for i in range(int(self.Trapnumber)): # edit this to remove the need for the trapnumber variable
            H2 = H2+self.Coord[i,2]
            if i > 0:
                H1 = H1+self.Coord[i-1,2] 
            x1 = self.Coord[i,0]
            x4 = self.Coord[i,1]
            x2 = self.Coord[i+1,0]
            x3 = self.Coord[i+1,1]
            if x2==x1:
                x2=x2-0.000001
            if x4==x3:
                x4=x4-0.000001
            SL = self.Coord[i,2]/(x2-x1)
            SR = -self.Coord[i,2]/(x4-x3)
            
            A1 = (np.exp(1j*self.Qx*((H1-SR*x4)/SR))/(self.Qx/SR+self.Qz))*(np.exp(-1j*H2*(self.Qx/SR+self.Qz))-np.exp(-1j*H1*(self.Qx/SR+self.Qz)))
            A2 = (np.exp(1j*self.Qx*((H1-SL*x1)/SL))/(self.Qx/SL+self.Qz))*(np.exp(-1j*H2*(self.Qx/SL+self.Qz))-np.exp(-1j*H1*(self.Qx/SL+self.Qz)))
            self.form=self.form+(1j/self.Qx)*(A1-A2)*self.Coord[i,4]
        
    def TrapModelInitiatlize(self, Trapnumber, TPAR, SLD, I0, DW, Bk, Pitch):
        self.Trapnumber=Trapnumber
        self.TPAR = TPAR
        self.SLD = SLD
        self.I0=I0
        self.DW=DW
        self.Bk = Bk
        self.Pitch=Pitch
    
    def SymCoordAssign_SingleMaterial(self):
    # assigns trapezoid coordinates for a symmetric trapezoid
    # consider combining with SymCoordAssign with SLD as a flag

        self.Coord=np.zeros([self.Trapnumber+1,5,1])
        for T in range (self.Trapnumber+1):
            if T==0:
                self.Coord[T,0,0]=0
                self.Coord[T,1,0]=self.TPAR[0,0]
                self.Coord[T,2,0]=self.TPAR[0,1]
                self.Coord[T,3,0]=0
                self.Coord[T,4,0]=1 # SLD - assigned to be 1 for a single material
            else:
                self.Coord[T,0,0]=self.Coord[T-1,0,0]+0.5*(self.TPAR[T-1,0]-self.TPAR[T,0])
                self.Coord[T,1,0]=self.Coord[T,0,0]+self.TPAR[T,0]
                self.Coord[T,2,0]=self.TPAR[T,1]
                self.Coord[T,3,0]=0
                self.Coord[T,4,0]=1# SLD - assigned to be 1 for a single material
    
    def SimTrap_SM(self):
        
        self.SymCoordAssign_SingleMaterial()
        self.FreeFormTrapezoid() 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimInt = np.power(Formfactor,2)*self.I0+self.Bk
    
    
    def plotSymTrap(self):
        Coordp=np.zeros([self.Trapnumber+1,5,2])
        Coordp[:,:,0]=self.Coord[:,:,0]
        Coordp[:,:,1]=self.Coord[:,:,0]
        Coordp[:,0:1,1]=Coordp[:,0:1,1]+self.Pitch
        for S in range(1):
            h=0
            Lc= np.zeros([self.Trapnumber+1,2])
            Rc= np.zeros([self.Trapnumber+1,2])
            
            for i in range(self.Trapnumber+1):
                Lc[i,0]=Coordp[i,0,S]
                Rc[i,0]=Coordp[i,1,S]
                Lc[i,1]=h
                Rc[i,1]=h
                h=h+Coordp[i,2,S]
            plt.plot(Lc[:,0],Lc[:,1], color='black')
            plt.plot(Rc[:,0],Rc[:,1], color='black')
            Cc=np.zeros([2,2])
            for i in range(self.Trapnumber):
                Cc[0,0]=Lc[i+1,0]
                Cc[0,1]=Lc[i+1,1]
                Cc[1,0]=Rc[i+1,0]
                Cc[1,1]=Rc[i+1,1]
                plt.plot(Cc[:,0],Cc[:,1], color='black')
        plt.xlabel('Width (Å)')
        plt.ylabel('Height (Å)')             

        plt.show()
        plt.close()
        
    def PlotQzCut(self,numbercuts,scale):
        S=self.SimInt
        I=deepcopy(self.Intensity)   
        if scale =='yes': 
            for i in range(0,numbercuts):
                S[:,i]=S[:,i]/(50.**(i+1))
                I[:,i]=I[:,i]/(50.**(i+1))
        for i in range(numbercuts):
            plt.semilogy(self.Qz[:,i],I[:,i],'.', label='Exp '+str(i))
            plt.semilogy(self.Qz[:,i],S[:,i], label='Sim '+str(i), color='black')
        #plt.legend(loc='upper right')
        plt.xlabel('q ($Å^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.plot()
        
    def Misfit(self):
        self.Chi2= abs(np.log(self.Intensity)-np.log(self.SimInt))
        
        self.Chi2[np.isnan(self.Chi2)]=0
    
   