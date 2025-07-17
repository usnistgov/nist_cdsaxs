# Functions for loading and fitting CDSAXS data - currently similar to legacy from, consider adapting to object structure for additional flexibility
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.special as sp
import matplotlib.patches as mpatches

# FreeFormTrapezoid - uses coordinate input to calculate simlate amplitude of the form factor a trapezoid. Coordinates are of the form [xL,xR,H] for each layer. Does not need to be symmetric - code should be checked for computational efficiency
def FreeFormTrapezoid(Coord,Qx,Qz,Trapnumber):
    H1 = Coord[0,3]
    H2 = Coord[0,3]
    form=np.zeros([len(Qx[:,1]),len(Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
    for i in range(int(Trapnumber)): # edit this to remove the need for the trapnumber variable
        H2 = H2+Coord[i,2]
        if i > 0:
            H1 = H1+Coord[i-1,2] 
        x1 = Coord[i,0]
        x4 = Coord[i,1]
        x2 = Coord[i+1,0]
        x3 = Coord[i+1,1]
        if x2==x1:
            x2=x2-0.000001
        if x4==x3:
            x4=x4-0.000001
        SL = Coord[i,2]/(x2-x1)
        SR = -Coord[i,2]/(x4-x3)
        
        A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
        A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
        form=form+(1j/Qx)*(A1-A2)*Coord[i,4]
    return form


def ConeFourierTransform(CPAR,ConeNumber,Qr,Qz,Discretization,SLD):
    # Fourier transform for a cone in cylindrical coordinates (Qr,Qz) 
    H1 = 0
    H2 = 0
    Form=np.zeros([int(len(Qr[:,0])),int(len(Qr[0,:]))])
    
    for i in range (ConeNumber):
        H2=H2+CPAR[i,1]
        z=np.zeros([int(Discretization[i])])
        stepsize=CPAR[i,1]/Discretization[i]
        z=np.arange(H1,H2+0.01,stepsize)
        if i > 0 :
            H1=H1+CPAR[i-1,1]
            
        z=np.arange(H1,H2+0.01,stepsize)
        R1=CPAR[i,0]
        R2=CPAR[i+1,0]
        if R1==R2:
            R1=R1+0.000001
        Slope=(H2-H1)/(R2-R1)
        for ii in range(len(z)-1):
            RI1=(z[ii]-H1)/Slope+R1
            RI2=(z[ii+1]-H1)/Slope+R1
            fa=2*np.pi*RI1/Qr*sp.jv(1,Qr*RI1)*np.exp(1j*Qz*z[ii])
            fb=2*np.pi*RI2/Qr*sp.jv(1,Qr*RI2)*np.exp(1j*Qz*z[ii+1])
            Form=Form+stepsize*(fb+fa)/2*SLD[i]
    return Form


def importCDSAXS1D(Intensitydata,Qxdata,Qzdata):
    # imports data from a 1D grating
    Intensity = np.loadtxt(Intensitydata)
    Qx=np.loadtxt(Qxdata)
    Qz=np.loadtxt(Qzdata)
    
    Intensity[Intensity == 0]=np.nan # replaces 
    Qx[Qx == 0]=np.nan
    Qz[Qz == 0]=np.nan
    return Intensity,Qx,Qz

def importCDSAXS1D(Intensitydata,Qxdata,Qzdata):
    # imports data from a 1D grating
    Intensity = np.loadtxt(Intensitydata)
    Qx=np.loadtxt(Qxdata)
    Qz=np.loadtxt(Qzdata)
    
    Intensity[Intensity == 0]=np.nan # replaces 
    Qx[Qx == 0]=np.nan
    Qz[Qz == 0]=np.nan
    return Intensity,Qx,Qz

def SymCoordAssign(TPAR,SLD):
    # assigns trapezoid coordinates for a symmetric trapezoid
    
    Trapnumber=len(TPAR[:,0])-1
    Coord=np.zeros([Trapnumber+1,5,1])
    for T in range (Trapnumber+1):
        if T==0:
            Coord[T,0,0]=0
            Coord[T,1,0]=TPAR[0,0]
            Coord[T,2,0]=TPAR[0,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[0,0]
        else:
            Coord[T,0,0]=Coord[T-1,0,0]+0.5*(TPAR[T-1,0]-TPAR[T,0])
            Coord[T,1,0]=Coord[T,0,0]+TPAR[T,0]
            Coord[T,2,0]=TPAR[T,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[T,0]
 
    return (Coord)

def SymCoordAssign_SingleMaterial(TPAR):
    # assigns trapezoid coordinates for a symmetric trapezoid
    # consider combining with SymCoordAssign with SLD as a flag
    Trapnumber=len(TPAR[:,0])-1
    Coord=np.zeros([Trapnumber+1,5,1])
    for T in range (Trapnumber+1):
        if T==0:
            Coord[T,0,0]=0
            Coord[T,1,0]=TPAR[0,0]
            Coord[T,2,0]=TPAR[0,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=1 # SLD - assigned to be 1 for a single material
        else:
            Coord[T,0,0]=Coord[T-1,0,0]+0.5*(TPAR[T-1,0]-TPAR[T,0])
            Coord[T,1,0]=Coord[T,0,0]+TPAR[T,0]
            Coord[T,2,0]=TPAR[T,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=1# SLD - assigned to be 1 for a single material
 
    return (Coord)


def SimTrap(Qx,Qz,FITPAR,Trapnumber,SLD):
    TPARs=np.zeros([Trapnumber+1,2])
    TPARs[:,0:2]=np.reshape(FITPAR[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    SPAR=FITPAR[Trapnumber*2+2:Trapnumber*2+5]
    (Coord)= SymCoordAssign(TPARs,SLD)
    F1 = FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber) 
    
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5)
    Formfactor=F1*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*SPAR[1]+SPAR[2]
    return SimInt

def SimTrap_SM(Qx,Qz,FITPAR,Trapnumber):
    TPARs=np.zeros([Trapnumber+1,2])
    TPARs[:,0:2]=np.reshape(FITPAR[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    SPAR=FITPAR[Trapnumber*2+2:Trapnumber*2+5]
    (Coord)= SymCoordAssign_SingleMaterial(TPARs)
    F1 = FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber) 
    
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5)
    Formfactor=F1*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*SPAR[1]+SPAR[2]
    return SimInt



def PBA_SymTrap(TPAR,SPAR,Bounds):
    # Assigns parameter bounds for the MCMC and DE algorithm, this version is
    # Bounds are assigned from 0.01-0.99
    SPARLB=SPAR[0:4]*(1-Bounds)
    SPARUB=SPAR[0:4]*(1+Bounds)

    FITPAR=TPAR[:,0:2].ravel()
    FITPARLB=FITPAR*(1-Bounds)
    FITPARUB=FITPAR*(1+Bounds)
    FITPAR=np.append(FITPAR,SPAR)
       
    FITPARLB=np.append(FITPARLB,SPARLB)
    
    FITPARUB=np.append(FITPARUB,SPARUB)
    
    return (FITPAR,FITPARLB,FITPARUB)





#### - Plotting functions

def plotSymTrap(Coord,Trapnumber,Pitch,SampleName):
    Coordp=np.zeros([Trapnumber+1,5,2])
    Coordp[:,:,0]=Coord[:,:,0]
    Coordp[:,:,1]=Coord[:,:,0]
    Coordp[:,0:1,1]=Coordp[:,0:1,1]+Pitch
    for S in range(1):
        h=0
        Lc= np.zeros([Trapnumber+1,2])
        Rc= np.zeros([Trapnumber+1,2])
        
        for i in range(Trapnumber+1):
            Lc[i,0]=Coordp[i,0,S]
            Rc[i,0]=Coordp[i,1,S]
            Lc[i,1]=h
            Rc[i,1]=h
            h=h+Coordp[i,2,S]
        plt.plot(Lc[:,0],Lc[:,1], color='black')
        plt.plot(Rc[:,0],Rc[:,1], color='black')
        Cc=np.zeros([2,2])
        for i in range(Trapnumber):
            Cc[0,0]=Lc[i+1,0]
            Cc[0,1]=Lc[i+1,1]
            Cc[1,0]=Rc[i+1,0]
            Cc[1,1]=Rc[i+1,1]
            plt.plot(Cc[:,0],Cc[:,1], color='black')
    plt.xlabel('Width (A)')
    plt.ylabel('Height (A)')             

    plt.show()
    plt.close()

def PlotQzCut(Qx,Qz,FITPAR,Trapnumber,ExpI,numbercuts,scale):
    S=SimTrap_SM(Qx,Qz,FITPAR,Trapnumber)
    I=deepcopy(ExpI)   
    if scale =='yes': 
        for i in range(0,numbercuts):
            S[:,i]=S[:,i]/(50.**(i+1))
            I[:,i]=I[:,i]/(50.**(i+1))
    for i in range(numbercuts):
        plt.semilogy(Qz[:,i],I[:,i],'.', label='Exp '+str(i))
        plt.semilogy(Qz[:,i],S[:,i], label='Sim '+str(i), color='black')
    #plt.legend(loc='upper right')
    plt.xlabel('q ($Å^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.plot()
    
def PlotQzCut_NoScale(Qx,Qz,FITPAR,Trapnumber,ExpI,numbercuts):
    S=SimTrap_SM(Qx,Qz,FITPAR,Trapnumber)
    I=deepcopy(ExpI)    
    for i in range(numbercuts):
        plt.semilogy(Qz[:,i],I[:,i],'.', label='Exp '+str(i))
        plt.semilogy(Qz[:,i],S[:,i], label='Sim '+str(i))
    plt.plot()

def PlotQzCutComp(Qz,FITPAR,Trapnumber,ExpI,numbercuts):
    colorlist=('black','orange')
    I=deepcopy(ExpI)
    for k,v in enumerate(FITPAR):
        S=SimTrap(v,Trapnumber)
            
        for i in range(0,numbercuts):
            S[:,i]=S[:,i]/(50.**(i+1))
            I[:,i]=I[:,i]/(50.**(i+1))
        for i in range(numbercuts):
            if k==0:
                plt.semilogy(Qz[:,i],I[:,i],'.')
                plt.semilogy(Qz[:,i],S[:,i],color=colorlist[k])
            else:
                plt.semilogy(Qz[:,i],S[:,i], color=colorlist[k])
        #plt.legend(loc='upper right')
    patch1 = mpatches.Patch(color='black', label='Initial')
    plt.legend(handles=[patch1])
    patch2 = mpatches.Patch(color='orange', label='Final')
    plt.legend(handles=[patch2])
    plt.legend(loc='upper right')
    plt.plot()
    
def TPARfromFITPAR(FITPAR,Trapnumber):
    TPARs=np.zeros([Trapnumber+1,2])
    TPARs[:,0:2]=np.reshape(FITPAR[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    SPAR=FITPAR[Trapnumber*2+2:Trapnumber*2+5]
    return TPARs,SPAR

def Misfit(Exp,Sim):
    Chi2= abs(np.log(Exp)-np.log(Sim))
    #ms=np.zeros([len(Exp[:,1]),len(Exp[1,:]),2])
    #ms[:,:,0]=Sim
    #ms[:,:,1]=Exp
    #MS= np.nanmin(ms,2)
    #Chi2=np.power((D/MS),2)
    Chi2[np.isnan(Chi2)]=0
    return Chi2