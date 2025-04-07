# Attempt to rewrite the CDSAXS functions in a class structure
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.special as sp
import matplotlib.patches as mpatches
from scipy.optimize import differential_evolution
import math
import re
import pandas as pd
#Examples of assigning attributes names with a variable
# class MyAttribute:
#     def __set_name__(self, owner, name):
#         self.name = name

# class MyClass:
#     attr1 = MyAttribute()
#     attr2 = MyAttribute()

# # Accessing the attribute names
# print(MyClass.attr1.name) # Output: attr1
# print(MyClass.attr2.name) # Output: attr2


# class MyClass:
#     def __init__(self, attribute_name, value):
#         setattr(self, attribute_name, value)

# # Creating an instance and setting an attribute dynamically
# instance = MyClass("dynamic_attr", 10)
# print(instance.dynamic_attr) # Output: 10


class CDSAXS_Model():
    
    def __init__(self,geometry,model,layers,PAR,SLD,DW,I0,Bk,Pitch):
        self.geoemtry=geometry
        self.model=model
        self.PAR=PAR
        self.PAR_Initial=PAR
        self.layers=layers
        self.SLD=SLD
        self.DW=DW
        self.I0=I0
        self.Bk=Bk
        self.DW_Initial=DW
        self.I0_Initial=I0
        self.Bk_Initial=Bk
        self.Pitch=Pitch
        self.SimPar=np.append(self.PAR.ravel(),[self.I0,self.DW,self.Bk])
  
        
        
### Data imports
        
    def importCDSAXSQxQz(self,Intensitydata,Qxdata,Qzdata):
        # imports data from a 1D grating
            self.Intensity = np.loadtxt(Intensitydata)
            self.Qx=np.loadtxt(Qxdata)
            self.Qz=np.loadtxt(Qzdata)
        
            self.Intensity[self.Intensity == 0]=np.nan # replaces 
            self.Qx[self.Qx == 0]=np.nan
            self.Qz[self.Qz == 0]=np.nan
            self.numberpoints=np.sum(np.isreal(self.Intensity))
            self.SymCoordAssign_SingleMaterial()
            self.SimTrap_SM()
            self.SimInt_Initial=self.SimInt
            self.GF = self.GF_calc(self.SimInt)
            self.GF_Initial=self.GF
            self.BIC= self.BIC_calc(self.GF)
            self.GF_Initial=self.BIC
   
    def importCDSAXS_GUI(self,Datafile):
        Data=pd.read_csv(Datafile)
        # checks the number of cuts
        num_columns = len(Data.columns)
        numbercuts =num_columns//2
               
        headers = Data.columns.tolist()
    
        qxlist = []
        
        # Check every other column starting with index 1 (second column)
        for i in range(1, len(headers), 2):
            # Look for pattern 'qx = number' in the header
            match = re.search(r'qx\s*=\s*(\d+\.?\d*)', headers[i])
            if match:
                number = float(match.group(1))
                # Convert to int if it's a whole number
                if number.is_integer():
                    number = int(number)
                qxlist.append(number)
        #Converts to numpy
        Data1=Data.to_numpy()
        self.Intensity=np.zeros([len(Data1[:,0]),numbercuts])
        self.Qz=np.zeros([len(Data1[:,0]),numbercuts])
        for i in range(0,numbercuts):
            self.Intensity[:,i]=Data1[:,(i*2+1)]
            self.Qz[:,i]=Data1[:,(i*2)]
        self.Qx=self.Qz.copy()
        self.Qx[~np.isnan(self.Qx)] = 1
        for k, v in enumerate(qxlist):
            self.Qx[:,k]=self.Qx[:,k]*v     
        self.numberpoints=np.sum(np.isreal(self.Intensity))
        self.SymCoordAssign_SingleMaterial()
        self.SimTrap_SM()
        self.SimInt_Initial=self.SimInt
        self.GF = self.GF_calc(self.SimInt)
        self.GF_Initial=self.GF
        self.BIC= self.BIC_calc(self.GF)
        self.GF_Initial=self.BIC


    def importCDSAXSQrQz(self,Intensitydata,Qrdata,Qzdata):
        # imports data from a 1D grating
            self.Intensity = np.loadtxt(Intensitydata)
            self.Qr=np.loadtxt(Qrdata)
            self.Qz=np.loadtxt(Qzdata)
        
            self.Intensity[self.Intensity == 0]=np.nan # replaces 
            self.Qr[self.Qx == 0]=np.nan
            self.Qz[self.Qz == 0]=np.nan
            self.numberpoints=np.sum(np.isreal(self.Intensity))

# ### Fourier Transforms
    def FreeFormTrapezoid(self):
        H1 = self.Coord[0,3]
        H2 = self.Coord[0,3]
        self.form=np.zeros([len(self.Qx[:,1]),len(self.Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
        for i in range(int(self.layers)): # edit this to remove the need for the trapnumber variable
            H2 = H2+self.Coord[i,2]
            if i > 0:
                H1 = H1+self.Coord[i-1,2] 
            x1 = self.Coord[i,0]
            x4 = self.Coord[i,1]
            x2 = self.Coord[i+1,0]
            x3 = self.Coord[i+1,1]
            # Avoid division by zero
            x2 = x1 - 1e-6 if np.isclose(x2, x1) else x2
            x4 = x3 - 1e-6 if np.isclose(x4, x3) else x4
            SL = self.Coord[i,2]/(x2-x1)
            SR = -self.Coord[i,2]/(x4-x3)
            
            A1 = (np.exp(1j*self.Qx*((H1-SR*x4)/SR))/(self.Qx/SR+self.Qz))*(np.exp(-1j*H2*(self.Qx/SR+self.Qz))-np.exp(-1j*H1*(self.Qx/SR+self.Qz)))
            A2 = (np.exp(1j*self.Qx*((H1-SL*x1)/SL))/(self.Qx/SL+self.Qz))*(np.exp(-1j*H2*(self.Qx/SL+self.Qz))-np.exp(-1j*H1*(self.Qx/SL+self.Qz)))
            self.form=self.form+(1j/self.Qx)*(A1-A2)*self.Coord[i,4]
        
    
    def FreeFormTrapezoidOpt(self,Coord,layers,Qx,Qz):
        # this version exists to accomate the form required by the gen algorithm, consider recombining and simplifying if possible
        H1 = Coord[0,3]
        H2 = Coord[0,3]
        form=np.zeros([len(Qx[:,1]),len(Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
        for i in range(int(layers)): 
            H2 = H2+Coord[i,2]
            if i > 0:
                H1 = H1+Coord[i-1,2] 
            x1 = Coord[i,0]
            x4 = Coord[i,1]
            x2 = Coord[i+1,0]
            x3 = Coord[i+1,1]
             # Avoid division by zero
            x2 = x1 - 1e-6 if np.isclose(x2, x1) else x2
            x4 = x3 - 1e-6 if np.isclose(x4, x3) else x4
            
            SL = Coord[i,2]/(x2-x1)
            SR = -Coord[i,2]/(x4-x3)
            
            A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
            A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
            form=form+(1j/Qx)*(A1-A2)*Coord[i,4]
        return form
    
    def GF_calc(self,SimInt):
        GF_M= abs(np.log(self.Intensity)-np.log(SimInt))
        
        GF_M[np.isnan(GF_M)]=0
        GF=np.sum(GF_M)
        return (GF)
            
    def BIC_calc(self, GF):
        k = 2*self.layers+2 # number of fitting parameters
        BIC=(self.numberpoints-k)*GF/self.numberpoints+k*math.log(self.numberpoints)
        return BIC
    
    
    def ConeFourierTransform(self,Discretization):
        # Fourier transform for a cone in cylindrical coordinates (Qr,Qz) 
        H1 = 0
        H2 = 0
        self.Form=np.zeros([int(len(self.Qr[:,0])),int(len(self.Qr[0,:]))])
        
        for i in range (self.layers):
            H2=H2+self.PAR[i,1]
            z=np.zeros([int(Discretization[i])])
            stepsize=self.PAR[i,1]/Discretization[i]
            z=np.arange(H1,H2+0.01,stepsize)
            if i > 0 :
                H1=H1+self.PAR[i-1,1]
                
            z=np.arange(H1,H2+0.01,stepsize)
            R1=self.PAR[i,0]
            R2=self.PAR[i+1,0]
            if R1==R2:
                R1=R1+0.000001
            Slope=(H2-H1)/(R2-R1)
            for ii in range(len(z)-1):
                RI1=(z[ii]-H1)/Slope+R1
                RI2=(z[ii+1]-H1)/Slope+R1
                fa=2*np.pi*RI1/self.Qr*sp.jv(1,self.Qr*RI1)*np.exp(1j*self.Qz*z[ii])
                fb=2*np.pi*RI2/self.Qr*sp.jv(1,self.Qr*RI2)*np.exp(1j*self.Qz*z[ii+1])
                self.Form=self.Form+stepsize*(fb+fa)/2 # if you had an SLD variation you would multiply by the SLD here
        return self.Form
    
    def ConeFourierTransformOpt(self,PAR,layers, Qz, Qr,Discretization):
        # Fourier transform for a cone in cylindrical coordinates (Qr,Qz) 
        H1 = 0
        H2 = 0
        Form=np.zeros([int(len(Qr[:,0])),int(len(Qr[0,:]))])
        
        for i in range (layers):
            H2=H2+PAR[i,1]
            z=np.zeros([int(Discretization[i])])
            stepsize=PAR[i,1]/Discretization[i]
            z=np.arange(H1,H2+0.01,stepsize)
            if i > 0 :
                H1=H1+PAR[i-1,1]
                
            z=np.arange(H1,H2+0.01,stepsize)
            R1=PAR[i,0]
            R2=PAR[i+1,0]
            if R1==R2:
                R1=R1+0.000001
            Slope=(H2-H1)/(R2-R1)
            for ii in range(len(z)-1):
                RI1=(z[ii]-H1)/Slope+R1
                RI2=(z[ii+1]-H1)/Slope+R1
                fa=2*np.pi*RI1/Qr*sp.jv(1,Qr*RI1)*np.exp(1j*Qz*z[ii])
                fb=2*np.pi*RI2/Qr*sp.jv(1,Qr*RI2)*np.exp(1j*Qz*z[ii+1])
                Form=Form+stepsize*(fb+fa)/2 # if you had an SLD variation you would multiply by the SLD here
        return Form
    
    
### Coordinate Assignment Code
    def SymCoordAssign_SingleMaterial(self):
    # assigns trapezoid coordinates for a symmetric trapezoid
    # consider combining with SymCoordAssign with SLD as a flag

        self.Coord=np.zeros([self.layers+1,5,1])
        for T in range (self.layers+1):
            if T==0:
                self.Coord[T,0,0]=0
                self.Coord[T,1,0]=self.PAR[0,0]
                self.Coord[T,2,0]=self.PAR[0,1]
                self.Coord[T,3,0]=0
                self.Coord[T,4,0]=1 # SLD - assigned to be 1 for a single material
            else:
                self.Coord[T,0,0]=self.Coord[T-1,0,0]+0.5*(self.PAR[T-1,0]-self.PAR[T,0])
                self.Coord[T,1,0]=self.Coord[T,0,0]+self.PAR[T,0]
                self.Coord[T,2,0]=self.PAR[T,1]
                self.Coord[T,3,0]=0
                self.Coord[T,4,0]=1# SLD - assigned to be 1 for a single material


    def SymCoordAssign_SingleMaterial(self):
        # assigns trapezoid coordinates for a symmetric trapezoid
        # consider combining with SymCoordAssign with SLD as a flag

            self.Coord=np.zeros([self.layers+1,5,1])
            for T in range (self.layers+1):
                if T==0:
                    self.Coord[T,0,0]=0
                    self.Coord[T,1,0]=self.PAR[0,0]
                    self.Coord[T,2,0]=self.PAR[0,1]
                    self.Coord[T,3,0]=0
                    self.Coord[T,4,0]=1 # SLD - assigned to be 1 for a single material
                else:
                    self.Coord[T,0,0]=self.Coord[T-1,0,0]+0.5*(self.PAR[T-1,0]-self.PAR[T,0])
                    self.Coord[T,1,0]=self.Coord[T,0,0]+self.PAR[T,0]
                    self.Coord[T,2,0]=self.PAR[T,1]
                    self.Coord[T,3,0]=0
                    self.Coord[T,4,0]=1# SLD - assigned to be 1 for a single material
        
    def SymCoordAssign_SingleMaterialOpt(self,PAR,layers):
        # assigns trapezoid coordinates for a symmetric trapezoid
        # consider combining with SymCoordAssign with SLD as a flag
        #should be able to combine this with the previous funciton
            Coord=np.zeros([layers+1,5,1])
            for T in range (layers+1):
                if T==0:
                    Coord[T,0,0]=0
                    Coord[T,1,0]=PAR[0,0]
                    Coord[T,2,0]=PAR[0,1]
                    Coord[T,3,0]=0
                    Coord[T,4,0]=1 # SLD - assigned to be 1 for a single material
                else:
                    Coord[T,0,0]=Coord[T-1,0,0]+0.5*(PAR[T-1,0]-PAR[T,0])
                    Coord[T,1,0]=Coord[T,0,0]+PAR[T,0]
                    Coord[T,2,0]=PAR[T,1]
                    Coord[T,3,0]=0
                    Coord[T,4,0]=1# SLD - assigned to be 1 for a single material
            return Coord

    
    ### Simulations
    def SimTrap_SM(self):
        
        self.SymCoordAssign_SingleMaterial()
        self.FreeFormTrapezoid() 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimInt = np.power(Formfactor,2)*self.I0+self.Bk
        return self.SimInt
    
    def SimTrap_SMOpt(self,SimPar,layers,Qx,Qz):
        
        Coord=self.SymCoordAssign_SingleMaterialOpt(SimPar,layers)
        print('Used Coordinate ', Coord)
        form=self.FreeFormTrapezoidOpt(Coord,layers,Qx,Qz) 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW_Optimized,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimIntOpt = np.power(Formfactor,2)*self.I0_Optimized+self.Bk_Optimized
        return self.SimIntOpt
    
    def SimTrap_SMFinal(self,layers,Qx,Qz):
    
        form=self.FreeFormTrapezoidOpt(self.Coord_Optimized,layers,Qx,Qz) 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW_Optimized,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimIntOpt = np.power(Formfactor,2)*self.I0_Optimized+self.Bk_Optimized
        return self.SimIntOpt
    
    def SimCyl_SM(self, Discretization):
        
        
        self.ConeFourierTransform() 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimInt = np.power(Formfactor,2)*self.I0+self.Bk
        return self.SimInt
    
    
    
    ### optimization code
    
    def GenBounds(self,limit):
        self.bounds=[]
        lower_bounds=self.SimPar*(1-limit)
        upper_bounds=self.SimPar*(1+limit)
        self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
        
    def SimTrap_GF(self, SimPar, layers, Intensity, Qx, Qz):
        PARs=np.zeros([layers+1,2])
        PARs[:,0:2]=np.reshape(SimPar[0:(layers+1)*2],(layers+1,2))
        [I0,DW,Bk]=SimPar[layers*2+2:layers*2+5]
        (Coord)=self.SymCoordAssign_SingleMaterialOpt(PARs,layers)
        F1 = self.FreeFormTrapezoidOpt(Coord[:,:,0],layers,Qx,Qz) 
        M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(DW,2)),0.5)
        Formfactor=F1*M
        Formfactor=abs(Formfactor)
        SimInt = np.power(Formfactor,2)*I0+Bk
        Chi2= abs(np.log(Intensity)-np.log(SimInt))
        Chi2[np.isnan(Chi2)]=0
        Chi2=np.sum(Chi2)
        return Chi2
    
    def SimCyl_GF(self, SimPar, layers, Intensity, Qx, Qz, Discretization):
        PARs=np.zeros([layers+1,2])
        PARs[:,0:2]=np.reshape(SimPar[0:(layers+1)*2],(layers+1,2))
        [I0,DW,Bk]=SimPar[layers*2+2:layers*2+5]
        ConeFourierTransformOpt(self,PAR,layers, Qz, Qr,Discretization)
        F1 = self.FreeFormTrapezoidOpt(Coord[:,:,0],layers,Qx,Qz) 
        M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(DW,2)),0.5)
        Formfactor=F1*M
        Formfactor=abs(Formfactor)
        SimInt = np.power(Formfactor,2)*I0+Bk
        Chi2= abs(np.log(Intensity)-np.log(SimInt))
        Chi2[np.isnan(Chi2)]=0
        Chi2=np.sum(Chi2)
        return Chi2
   
    def CDSAXS_DiffEvolution(self,limit):
        
        self.GenBounds(limit)
        
        self.SimPar_Optimized = differential_evolution(self.SimTrap_GF,self.bounds, args=(self.layers,self.Intensity,self.Qx,self.Qz),polish=True)
        
        self.PAR=np.reshape(self.SimPar_Optimized.x[0:(self.layers+1)*2],(self.layers+1,2))

        [self.I0,self.DW,self.Bk]= self.SimPar_Optimized.x[self.layers*2+2:self.layers*2+5]
               
        self.SymCoordAssign_SingleMaterial()
        self.SimTrap_SM()
        self.GF = self.GF_calc(self.SimInt)
        self.BIC= self.BIC_calc(self.GF)
        #self.PlotQzCutComp(10,'yes')
        print('Initial ', self.GF_Initial, ' Final ', self.GF) 
        return (self.PAR,self.I0,self.DW,self.Bk)
        
        
    ### plotting code
    
    
    
    def plotSymTrap(self):
        # Check if self.Coord is initialized properly
        if not hasattr(self, 'Coord') or self.Coord is None:
            print("Error: Coordinates not assigned. Call SymCoordAssign_SingleMaterial first.")
            return
            
        # Create a copy of coordinates for plotting
        Coordp = np.zeros([self.layers+1, 5, 2])
        for i in range(self.layers+1):
            for j in range(5):
                Coordp[i, j, 0] = self.Coord[i, j, 0]
                Coordp[i, j, 1] = self.Coord[i, j, 0]
        
        # Add pitch to specific coordinates
        Coordp[:, 0:1, 1] = Coordp[:, 0:1, 1] + self.Pitch
        
        for S in range(1):
            h = 0
            Lc = np.zeros([self.layers+1, 2])
            Rc = np.zeros([self.layers+1, 2])
            
            for i in range(self.layers+1):
                Lc[i, 0] = Coordp[i, 0, S]
                Rc[i, 0] = Coordp[i, 1, S]
                Lc[i, 1] = h
                Rc[i, 1] = h
                h = h + Coordp[i, 2, S]
                
            plt.plot(Lc[:, 0], Lc[:, 1], color='black')
            plt.plot(Rc[:, 0], Rc[:, 1], color='black')
            
            Cc = np.zeros([2, 2])
            for i in range(self.layers):
                Cc[0, 0] = Lc[i+1, 0]
                Cc[0, 1] = Lc[i+1, 1]
                Cc[1, 0] = Rc[i+1, 0]
                Cc[1, 1] = Rc[i+1, 1]
                plt.plot(Cc[:, 0], Cc[:, 1], color='black')
                
        plt.xlabel('Width (Å)')
        plt.ylabel('Height (Å)')
        plt.show()
        plt.close()
        
    def PlotQzCut(self,numbercuts,SP,scale):

        S=deepcopy(SP)
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
        del I
        plt.show()
        plt.close()
        
   
    
    def PlotQzCutComp(self,numbercuts,scale):
        S_Init=deepcopy(self.SimInt_Initial)
        S_Opt =deepcopy(self.SimInt)
        I=deepcopy(self.Intensity)   
        if scale =='yes': 
            for i in range(0,numbercuts):
                S_Init[:,i]=S_Init[:,i]/(50.**(i+1))
                S_Opt[:,i]=S_Opt[:,i]/(50.**(i+1))
                I[:,i]=I[:,i]/(50.**(i+1))
        for i in range(numbercuts):
            plt.semilogy(self.Qz[:,i],I[:,i],'.', label='Exp '+str(i))
            plt.semilogy(self.Qz[:,i],S_Init[:,i], label='Sim '+str(i), color='black', linestyle='--')
            plt.semilogy(self.Qz[:,i],S_Opt[:,i], label='Sim '+str(i), color='black')
        #plt.legend(loc='upper right')
        del I
        plt.xlabel('q ($Å^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()
        plt.close()
        

    def combined_plots(self, numbercuts=None, SP=None, scale=None): # Currently creates two identical plots?
        """
        Creates a single figure with two subplots side by side:
        1. Symmetric trapezoid structure (plotSymTrap)
        2. Qz cuts (PlotQzCut)
        
        Parameters:
        - numbercuts: Number of cuts for PlotQzCut
        - SP: Data for PlotQzCut
        - scale: Scaling option for PlotQzCut ('yes' or 'no')
        """
        # Create figure with two subplots side by side
        fig, ax1 = plt.subplots(1, 2, figsize=(12, 5))
        
        # First subplot: plotSymTrap
        # Check if self.Coord is initialized properly
        if not hasattr(self, 'Coord') or self.Coord is None:
            print("Error: Coordinates not assigned. Call SymCoordAssign_SingleMaterial first.")
            return
            
        # Create a copy of coordinates for plotting
        Coordp = np.zeros([self.layers+1, 5, 2])
        for i in range(self.layers+1):
            for j in range(5):
                Coordp[i, j, 0] = self.Coord[i, j, 0]
                Coordp[i, j, 1] = self.Coord[i, j, 0]
        
        # Add pitch to specific coordinates
        Coordp[:, 0:1, 1] = Coordp[:, 0:1, 1] + self.Pitch
        
        for S in range(1):
            h = 0
            Lc = np.zeros([self.layers+1, 2])
            Rc = np.zeros([self.layers+1, 2])
            
            for i in range(self.layers+1):
                Lc[i, 0] = Coordp[i, 0, S]
                Rc[i, 0] = Coordp[i, 1, S]
                Lc[i, 1] = h
                Rc[i, 1] = h
                h = h + Coordp[i, 2, S]
                
            ax1[0].plot(Lc[:, 0], Lc[:, 1], color='black')
            ax1[0].plot(Rc[:, 0], Rc[:, 1], color='black')
            
            Cc = np.zeros([2, 2])
            for i in range(self.layers):
                Cc[0, 0] = Lc[i+1, 0]
                Cc[0, 1] = Lc[i+1, 1]
                Cc[1, 0] = Rc[i+1, 0]
                Cc[1, 1] = Rc[i+1, 1]
                ax1[0].plot(Cc[:, 0], Cc[:, 1], color='black')
                
        ax1[0].set_xlabel('Width (Å)')
        ax1[0].set_ylabel('Height (Å)')
        ax1[0].set_title('Symmetric Trapezoid Structure')
        
        # # Second subplot: PlotQzCut
        # if SP is not None and numbercuts is not None:
        #     S = deepcopy(SP)
        #     I = deepcopy(self.Intensity)   
        #     if scale == 'yes': 
        #         for i in range(0, numbercuts):
        #             S[:, i] = S[:, i]/(50.**(i+1))
        #             I[:, i] = I[:, i]/(50.**(i+1))
        #     for i in range(numbercuts):
        #         ax2.semilogy(self.Qz[:, i], I[:, i], '.')
        #         ax2.semilogy(self.Qz[:, i], S[:, i], color='black')
        #     ax2.set_xlabel('q ($Å^{-1}$)')
        #     ax2.set_ylabel('Intensity (a.u.)')
        #     ax2.set_title('Qz Cuts')
        #     #ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return fig

##this class should simplify logging and comparing results
class CDSAXS_fitter():
    def __init__(self,attribute_name, value):
        setattr(self,attribute_name,value)
        self.fitlog = pd.DataFrame(columns=["Model",'layers', "GF_Optimized", "BIC_Optimized"])
        print(value.GF)
        new_row = {'Model': attribute_name, 'layers': value.layers, 'GF_Optimized': value.GF, 'BIC_Optimized':value.BIC}
        self.fitlog.loc[len(self.fitlog)] = new_row
        print(self.fitlog)
    def add_model(self,attribute_name, value):
        setattr(self,attribute_name,value)
        new_row = {'Model': attribute_name, 'layers': value.layers, 'GF': value.GF, 'BIC':value.BIC}
        self.fitlog.loc[len(self.fitlog)] = new_row
    
    def selectmodel(self,modelname):
        modelselect= getattr(self,modelname)
        return modelselect
    
    def PrintCoord(self,model):
        modelselect=self.selectmodel(model)
        print(modelselect.Coord)
    
    def selectattribute(self,modelselect,attribute_name):
        attribute=getattr(modelselect,attribute_name)
        return attribute
    
    
    def plot_Trap_InitOpt(self,Model):
        modelselect=self.selectmodel(Model)
        Coord=(modelselect.Coord,modelselect.Coord_Optimized)
        colorlist=('r','b','g')
        for k,v in enumerate(Coord):
            Coordp=np.zeros([modelselect.layers+1,5,2])
            Coordp[:,:,0]=v[:,:,0]
            Coordp[:,:,1]=v[:,:,0]
            Coordp[:,0:1,1]=Coordp[:,0:1,1]+modelselect.Pitch
            for S in range(1):
                h=0
                Lc= np.zeros([modelselect.layers+1,2])
                Rc= np.zeros([modelselect.layers+1,2])

                for i in range(modelselect.layers+1):
                    Lc[i,0]=Coordp[i,0,S]
                    Rc[i,0]=Coordp[i,1,S]
                    Lc[i,1]=h
                    Rc[i,1]=h
                    h=h+Coordp[i,2,S]
                plt.plot(Lc[:,0],Lc[:,1], color=colorlist[k])
                plt.plot(Rc[:,0],Rc[:,1], color=colorlist[k])
                Cc=np.zeros([2,2])
                for i in range(modelselect.layers):
                    Cc[0,0]=Lc[i+1,0]
                    Cc[0,1]=Lc[i+1,1]
                    Cc[1,0]=Rc[i+1,0]
                    Cc[1,1]=Rc[i+1,1]
                    if i ==0 and k==0:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],label='Initial')
                    elif i==0 and k==1:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],label='Optimized')
                    else:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],)
        #plt.title(SampleName)
        plt.legend(loc='upper right')
        #plt.xlim([0,600])
        plt.xlabel('Width (Å)')
        plt.ylabel('Height (Å)')
        plt.legend(loc='lower center')
        plt.show()
        
    def plot_Trap_ModelComp(self,Model1,Model2):
        modelselect1=self.selectmodel(Model1)
        modelselect2=self.selectmodel(Model2)
        Coord=(modelselect1.Coord_Optimized,modelselect2.Coord_Optimized)
        colorlist=('r','b','g')
        for k,v in enumerate(Coord):
            Coordp=np.zeros([modelselect.layers+1,5,2])
            Coordp[:,:,0]=v[:,:,0]
            Coordp[:,:,1]=v[:,:,0]
            Coordp[:,0:1,1]=Coordp[:,0:1,1]+modelselect.Pitch
            for S in range(1):
                h=0
                Lc= np.zeros([modelselect.layers+1,2])
                Rc= np.zeros([modelselect.layers+1,2])

                for i in range(modelselect.layers+1):
                    Lc[i,0]=Coordp[i,0,S]
                    Rc[i,0]=Coordp[i,1,S]
                    Lc[i,1]=h
                    Rc[i,1]=h
                    h=h+Coordp[i,2,S]
                plt.plot(Lc[:,0],Lc[:,1], color=colorlist[k])
                plt.plot(Rc[:,0],Rc[:,1], color=colorlist[k])
                Cc=np.zeros([2,2])
                for i in range(modelselect.layers):
                    Cc[0,0]=Lc[i+1,0]
                    Cc[0,1]=Lc[i+1,1]
                    Cc[1,0]=Rc[i+1,0]
                    Cc[1,1]=Rc[i+1,1]
                    if i ==0 and k==0:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],label='Initial')
                    elif i==0 and k==1:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],label='Optimized')
                    else:
                        plt.plot(Cc[:,0],Cc[:,1],color=colorlist[k],)
        #plt.title(SampleName)
        plt.legend(loc='upper right')
        #plt.xlim([0,600])
        plt.xlabel('Width (Å)')
        plt.ylabel('Height (Å)')
        plt.legend(loc='lower center')
        plt.show()