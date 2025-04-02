# Attempt to rewrite the CDSAXS functions in a class structure
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.special as sp
import matplotlib.patches as mpatches
from scipy.optimize import differential_evolution
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
        self.layers=layers
        self.SLD=SLD
        self.DW=DW
        self.I0=I0
        self.Bk=Bk
        self.Pitch=Pitch
        self.PAR_Optimized=PAR
        self.SLD_Optimized=SLD
        self.DW_Optimized=DW
        self.I0_Optimized=I0
        self.Bk_Optimized=Bk
        self.Coord=[]
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

    def importCDSAXSQrQz(self,Intensitydata,Qrdata,Qzdata):
        # imports data from a 1D grating
            self.Intensity = np.loadtxt(Intensitydata)
            self.Qr=np.loadtxt(Qrdata)
            self.Qz=np.loadtxt(Qzdata)
        
            self.Intensity[self.Intensity == 0]=np.nan # replaces 
            self.Qr[self.Qx == 0]=np.nan
            self.Qz[self.Qz == 0]=np.nan
            self.numberpoints=np.sum(np.isreal(self.Intensity))

### Fourier Transforms
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
        


    def FreeFormTrapezoidOpt(self,Coord,layers,Qx,Qz):
        # this version exists to accomate the form required by the gen algorithm, consider recombining and simplifying if possible
        H1 = Coord[0,3]
        H2 = Coord[0,3]
        form=np.zeros([len(Qx[:,1]),len(Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
        for i in range(int(layers)): # edit this to remove the need for the trapnumber variable
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
    
    ### optimization code
    
    def GenBounds(self,limit):
        self.bounds=[]
        lower_bounds=self.SimPar*(1-limit)
        upper_bounds=self.SimPar*(1+limit)
        self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
        
    def SimGF(self, SimPar, layers, Intensity, Qx, Qz):
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
   
    def CDSAXS_DiffEvolution(self,limit):
        self.GenBounds(limit)
        self.result = differential_evolution(self.SimGF,self.bounds, args=(self.layers,self.Intensity,self.Qx,self.Qz),polish=True)
    
    
    ### plotting code
    
    def plotSymTrap(self):
        Coordp=np.zeros([self.layers+1,5,2])
        Coordp[:,:,0]=self.Coord[:,:,0]
        Coordp[:,:,1]=self.Coord[:,:,0]
        Coordp[:,0:1,1]=Coordp[:,0:1,1]+self.Pitch
        for S in range(1):
            h=0
            Lc= np.zeros([self.layers+1,2])
            Rc= np.zeros([self.layers+1,2])
            
            for i in range(self.layers+1):
                Lc[i,0]=Coordp[i,0,S]
                Rc[i,0]=Coordp[i,1,S]
                Lc[i,1]=h
                Rc[i,1]=h
                h=h+Coordp[i,2,S]
            plt.plot(Lc[:,0],Lc[:,1], color='black')
            plt.plot(Rc[:,0],Rc[:,1], color='black')
            Cc=np.zeros([2,2])
            for i in range(self.layers):
                Cc[0,0]=Lc[i+1,0]
                Cc[0,1]=Lc[i+1,1]
                Cc[1,0]=Rc[i+1,0]
                Cc[1,1]=Rc[i+1,1]
                plt.plot(Cc[:,0],Cc[:,1], color='black')
        plt.xlabel('Width (Å)')
        plt.ylabel('Height (Å)')             

        plt.show()
        plt.close()
        
    
   
    
    
    
        
    
    
        
    
        

        
    # def Optimize_CDSAXS(self):
        
    #     def Optimize_CDSAXS(par,Trapnumber,Intensity,Qx,Qz):
    # Sim=SimTrap(par,Trapnumber)

    # ChiPost=np.sum(CD.Misfit(Intensity,Sim))
    # return ChiPost
        
    #     bounds1=GenBounds(TPAR_1T,SPAR_1T,0.35)
    #     result1 = differential_evolution(Optimize_CDSAXS, bounds1, args=(Trapnumber,Intensity,Qx,Qz),polish=True)
    #     print(result1)
    #     PlotQzCutComp(Qz,(FITPAR_1T,result1.x),Trapnumber,Intensity,14)
    #     SimPost_1T=SimTrap(result1.x,Trapnumber)
    #     ChiPost_1T=np.sum(CD.Misfit(Intensity,SimPost_1T))
    #     plt.show()

    #     TPARs_1T=np.zeros([Trapnumber+1,2])
    #     TPARs_1T[:,0:2]=np.reshape(result1.x[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    #     Coords_1T=SymCoordAssign_SingleMaterial(TPARs_1T)
    #     plotMultiTrap((Coord_1T,Coords_1T),Trapnumber,Pitch,('Initial','Final'))
    #     print(ChiPost_1T)
#     def PlotQzCut(self,numbercuts,scale):
#         S=self.SimInt
#         I=deepcopy(self.Intensity)   
#         if scale =='yes': 
#             for i in range(0,numbercuts):
#                 S[:,i]=S[:,i]/(50.**(i+1))
#                 I[:,i]=I[:,i]/(50.**(i+1))
#         for i in range(numbercuts):
#             plt.semilogy(self.Qz[:,i],I[:,i],'.', label='Exp '+str(i))
#             plt.semilogy(self.Qz[:,i],S[:,i], label='Sim '+str(i), color='black')
#         #plt.legend(loc='upper right')
#         plt.xlabel('q ($Å^{-1}$)')
#         plt.ylabel('Intensity (a.u.)')
#         plt.plot()
        
#     def Misfit(self):
#         self.Chi2= abs(np.log(self.Intensity)-np.log(self.SimInt))
        
#         self.Chi2[np.isnan(self.Chi2)]=0
    
    
#     def InitializeModel(self,geometry,model,layers,TPAR, SLD, I0, DW, Bk, Pitch):
#         self.layerlog=np.concatenate((self.layerlog,layers))
#         self.modellog_initial=np.concatenate((np.flatten(TPAR),I0,DW,Bk))
#         self.
#         Model_map={'Symmetric':SymCoordAssign_SingleMaterial}
#         self.Model_map
    
#     def SimandPlot(self,geometry,model,layers):
#         geometry_sel={'trapezoid':}
   