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

import os
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
        self.geometry=geometry
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
    
    def importCDSAXSQxQz(self, Intensitydata, Qxdata, Qzdata):
        """
        Imports data from a 1D grating with input validation
        
        Parameters:
        -----------
        Intensitydata : str
            Path to intensity data file
        Qxdata : str
            Path to Qx data file
        Qzdata : str
            Path to Qz data file
        """
        # Check if input variables exist
        if Intensitydata is None or not isinstance(Intensitydata, str):
            raise ValueError("Intensitydata must be a valid file path")
        if Qxdata is None or not isinstance(Qxdata, str):
            raise ValueError("Qxdata must be a valid file path")
        if Qzdata is None or not isinstance(Qzdata, str):
            raise ValueError("Qzdata must be a valid file path")
        
        # Check if files exist
        for filepath in [Intensitydata, Qxdata, Qzdata]:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Load data from files
            self.Intensity = np.loadtxt(Intensitydata)
            self.Qx = np.loadtxt(Qxdata)
            self.Qz = np.loadtxt(Qzdata)
            
            # Replace zeros with NaN
            self.Intensity[self.Intensity == 0] = np.nan
            self.Qx[self.Qx == 0] = np.nan
            self.Qz[self.Qz == 0] = np.nan
            
            # Calculate number of valid points
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            # Execute trapezoid-specific code if that geometry is set
            if hasattr(self, 'geometry') and self.geometry == 'trapezoid':
                self.SymCoordAssign_SingleMaterial()
                self.SimTrap_SM()
                self.SimInt_Initial = self.SimInt
                self.GF = self.GF_calc(self.SimInt)
                self.GF_Initial = self.GF
                self.BIC = self.BIC_calc(self.GF)
                self.GF_Initial = self.BIC
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")
    
       
    # def importCDSAXSQxQz(self,Intensitydata,Qxdata,Qzdata):
    #     # imports data from a 1D grating
    #         self.Intensity = np.loadtxt(Intensitydata)
    #         self.Qx=np.loadtxt(Qxdata)
    #         self.Qz=np.loadtxt(Qzdata)
        
    #         self.Intensity[self.Intensity == 0]=np.nan # replaces 
    #         self.Qx[self.Qx == 0]=np.nan
    #         self.Qz[self.Qz == 0]=np.nan
    #         self.numberpoints=np.sum(np.isreal(self.Intensity))
    #         if self.geometry == 'trapezoid':
    #             self.SymCoordAssign_SingleMaterial()
    #             self.SimTrap_SM()
    #             self.SimInt_Initial=self.SimInt
    #             self.GF = self.GF_calc(self.SimInt)
    #             self.GF_Initial=self.GF
    #             self.BIC= self.BIC_calc(self.GF)
    #             self.GF_Initial=self.BIC
   
   
   
    def importCDSAXS_GUI(self, Datafile):
        """
        Imports CDSAXS data from a GUI-created file with input validation
        
        Parameters:
        -----------
        Datafile : str
            Path to the data file (CSV format)
        """
        # Check if input variable exists and is valid
        if Datafile is None or not isinstance(Datafile, str):
            raise ValueError("Datafile must be a valid file path")
        
        # Check if file exists
        if not os.path.isfile(Datafile):
            raise FileNotFoundError(f"File not found: {Datafile}")
        
        try:
            # Import data using pandas
            Data = pd.read_csv(Datafile)
            
            # Check if file has content
            if Data.empty:
                raise ValueError("The data file is empty")
            
            # Check the number of cuts
            num_columns = len(Data.columns)
            if num_columns < 2:
                raise ValueError("Data must have at least 2 columns")
                
            numbercuts = num_columns // 2
            headers = Data.columns.tolist()
            
            # Extract qx values from headers
            qxlist = []
            for i in range(1, len(headers), 2):
                # Look for pattern 'qx = number' in the header
                match = re.search(r'qx\s*=\s*(\d+\.?\d*)', headers[i])
                if match:
                    number = float(match.group(1))
                    # Convert to int if it's a whole number
                    if number.is_integer():
                        number = int(number)
                    qxlist.append(number)
            
            # Check if we found any qx values
            if not qxlist:
                raise ValueError("No qx values found in headers")
                
            # Convert to numpy array
            Data1 = Data.to_numpy()
            
            # Initialize arrays
            data_rows = len(Data1[:,0])
            self.Intensity = np.zeros([data_rows, numbercuts])
            self.Qz = np.zeros([data_rows, numbercuts])
            
            # Fill arrays with data
            for i in range(0, numbercuts):
                if (i*2+1) < num_columns:  # Check if column exists
                    self.Intensity[:,i] = Data1[:,(i*2+1)]
                    self.Qz[:,i] = Data1[:,(i*2)]
            
            # Create Qx array
            self.Qx = self.Qz.copy()
            self.Qx[~np.isnan(self.Qx)] = 1
            
            # Apply qx values to each column
            for k, v in enumerate(qxlist):
                if k < self.Qx.shape[1]:  # Check if column exists
                    self.Qx[:,k] = self.Qx[:,k] * v
            
            # Calculate number of valid points
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            # Execute trapezoid-specific code if that geometry is set
            if hasattr(self, 'geometry') and self.geometry == 'trapezoid':
                self.SymCoordAssign_SingleMaterial()
                self.SimTrap_SM()
                self.SimInt_Initial = self.SimInt
                self.GF = self.GF_calc(self.SimInt)
                self.GF_Initial = self.GF
                self.BIC = self.BIC_calc(self.GF)
                self.GF_Initial = self.BIC
                
        except pd.errors.EmptyDataError:
            raise ValueError("The data file is empty or not properly formatted")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV file. Check the file format")
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")
    
    def convert_Cartesian_Cylindrical(self):
        """
        Converts Cartesian coordinates (Qx, Qy) to Cylindrical coordinates (Qr, Alpha)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qy'):
                raise AttributeError("Missing required attributes: Qx and Qy must be defined")
            
            # Check if arrays are properly initialized and have compatible shapes
            if not isinstance(self.Qx, np.ndarray) or not isinstance(self.Qy, np.ndarray):
                raise TypeError("Qx and Qy must be numpy arrays")
                
            if self.Qx.shape != self.Qy.shape:
                raise ValueError(f"Shape mismatch: Qx shape {self.Qx.shape} different from Qy shape {self.Qy.shape}")
                
            # Check for NaN or empty arrays
            if np.all(np.isnan(self.Qx)) or np.all(np.isnan(self.Qy)):
                raise ValueError("Input arrays contain only NaN values")
                
            # Handle division by zero (when Qx = 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                Alpha = np.arctan(self.Qy / self.Qx)
                # Replace NaN resulting from 0/0 with 0 or appropriate value
                Alpha = np.nan_to_num(Alpha, nan=0.0)
            
            # Compute Qr 
            self.Qr = self.Qx/np.cos(Alpha)
            
            # Store Alpha for future use
            self.Alpha = Alpha
            
            return True
            
        except Exception as e:
            print(f"Error in convert_Cartesian_Cylindrical: {str(e)}")
            return False
              


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
    # def FreeFormTrapezoid(self):
    #     # this version is validated against old code, keeping it in case the AI messes up when generating the new version
    #     H1 = self.Coord[0,3]
    #     H2 = self.Coord[0,3]
    #     self.form=np.zeros([len(self.Qx[:,1]),len(self.Qx[1,:])]) # initialize structure of the amplitude - (labeled form here)
    #     for i in range(int(self.layers)): # edit this to remove the need for the trapnumber variable
    #         H2 = H2+self.Coord[i,2]
    #         if i > 0:
    #             H1 = H1+self.Coord[i-1,2] 
    #         x1 = self.Coord[i,0]
    #         x4 = self.Coord[i,1]
    #         x2 = self.Coord[i+1,0]
    #         x3 = self.Coord[i+1,1]
    #         # Avoid division by zero
    #         x2 = x1 - 1e-6 if np.isclose(x2, x1) else x2
    #         x4 = x3 - 1e-6 if np.isclose(x4, x3) else x4
    #         SL = self.Coord[i,2]/(x2-x1)
    #         SR = -self.Coord[i,2]/(x4-x3)
            
    #         A1 = (np.exp(1j*self.Qx*((H1-SR*x4)/SR))/(self.Qx/SR+self.Qz))*(np.exp(-1j*H2*(self.Qx/SR+self.Qz))-np.exp(-1j*H1*(self.Qx/SR+self.Qz)))
    #         A2 = (np.exp(1j*self.Qx*((H1-SL*x1)/SL))/(self.Qx/SL+self.Qz))*(np.exp(-1j*H2*(self.Qx/SL+self.Qz))-np.exp(-1j*H1*(self.Qx/SL+self.Qz)))
    #         self.form=self.form+(1j/self.Qx)*(A1-A2)*self.Coord[i,4]
    def FreeFormTrapezoid(self):
        """
        Calculates the form factor for a free-form trapezoid structure.
        
        This function computes the form factor for a complex trapezoid structure defined by
        the self.Coord array, using the Qx and Qz scattering vectors.
        
        Requirements:
            - self.Coord: numpy array with trapezoid coordinates and parameters
            - self.Qx, self.Qz: 2D numpy arrays with scattering vector components
            - self.layers: number of layers in the trapezoid structure
        
        Returns:
            None, but sets self.form attribute with the calculated form factor
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Coord'):
                raise AttributeError("Missing required attribute: Coord")
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                raise AttributeError("Missing required scattering vector attributes: Qx and/or Qz")
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
                
            # Check if arrays have proper dimensions
            if len(self.Qx.shape) != 2:
                raise ValueError(f"Qx must be a 2D array, got shape {self.Qx.shape}")
                
            # Validate Coord shape for indexing
            if int(self.layers) + 1 > len(self.Coord):
                raise IndexError(f"Not enough rows in Coord ({len(self.Coord)}) for {int(self.layers)} layers")
            
            # this version is validated against old code, keeping it in case the AI messes up when generating the new version
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
            
        except Exception as e:
            print(f"Error in FreeFormTrapezoid: {str(e)}")
            # Keep the form attribute as None or zeros in case of error
            self.form = None
    
    
        
    def FreeFormTrapezoidOpt(self, Coord, layers, Qx, Qz):
        """
        Optimized version of FreeFormTrapezoid that takes parameters directly
        instead of using class attributes. Used by genetic algorithm.
        
        Parameters:
        -----------
        Coord : numpy.ndarray
            Coordinate array with shape (n, 5) containing trapezoid parameters:
            - Column 0: x1 coordinate (left bottom)
            - Column 1: x4 coordinate (right bottom)
            - Column 2: height of layer
            - Column 3: initial height (for layer 0)
            - Column 4: electron density or similar parameter
        layers : int
            Number of layers in the trapezoid structure
        Qx : numpy.ndarray
            X-component of scattering vector, 2D array
        Qz : numpy.ndarray
            Z-component of scattering vector, 2D array
            
        Returns:
        --------
        numpy.ndarray
            The calculated form factor for the trapezoid structure
        """
        try:
            # Validate input parameters
            if Coord is None or not isinstance(Coord, np.ndarray):
                raise TypeError("Coord must be a numpy array")
                
            if layers is None or not isinstance(layers, (int, float)) or layers <= 0:
                raise ValueError(f"layers must be a positive number, got {layers}")
                
            if Qx is None or Qz is None:
                raise ValueError("Qx and Qz must not be None")
                
            # Check if we have enough layers in Coord
            if int(layers) + 1 > Coord.shape[0]:
                raise ValueError(f"Not enough rows in Coord ({Coord.shape[0]}) for {int(layers)} layers")
                
            # Check array dimensions
            if len(Qx.shape) != 2 or len(Qz.shape) != 2:
                raise ValueError(f"Qx and Qz must be 2D arrays, got shapes {Qx.shape} and {Qz.shape}")
            
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
            
        except Exception as e:
            print(f"Error in FreeFormTrapezoidOpt: {str(e)}")
            return None
    
    
    
    def GF_calc(self, SimInt):
        """
        Calculates the goodness of fit (GF) metric between experimental and simulated intensities.
        Uses log intensity
        
        This function computes the goodness of fit by summing the absolute difference between 
        the natural logarithm of experimental intensity (self.Intensity) and simulated intensity 
        (SimInt). NaN values in the result are treated as zeros.
        
        Parameters:
        -----------
        SimInt : numpy.ndarray
            Simulated intensity array with the same shape as self.Intensity
            
        Returns:
        --------
        float
            The goodness of fit value; lower values indicate better fit
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Intensity'):
                raise AttributeError("Missing required attribute: Intensity")
                
            # Check if input is valid
            if SimInt is None:
                raise ValueError("SimInt must not be None")
                
            # Check if shapes are compatible
            if hasattr(self, 'Intensity') and self.Intensity.shape != SimInt.shape:
                raise ValueError(f"Shape mismatch: self.Intensity shape {self.Intensity.shape} different from SimInt shape {SimInt.shape}")
            
            # Compute the goodness of fit
            GF_M = abs(np.log(self.Intensity) - np.log(SimInt))
            
            # Replace NaN values with zeros
            GF_M[np.isnan(GF_M)] = 0
            
            # Sum to get the overall goodness of fit
            GF = np.sum(GF_M)
            
            return GF
            
        except Exception as e:
            print(f"Error in GF_calc: {str(e)}")
            return float('inf')  # Return infinity as a worst-case fit value
            
    def BIC_calc(self, GF):
        """
        Calculates the Bayesian Information Criterion (BIC) based on goodness of fit.
        
        BIC is a criterion for model selection that balances the goodness of fit with model complexity.
        It penalizes models with more parameters to prevent overfitting.
        
        Parameters:
        -----------
        GF : float
            Goodness of fit value obtained from GF_calc method
        
        Returns:
        --------
        float
            BIC value; lower values indicate better models considering both fit and complexity
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
            if not hasattr(self, 'numberpoints'):
                raise AttributeError("Missing required attribute: numberpoints")
                
            # Check if input is valid
            if GF is None or not isinstance(GF, (int, float)):
                raise ValueError(f"GF must be a numeric value, got {type(GF)}")
                
            # Check if we have sufficient data points
            if self.numberpoints <= 0:
                raise ValueError(f"Invalid number of data points: {self.numberpoints}")
            
            # Calculate number of fitting parameters
            k = 2 * self.layers + 2  # number of fitting parameters
            
            # Calculate BIC
            BIC = (self.numberpoints - k) * GF / self.numberpoints + k * math.log(self.numberpoints)
            
            return BIC
            
        except Exception as e:
            print(f"Error in BIC_calc: {str(e)}")
            return float('inf')  # Return infinity as a worst-case BIC value
    
    
    def ConeFourierTransform(self,Discretization):
        # Fourier transform for a cone in cylindrical coordinates (Qr,Qz) 
        H1 = 0
        H2 = 0
        self.form=np.zeros([int(len(self.Qr[:,0])),int(len(self.Qr[0,:]))])
        
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
                self.form=self.form+stepsize*(fb+fa)/2 # if you had an SLD variation you would multiply by the SLD here
        return self.form
    
    def ConeFourierTransformOptimized(self, Discretization):
        """
        Optimized version of ConeFourierTransform for faster calculation. !!!! HAVE NOT VALIDATED THIS CODE
        
        Calculates the Fourier transform for a cone in cylindrical coordinates (Qr, Qz)
        with performance optimizations for faster execution.
        
        Parameters:
        -----------
        Discretization : list or numpy.ndarray
            Number of discretization steps for each layer
        
        Returns:
        --------
        numpy.ndarray
            The calculated form factor (self.form)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qr') or not hasattr(self, 'Qz'):
                raise AttributeError("Missing required attributes: Qr and/or Qz")
            if not hasattr(self, 'PAR'):
                raise AttributeError("Missing required attribute: PAR")
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
                
            # Check if arrays have proper dimensions
            if len(self.Qr.shape) != 2 or len(self.Qz.shape) != 2:
                raise ValueError(f"Qr and Qz must be 2D arrays, got shapes {self.Qr.shape} and {self.Qz.shape}")
                
            # Check Discretization input
            if Discretization is None:
                raise ValueError("Discretization must not be None")
            if len(Discretization) < self.layers:
                raise ValueError(f"Discretization array must have at least {self.layers} elements")
                
            # Check PAR dimensions for indexing
            if self.layers + 1 > len(self.PAR):
                raise IndexError(f"Not enough rows in PAR ({len(self.PAR)}) for {self.layers} layers")
            
            # Pre-allocate arrays and initial values
            shape = (int(len(self.Qr[:,0])), int(len(self.Qr[0,:])))
            self.form = np.zeros(shape, dtype=complex)
            H1 = 0
            H2 = 0
            
            # Precompute division by Qr to avoid repeated divisions
            # Add small value to avoid division by zero
            Qr_safe = np.where(np.abs(self.Qr) < 1e-10, 1e-10, self.Qr)
            inv_Qr = 1.0 / Qr_safe
            
            # Loop over layers
            for i in range(self.layers):
                H2 = H2 + self.PAR[i, 1]
                if i > 0:
                    H1 = H1 + self.PAR[i-1, 1]
                    
                # Calculate parameters for this layer
                R1 = self.PAR[i, 0]
                R2 = self.PAR[i+1, 0]
                
                # Avoid exact equality for numerical stability
                if abs(R1 - R2) < 1e-6:
                    R1 = R1 + 1e-6
                    
                Slope = (H2 - H1) / (R2 - R1)
                stepsize = self.PAR[i, 1] / Discretization[i]
                
                # Generate z values for integration once
                z = np.linspace(H1, H2, int(Discretization[i]) + 1)
                
                # Vectorize inner loop calculations
                for ii in range(len(z) - 1):
                    # Calculate radii at current heights
                    RI1 = (z[ii] - H1) / Slope + R1
                    RI2 = (z[ii+1] - H1) / Slope + R1
                    
                    # Compute Bessel functions once for each radius
                    bessel_RI1 = sp.jv(1, self.Qr * RI1)
                    bessel_RI2 = sp.jv(1, self.Qr * RI2)
                    
                    # Compute exponentials once for each z
                    exp_z1 = np.exp(1j * self.Qz * z[ii])
                    exp_z2 = np.exp(1j * self.Qz * z[ii+1])
                    
                    # Combine terms
                    fa = 2 * np.pi * RI1 * inv_Qr * bessel_RI1 * exp_z1
                    fb = 2 * np.pi * RI2 * inv_Qr * bessel_RI2 * exp_z2
                    
                    # Trapezoidal rule integration
                    self.form = self.form + stepsize * (fa + fb) / 2
                    
            return self.form
        
        except Exception as e:
            print(f"Error in ConeFourierTransformOptimized: {str(e)}")
            self.form = None
            return None
    
    def ConeFourierTransformOpt(self, PAR, layers, Qz, Qr, Discretization):
        """
        Optimized Fourier transform for a cone in cylindrical coordinates (Qr, Qz).
        
        Parameters:
        -----------
        PAR : numpy.ndarray
            Parameter array with shape (n, 2) containing radius and height information
        layers : int
            Number of layers in the cone structure
        Qz : numpy.ndarray
            Z-component of scattering vector, 2D array
        Qr : numpy.ndarray
            Radial component of scattering vector, 2D array
        Discretization : list or numpy.ndarray
            Number of discretization steps for each layer
            
        Returns:
        --------
        numpy.ndarray
            The calculated form factor
        """
        try:
            # Validate input parameters
            if PAR is None or not isinstance(PAR, np.ndarray):
                raise TypeError("PAR must be a numpy array")
                
            if layers is None or not isinstance(layers, (int, float)) or layers <= 0:
                raise ValueError(f"layers must be a positive number, got {layers}")
                
            if Qr is None or Qz is None:
                raise ValueError("Qr and Qz must not be None")
                
            if not isinstance(Qr, np.ndarray) or not isinstance(Qz, np.ndarray):
                raise TypeError("Qr and Qz must be numpy arrays")
                
            if len(Qr.shape) != 2 or len(Qz.shape) != 2:
                raise ValueError(f"Qr and Qz must be 2D arrays, got shapes {Qr.shape} and {Qz.shape}")
                
            if Discretization is None or len(Discretization) < layers:
                raise ValueError(f"Discretization array must have at least {layers} elements")
                
            # Check PAR dimensions for indexing
            if layers + 1 > len(PAR):
                raise IndexError(f"Not enough rows in PAR ({len(PAR)}) for {layers} layers")
            
            # Main calculation code - unchanged
            H1 = 0
            H2 = 0
            Form = np.zeros([int(len(Qr[:,0])), int(len(Qr[0,:]))])
            
            for i in range(layers):
                H2 = H2 + PAR[i,1]
                z = np.zeros([int(Discretization[i])])
                stepsize = PAR[i,1] / Discretization[i]
                z = np.arange(H1, H2 + 0.01, stepsize)
                if i > 0:
                    H1 = H1 + PAR[i-1,1]
                    
                z = np.arange(H1, H2 + 0.01, stepsize)
                R1 = PAR[i,0]
                R2 = PAR[i+1,0]
                if R1 == R2:
                    R1 = R1 + 0.000001
                Slope = (H2 - H1) / (R2 - R1)
                for ii in range(len(z) - 1):
                    RI1 = (z[ii] - H1) / Slope + R1
                    RI2 = (z[ii+1] - H1) / Slope + R1
                    fa = 2 * np.pi * RI1 / Qr * sp.jv(1, Qr * RI1) * np.exp(1j * Qz * z[ii])
                    fb = 2 * np.pi * RI2 / Qr * sp.jv(1, Qr * RI2) * np.exp(1j * Qz * z[ii+1])
                    Form = Form + stepsize * (fb + fa) / 2
                    
            return Form
            
        except Exception as e:
            print(f"Error in ConeFourierTransformOpt: {str(e)}")
            return None
    
    
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
        """
        Assigns trapezoid coordinates for a symmetric trapezoid with a single material.
        
        This function generates the coordinate array (self.Coord) for a symmetric trapezoid
        structure based on the parameters in self.PAR. The SLD (Scattering Length Density)
        is set to 1 for all layers, representing a single material.
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'PAR'):
                raise AttributeError("Missing required attribute: PAR")
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
                
            # Validate PAR dimensions
            if not isinstance(self.PAR, np.ndarray):
                raise TypeError("PAR must be a numpy array")
                
            if len(self.PAR) < self.layers + 1:
                raise ValueError(f"PAR array must have at least {self.layers + 1} rows, but has {len(self.PAR)}")
                
            # Check PAR shape
            if len(self.PAR.shape) < 2 or self.PAR.shape[1] < 2:
                raise ValueError(f"PAR must have at least 2 columns, but has shape {self.PAR.shape}")
            
            # Main calculation code - unchanged
            self.Coord = np.zeros([self.layers+1, 5, 1])
            for T in range(self.layers+1):
                if T == 0:
                    self.Coord[T, 0, 0] = 0
                    self.Coord[T, 1, 0] = self.PAR[0, 0]
                    self.Coord[T, 2, 0] = self.PAR[0, 1]
                    self.Coord[T, 3, 0] = 0
                    self.Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
                else:
                    self.Coord[T, 0, 0] = self.Coord[T-1, 0, 0] + 0.5 * (self.PAR[T-1, 0] - self.PAR[T, 0])
                    self.Coord[T, 1, 0] = self.Coord[T, 0, 0] + self.PAR[T, 0]
                    self.Coord[T, 2, 0] = self.PAR[T, 1]
                    self.Coord[T, 3, 0] = 0
                    self.Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
            
            return True
        
        except Exception as e:
            print(f"Error in SymCoordAssign_SingleMaterial: {str(e)}")
            return False
        
    def SymCoordAssign_SingleMaterialOpt(self, PAR, layers):
        """
        Optimized version that assigns trapezoid coordinates for a symmetric trapezoid with a single material.
        Used in the Differential Evolution calculation
        Parameters:
        -----------
        PAR : numpy.ndarray
            Array with parameters for each layer, with shape (n, 2) where n >= layers+1
        layers : int
            Number of layers in the trapezoid structure
            
        Returns:
        --------
        numpy.ndarray
            Coordinate array for the trapezoid structure
        """
        try:
            # Validate input parameters
            if PAR is None or not isinstance(PAR, np.ndarray):
                raise TypeError("PAR must be a numpy array")
                
            if layers is None or not isinstance(layers, (int, float)) or layers < 0:
                raise ValueError(f"layers must be a non-negative number, got {layers}")
                
            # Check PAR dimensions
            if len(PAR) < layers + 1:
                raise ValueError(f"PAR array must have at least {layers + 1} rows, but has {len(PAR)}")
                
            if len(PAR.shape) < 2 or PAR.shape[1] < 2:
                raise ValueError(f"PAR must have at least 2 columns, but has shape {PAR.shape}")
            
            # Main calculation code - unchanged
            Coord = np.zeros([layers+1, 5, 1])
            for T in range(layers+1):
                if T == 0:
                    Coord[T, 0, 0] = 0
                    Coord[T, 1, 0] = PAR[0, 0]
                    Coord[T, 2, 0] = PAR[0, 1]
                    Coord[T, 3, 0] = 0
                    Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
                else:
                    Coord[T, 0, 0] = Coord[T-1, 0, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
                    Coord[T, 1, 0] = Coord[T, 0, 0] + PAR[T, 0]
                    Coord[T, 2, 0] = PAR[T, 1]
                    Coord[T, 3, 0] = 0
                    Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
                    
            return Coord
            
        except Exception as e:
            print(f"Error in SymCoordAssign_SingleMaterialOpt: {str(e)}")
            return None

    
    ### Simulations
    def SimTrap_SM(self):
        """
        Simulates the intensity for a single material trapezoid structure.
        
        This function:
        1. Generates coordinates using SymCoordAssign_SingleMaterial
        2. Calculates the form factor using FreeFormTrapezoid
        3. Applies Debye-Waller factor
        4. Computes the intensity
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (self.SimInt)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                raise AttributeError("Missing required scattering vector attributes: Qx and/or Qz")
            
            if not hasattr(self, 'DW'):
                raise AttributeError("Missing required attribute: DW (Debye-Waller factor)")
                
            if not hasattr(self, 'I0'):
                raise AttributeError("Missing required attribute: I0 (Intensity scaling factor)")
                
            if not hasattr(self, 'Bk'):
                raise AttributeError("Missing required attribute: Bk (Background intensity)")
            
            # Execute coordinate assignment and form factor calculation
            success = self.SymCoordAssign_SingleMaterial()
            if not success:
                raise RuntimeError("Failed to assign coordinates in SymCoordAssign_SingleMaterial")
                
            self.FreeFormTrapezoid()
            if not hasattr(self, 'form') or self.form is None:
                raise RuntimeError("Failed to calculate form factor in FreeFormTrapezoid")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(self.Qx, 2) + np.power(self.Qz, 2)) * np.power(self.DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = self.form * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity
            self.SimInt = np.power(Formfactor, 2) * self.I0 + self.Bk
            
            return self.SimInt
       
        except Exception as e:
            print(f"Error in SimTrap_SM: {str(e)}")
            self.SimInt = None
            return None
        
        
    def SimTrap_SMOpt(self,SimPar,layers,Qx,Qz):
        
        Coord=self.SymCoordAssign_SingleMaterialOpt(SimPar,layers)
        print('Used Coordinate ', Coord)
        form=self.FreeFormTrapezoidOpt(Coord,layers,Qx,Qz) 
        
        M=np.power(np.exp(-1*(np.power(self.Qx,2)+np.power(self.Qz,2))*np.power(self.DW_Optimized,2)),0.5)
        Formfactor = self.form*M
        Formfactor=abs(Formfactor)
        self.SimIntOpt = np.power(Formfactor,2)*self.I0_Optimized+self.Bk_Optimized
        return self.SimIntOpt
    
   
    
    def SimCyl_SM(self, Discretization):
        
        
        self.ConeFourierTransform(Discretization) 
        
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
    
    def SimCyl_GF(self, SimPar, layers, Intensity, Qr, Qz, Discretization):
        PARs=np.zeros([layers+1,2])
        PARs[:,0:2]=np.reshape(SimPar[0:(layers+1)*2],(layers+1,2))
        [I0,DW,Bk]=SimPar[layers*2+2:layers*2+5]
        
        F1 = self.ConeFourierTransformOpt(PARs,layers, Qz, Qr,Discretization)
        M=np.power(np.exp(-1*(np.power(Qr,2)+np.power(Qz,2))*np.power(DW,2)),0.5)
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
        
        
    def CDSAXS_DiffEvolution_Cyl(self,limit,Discretization):
        
        self.GenBounds(limit)
        
        self.SimPar_Optimized = differential_evolution(self.SimCyl_GF,self.bounds, args=(self.layers,self.Intensity,self.Qx,self.Qz, Discretization),polish=True)
        
        self.PAR=np.reshape(self.SimPar_Optimized.x[0:(self.layers+1)*2],(self.layers+1,2))

        [self.I0,self.DW,self.Bk]= self.SimPar_Optimized.x[self.layers*2+2:self.layers*2+5]
               
        self.SimCyl_SM(Discretization)
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