import numpy as np


def trapezoid_form_factor(bottom_left = )
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