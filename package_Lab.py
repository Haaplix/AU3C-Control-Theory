import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from package_DBR import Process

#-----------------------------------


#-----------------------------------        
def LL_RT(MV,Kp,Tlag,Tlead,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "LL_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "LL_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2]))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*((Tlead/Ts)*MV[-1]+(1-(Tlead/Ts)*MV[-2])))
            # elif method == 'TRAP':
            #     PV.append((1/(2*Tlag+Ts))*((2*Tlag-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])

#-----------------------------------


def IMC_tuning_H(K,T,theta,gamma,C): #P,C,gamma (paramètres du prof)
    TOLP = T
    TCLP = gamma * TOLP
    Ti = TOLP + (theta/2)
    Td = (TOLP*theta)/((2*TOLP)+theta)
    KcK=(TOLP + (theta/2))/(TCLP+(theta/2))
    Kc = KcK/K
    C = Process({})
    C.parameters['Ti'] = Ti
    C.parameters['Td'] = Td
    C.parameters['Kc'] = Kc