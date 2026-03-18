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
                PV.append((1-K)*PV[-1] + K*Kp*((Tlead/Ts)*MV[-1]+((1-(Tlead/Ts))*MV[-2])))
            # elif method == 'TRAP':
            #     PV.append((1/(2*Tlag+Ts))*((2*Tlag-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])

#-----------------------------------


def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    
    """ 

The function "PID_RT" needs to be included in a "for or while loop".

:SP: SP (or SetPoint) vector

:PV: PV (or Process Value) vector

:Man: Man (or Manual controller mode) vector (True or False)

:MVMan: MVMan (or Manual value for MV) vector

:MVFF: MVFF (or Feedforward) vector

:Kc: controller gain

:Ti: integral time constant [s]

:Td: derivative time constant [s]

:alpha: Tfd = alpha * Td where Tfd is the derivative filter time constant [s]

:Ts: sampling period [s]

:MVMin: minimum value for MV (used for saturation and anti wind-up)

:MVMax: maximum value for MV (used for saturation and anti wind-up)

:MV: MV (or Manipulated Value) vector

:MVP: MVP (or Proportional part of MV) vector

:MVI: MVI (or Integral part of MV) vector

:MVD: MVD (or Derivative part of MV) vector

:E: E (or control Error) vector

:ManFF: Activated FF in manual mode (optional: default boolean value is False)

:PVInit: Initial value for PV (optional: default value is 0); used if PID_RT is run first in the sequence and no value of PV is available yet.

:method: discretisation method (optional: default value is 'EBD-EBD')

EBD-EBD:  EBD for integral action and EBD for derivative action  
EBD-TRAP: EBD for integral action and TRAP for derivative action  
TRAP-EBD: TRAP for integral action and EBD for derivative action  
TRAP-TRAP: TRAP for integral action and TRAP for derivative action

The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD".
The appended values are based on the PID algorithm, the controller mode, and feedforward.
Note that saturation of "MV" within the limits [MVMin, MVMax] is implemented with anti wind-up. """




    Tfd = alpha * Td
    # Compute the control error
    if len(E) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
    
    #MVP
    MVP.append(Kc*E[-1]) #proportional action

    #MVI 
    if len(MVI) == 0:
        MVI.append(Kc*Ts/Ti*E[-1]) #initial condition for MVI
    else:
        if method == 'TRAP':
            MVI.append(MVI[-1] + Kc*Ts/(2*Ti)*(E[-1] + E[-2])) #TRAP for integral action
        else:
            MVI.append(MVI[-1] + Kc*Ts/Ti*E[-1]) #default is EBD for integral action

    #MVD
    if len(MVD) == 0:
        if method == 'TRAP-EBD' or method == 'TRAP-TRAP':
            MVD.append((Kc*Td/(Tfd + (Ts/2)))*E[-1]) #initial condition for MVD with TRAP
        else:
            MVD.append((Kc*Td/(Tfd + Ts))*E[-1]) #initial condition for MVD
    else:
        if method == 'EBD-TRAP' or method == 'TRAP-TRAP':
            MVD.append(((Tfd-(Ts/2))/(Tfd + (Ts/2)))*MVD[-1] + ((Kc*Td)/(Tfd + (Ts/2)))*(E[-1] - E[-2])) #TRAP for derivative action
        else:
            MVD.append((Tfd/(Tfd + Ts))*MVD[-1] + ((Kc*Td)/(Tfd + Ts))*(E[-1] - E[-2])) #default is EBD for derivative action



    #MANUAL MODE + Wind-up 
    if Man[-1] == True:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else:
            MVI[-1] = MVMan[-1] - MVFF[-1] - MVP[-1] - MVD[-1]

    
    #sturation
    if MVMax < MVI[-1] + MVP[-1] + MVD[-1] + MVFF[-1]:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]
    elif MVMin > MVI[-1] + MVP[-1] + MVD[-1] + MVFF[-1]:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]

    #MV
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])

def IMC_tuning_H(K,T,theta,gamma): #P,C,gamma (paramètres du prof)
    TOLP = T
    TCLP = gamma * TOLP
    Ti = TOLP + (theta/2)
    Td = (TOLP*theta)/((2*TOLP)+theta)
    KcK=(TOLP + (theta/2))/(TCLP+(theta/2))
    Kc = KcK/K
    return Ti,Td,Kc


