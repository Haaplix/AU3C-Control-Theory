import os
from datetime import datetime
from arrow import now
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from package_DBR import Bode, Process

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
        print(MVI, 'initial MVI')
    else:
        if method in ['TRAP-EBD', 'TRAP-TRAP']:
            MVI.append(MVI[-1] + (Kc*Ts/(2*Ti))*(E[-1] + E[-2])) #TRAP for integral action
        else:
            MVI.append(MVI[-1] + ((Kc*Ts)/Ti)*E[-1]) #default is EBD for integral action

    #MVD
    if len(MVD) == 0:
        if method in ['EBD-TRAP', 'TRAP-TRAP']:
            MVD.append(((Kc*Td)/(Tfd + (Ts/2)))*E[-1]) #initial condition for MVD with TRAP
        else:
            MVD.append(((Kc*Td)/(Tfd + Ts))*E[-1]) #initial condition for MVD
    else:
        if method in ['EBD-TRAP', 'TRAP-TRAP']:
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

    

def IMC_tuning(Kp,T1,T2,theta,gamma,method="SO"): 

     
    """ 
    :Kp: protocol gain

    :T1: main time constant [s]

    :T2: secondary time constant [s]

    :theta: delay

    :gamma: in the range [0,2 ... 0,9], determines the rapidity, agressivity of the reponse

    :method: discretisation method (optional: default value is 'SO')

    SO:  SO for the case second order + dead-time 
    FO: FO for the case first order + dead-time (for a PID)

    The function "IMC_tuning" returns the value of Ti (integral time constant [s]), Td (derivative time constant [s]) and Kc (controller gain).
    The returned values are based on the table "IMC-Based PID Controller Settings for Gc(s)"."""


    TCLP = gamma * T1

    if (method == "FO"):
        Ti = T1 + (theta/2)
        Td = (T1*theta)/((2*T1)+theta)
        KcK=(T1 + (theta/2))/(TCLP+(theta/2))
        Kc = KcK/Kp
        
    if (method == "SO"):
        Ti = T1 + T2
        Td = (T1*T2)/(T1+T2)
        KcK= (T1+T2)/(TCLP+theta)
        Kc = KcK/Kp
    else:
        Ti = T1 + T2
        Td = (T1*T2)/(T1+T2)
        KcK= (T1+T2)/(TCLP+theta)
        Kc = KcK/Kp
    
    return Ti,Td,Kc
    
class Controller:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 1.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['TFD'] = parameters['TFD'] if 'TFD' in parameters else 0.0
        self.parameters['E'] = parameters['E'] if 'E' in parameters else 0.0
        


    
def Margin(P, C, omega, save_fig=False):

    """
    Calculate the gain and phase margins of a control system.

    P: Process transfer function

    C: Controller transfer function

    omega: frequency range for Bode plot

    save_fig: Boolean to decide whether to save the Bode plot figure in Plots folder

    The function returns the gain margin (dB) and the phase margin (degrees) of the system. It also plots and shows the Bode Diagram of P(s)*C(s) with the gain and phase margins annotated.
    """

    
    C_gain = 20*np.log10(np.abs(C)/5)
    C_phase = np.degrees(np.unwrap(np.angle(C)))


    P_gain = 20*np.log10(np.abs(P)/5)
    P_phase = np.degrees(np.unwrap(np.angle(P)))


    L = P * C
    L_gain = 20*np.log10(np.abs(L)/5)
    L_phase = np.degrees(np.unwrap(np.angle(L)))  

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1)
    fig.set_figheight(12)
    fig.set_figwidth(22)

    

    ax_gain.semilogx(omega,L_gain,label=r'$P(s)*C(s)$') # L gain
    ax_gain.semilogx(omega,C_gain,label=r'$C(s)$', linestyle='--') # C gain
    ax_gain.semilogx(omega,P_gain,label=r'$P(s)$', linestyle='--', color="black") # P gain
    gain_min = np.min(L_gain)
    gain_max = np.max(L_gain)
    ax_gain.set_xlim([np.min(omega), np.max(omega)])
    #ax_gain.set_ylim([gain_min, gain_max])
    ax_gain.set_ylabel('Amplitude' + r'\n $|P(s)*C(s)|$ [dB]')
    ax_gain.set_title('Bode plot of P(s)*C(s)')
    ax_gain.legend(loc='best')
    ax_gain.grid(which='both', linestyle='--', linewidth=0.5)

    ax_gain.axhline(0, color='red', linestyle='-.', linewidth=1, label='0 dB')

    # Phase crossover frequency → gain margin
    phase_cross_idx = np.argmin(np.abs(L_phase - (-180)))
    omega_phase_cross = omega[phase_cross_idx]
    gain_at_phase_cross = L_gain[phase_cross_idx]
    gain_margin = gain_at_phase_cross  

    ax_gain.axvline(omega_phase_cross, color='green', linestyle=':', linewidth=1.2)
    ax_phase.axvline(omega_phase_cross, color='green', linestyle=':', linewidth=1.2)
    ax_gain.annotate('', 
                 xy=(omega_phase_cross*1.1, gain_at_phase_cross),
                 xytext=(omega_phase_cross*1.1, 0),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    
    ax_gain.text(omega_phase_cross*1.15, gain_at_phase_cross/2, f'GM = {gain_margin:.2f} dB', color='green', fontsize=10)


    ax_phase.semilogx(omega, L_phase,label=r'$P(s)*C(s)$') # L phase
    ax_phase.semilogx(omega, C_phase,label=r'$C(s)$', linestyle='--') # C phase
    ax_phase.semilogx(omega, P_phase,label=r'$P(s)$', linestyle='--', color="black") # P phase   
    ax_phase.set_xlim([np.min(omega), np.max(omega)])
    ph_min = np.min(L_phase) - 10
    ph_max = np.max(L_phase) + 10
    ax_phase.set_ylim([np.max([ph_min, -200]), 25])
    ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')        
    ax_phase.set_ylabel('Phase' + r'\n $\,$'  + r'$\angle P(s)*C(s)$ [°]')
    ax_phase.legend(loc='best')
    ax_phase.grid(which='both', linestyle='--', linewidth=0.5)


    # -180° reference line
    ax_phase.axhline(-180, color='red', linestyle='-.', linewidth=1, label=r'$-180°$')

    # Gain crossover frequency (where gain = 0 dB) → phase margin
    gain_cross_idx = np.argmin(np.abs(L_gain))
    omega_gain_cross = omega[gain_cross_idx]
    phase_at_gain_cross = L_phase[gain_cross_idx]
    phase_margin = phase_at_gain_cross + 180  # PM = phase + 180 at gain crossover

    ax_phase.axvline(omega_gain_cross, color='purple', linestyle=':', linewidth=1.2)
    ax_gain.axvline(omega_gain_cross, color='purple', linestyle=':', linewidth=1.2)


    ax_phase.annotate('', 
                 xy=(omega_gain_cross*1.1, -180),
                 xytext=(omega_gain_cross*1.1, -95),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    
    ax_phase.text(omega_gain_cross*1.15, -137.5, f'PM = {phase_margin:.2f}°', color='purple', fontsize=10)
    
    if save_fig:
        if not os.path.exists('Plots'):
            os.makedirs('Plots')    
        date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        name = f'Margin_{date_time}'
        plt.savefig(f'Plots\\' + name+ '.png')
        plt.savefig(f'Plots\\' + name+ '.svg')
    

    return gain_margin, phase_margin
