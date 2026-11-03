import numpy as np
from dataclasses import dataclass


####################################################################################
# Store Parameters (frozen)
####################################################################################

@dataclass(frozen=True)
class Physical_Parameters: 

    # Temperature related parameters
    T_m : float = 273.15     # K, freezing temperature for briny cryomagma
    T_B : float = 264.0      # K, spherical boundary of inclusion
    T_0 : float = 275.0      # K, initial temperature of the liquid
    C_pl : float = 4186.0    # Jkg^{-1}K^{-1}, specific heat capacity of liquid water
    C_ps : float = 2100.0    # Jkg^{-1}K^{-1}, specific heat capacity of ice (Lesage et al., 2022)
    L_w : float = 3.34e5     # Jkg^{-1}, latent heat of fusion for water 
    k_l : float = 0.6        # Wm^{-1}K^{-1}, thermal conductivity of liquid water
    k_s : float = 2.3        # Wm^{-1}K^{-1}, thermal conductivity of ice (Lesage et al., 2022)

    # Salt related parameters
    m : float = 0.024       # ppt^{-1}, salinity coefficient in liquidus condition
    C_0 : float = 35.0      # ppt, reference salinity in EOS
    beta_S : float = 8.0e-4 # ppt^{-1}, 

    # Eutectic related parameters
    C_E : float = 170.0     # ppt, eutectic salinity for MgSO4-H20 system
    
    # Water related parameters
    rho_w : float = 1000.0   # kgm^{-3}, pure water at 0 overpressure
    rho_0 : float = 1035.0   # kgm^{-3}, water at 0 overpressure, reference salinity
    rho_s : float = 917.0    # kgm^{-3}, ice

    # Pressure related parameters
    beta : float = 5.0e-10   # Pa^{-1}, compressibility in EOS (Lesage et al. )

    @property
    def Gamma(self):
        """
        KPa^{-1}, thermal gradient in liquidus condition, 
        
        Γ = T (1/ρ_s - 1/ρ_l) / L
        """
        return self.T_m * (1 / self.rho_s - 1 / self.rho_w) / self.L_w    

    @property
    def alpha(self):
        """
        m^{2}s^{-1}, thermal diffusivity
        
        α = k_l / (ρ_0 C_pl)
        """
        return self.k_l / (self.rho_0 * self.C_pl)

    
p = Physical_Parameters()



####################################################################################
# Store states (not frozen)
####################################################################################
@dataclass(frozen=False)
class State: 
    H     : np.ndarray    # enthalpy
    h_l   : np.ndarray    # specific enthalpy of liquid brine
    h_s   : np.ndarray    # specific enthalpy of ice
    S     : np.ndarray    # bulk salinity 
    rho_l : np.ndarray    # density of the liquid
    T     : np.ndarray    # temperature 
    C     : np.ndarray    # liquid salinity 
    chi   : np.ndarray    # liquid fraction

    dP    : float         # overpressure term

    u_r   : np.ndarray    # radial speed

    J           : np.ndarray    # Brine flux
    q           : np.ndarray    # Heat flux
    h_faces     : np.ndarray    # Enthalpy at the faces
    C_faces     : np.ndarray    # Liquid salinity at the faces
    rho_l_faces : np.ndarray    # Liquid density at the faces

    rho_l_old   : np.ndarray    # Store old liquid density values
    chi_old     : np.ndarray    # Store old liquid fraction values

