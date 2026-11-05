import numpy as np
from dataclasses import dataclass


####################################################################################
# Store Parameters (frozen)
####################################################################################

@dataclass(frozen=True)
class Physical_Parameters: 

    # Temperature related parameters
    T_m : float = 273.15     # K, pure water freezing temperature
    T_B : float = 264.0      # K, temperature of spherical inclusion boundary 
    T_0 : float = 275.0      # K, initial temperature of the liquid
    C_pl : int = 3985        # Jkg^{-1}K^{-1}, brine specific heat capacity 
    C_ps : float = 2100.0    # Jkg^{-1}K^{-1}, pure water ice specific heat capacity 
    L_w : float = 3.335e5    # Jkg^{-1}, latent heat of fusion of water 
    k_l : float = 0.6        # Wm^{-1}K^{-1}, liquid water thermal conductivity
    k_s : float = 2.3        # Wm^{-1}K^{-1}, pure water ice thermal conductivity

    @property
    def alpha(self):
        """
        m^{2}s^{-1}, liquid water thermal diffusivity
        
        α = k_l / (ρ_0 C_pl)
        """
        return self.k_l / (self.rho_0 * self.C_pl)

    # Salt related parameters
    m : float = 0.0228      # K (g/kg)^{-1}, freezing point depression coefficient (salinity)
    C_0 : float = 0.0       # g/kg, reference salinity in EOS
    C_init : float = 35.0   # g/kg, initial salinity of the liquid
    beta_S : float = 8.3e-4 # (g/kg)^{-1}, saline contraction coefficient

    # Eutectic parameters
    C_E : int = 175         # g/kg, liquid salinity at the eutectic point for MgSO4-H20 system
    
    
    # Water related parameters
    rho_0 : int = 1000       # kgm^{-3}, referencewater density at 0 overpressure, reference salinity
    rho_s : int = 917        # kgm^{-3}, pure water ice

    @property
    def rho_b(self): 
        """ 
        kgm^{-3}, brine density used to compute Gamma
        
        ρ_b = ρ_0 exp{β_S (C_init - C_0)}
        """
        return self.rho_0 * np.exp(self.beta_S * (self.C_init - self.C_0) )



    # Pressure related parameters
    beta : float = 3.5e-10   # Pa^{-1}, liquid water compressibility (Lesage et al. )

    @property
    def Gamma(self):
        """
        KPa^{-1}, pressure slope in liquidus condition, 
        
        Γ = T (1/ρ_s - 1/ρ_b) / L
        """
        return self.T_m * (1 / self.rho_s - 1 / self.rho_b) / self.L_w    

    
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

