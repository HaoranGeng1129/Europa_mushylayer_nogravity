import numpy as np
from dataclasses import dataclass

####################################################################################
# Store Physical Parameters (frozen)
####################################################################################

@dataclass(frozen=True)
class Physical_Parameters: 

    # Temperature related parameters
    T_m : float    = 273.15    # K, pure water freezing temperature
    T_cold : float = 142.0     # K, initial temperature of surrounding ice shell
    T_B : float    = 142.0     # K, temperature at the domain boundary
    T_0 : float    = 275.0     # K, initial temperature of the liquid
    C_pl : int     = 3985      # Jkg^{-1}K^{-1}, brine specific heat capacity 
    C_ps : float   = 2100.0    # Jkg^{-1}K^{-1}, pure water ice specific heat capacity 
    L_w : float    = 3.335e5   # Jkg^{-1}, latent heat of fusion of water 
    k_l : float    = 0.6       # Wm^{-1}K^{-1}, liquid water thermal conductivity
    k_s : float    = 2.3       # Wm^{-1}K^{-1}, pure water ice thermal conductivity


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

    # Gravity relatd parameters
    g : float     = 1.315       # ms^{-2}, gravity on Europa, Pappalardo et al. (2009)
    depth : float = 2000.0      # m, depth of the resevoir, H in Lesage's paper

    # Pressure related parameters
    beta : float    = 3.5e-10   # Pa^{-1}, liquid water compressibility
    sigma_c : float = 1.5e6     # Pa , tensile strength of ice

    @property
    def Gamma(self):
        """
        KPa^{-1}, pressure slope in liquidus condition, 
        
        Γ = T (1/ρ_s - 1/ρ_b) / L
        """
        return self.T_m * (1 / self.rho_s - 1 / self.rho_b) / self.L_w      

    @property
    def critical_overpressure(self): 
        """ 
        ΔP_c = 2(σ_c + ρ_s g H)
        """
        return 2 * (self.sigma_c + self.rho_s * self.g * self.depth)

p = Physical_Parameters()



####################################################################################
# Store Domain Parameters
####################################################################################
class Domain_Parameters: 
    def __init__(self):
        self.N_iter : int = 500000              # number of iterations

        self.R_i : float  = 500.0               # m, radius of the resevoir
        self.R : float    = 1000.0              # m, radius of the entire domain
        self.N : int      = 50000               # number of cells in the domain  
        self.dr : float   = self.R / self.N     # m, radial distance increment

        self.r_f : np.ndarray = np.arange(0, self.N + 1, 1) * self.dr  # m, coordinates of cell faces
        self.r_c : np.ndarray = self.r_f[: -1] + self.dr / 2           # m, coordinates of cell centres

        self.A : np.ndarray = self.r_f ** 2   # m^2, surface area of spherical shell at r_f
        self.V : np.ndarray = (self.r_f[1:] ** 3 - self.r_f[:-1] ** 3) / 3  # m^3, volume of spherical shells

        self.dt : float = self.dr ** 2 /(2 * p.alpha * 10)  # s, time increment

d = Domain_Parameters()



####################################################################################
# Store States (not frozen)
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

state = State(
        H     = np.zeros(d.N), 
        h_l   = np.zeros(d.N), 
        h_s   = np.zeros(d.N), 
        S     = np.zeros(d.N), 
        rho_l = np.full(d.N, p.rho_b), 
        T     = np.full(d.N, p.T_0), 
        C     = np.full(d.N, p.C_init), 
        chi   = np.ones(d.N),
        dP    = 0.0, 
        u_r   = np.zeros(d.N + 1), 
        J     = np.zeros(d.N + 1), 
        q     = np.zeros(d.N + 1), 
        h_faces     = np.zeros(d.N + 1), 
        C_faces     = np.zeros(d.N + 1), 
        rho_l_faces = np.full(d.N + 1, p.rho_b), 
        rho_l_old   = np.zeros(d.N), 
        chi_old     = np.zeros(d.N)
              ) 



####################################################################################
# Store Historic Data
####################################################################################
@dataclass(frozen=False)
class Historic_Data:
    T_history     : np.ndarray  # temperature 
    rho_l_history : np.ndarray  # liquid density
    H_history     : np.ndarray  # enthalpy
    S_history     : np.ndarray  # bulk salinity

    time_history  : np.ndarray
    dP_history    : np.ndarray  # overpressure term 


h = Historic_Data(
        T_history     = np.zeros((d.N_iter // 500, d.N)), 
        rho_l_history = np.zeros((d.N_iter // 500, d.N)), 
        H_history     = np.zeros((d.N_iter // 500, d.N)),  
        S_history     = np.zeros((d.N_iter // 500, d.N)),  

        time_history  = np.zeros(d.N_iter // 500),
        dP_history    = np.zeros(d.N_iter // 500)      
                )
