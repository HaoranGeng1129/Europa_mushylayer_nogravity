from classes import p
import numpy as np


####################################################################################
# Enthalpy
####################################################################################

def calculate_liquid_enthalpy(state):
    """
    Specific enthalpy of liquid water

    h_l = L_w + C_pl * (T - T_m) + dP / ρ_l
    """
    return p.L_w + p.C_pl * (state.T - p.T_m) + state.dP / state.rho_l


def calculate_ice_enthalpy(state):
    """
    Specific enthalpy of ice

    h_s = C_ps (T - T_m) + dP / ρ_s
    """
    return p.C_ps * (state.T - p.T_m) + state.dP / p.rho_s


def calculate_enthalpy(state):
    """
    Enthalpy

    H = ρ_l h_l χ + ρ_s h_s (1 - χ)
    """
    return state.rho_l * state.h_l * state.chi + p.rho_s * state.h_s * (1 - state.chi)



####################################################################################
# Bulk Salinity
####################################################################################

def calculate_bulk_salinity(state):
    """
    Bulk salinity

    S = ρ_l C χ + ρ_s C_s (1 - χ), with C_s = 0
    """
    return state.rho_l * state.C * state.chi



####################################################################################
# Fluxes
####################################################################################

def calculate_brine_flux(state, V, dt):
    """
    Brine flux through unit area of the faces (kg s^{-1})

    J is defined as ρ_l u_r r^2. Quoting the mass conservation equation, its summation
    form is 

    J = Σ_i V_i [ m_new - m_old ] / dt , 

    where m_new = [ρ_l χ + ρ_s (1 - χ][new], m_old = [ρ_l χ + ρ_s (1 - χ)][old].
    np.cumsum() is used since the brine flux is determined by integration from R to r. 
    Minus sign since adding from wall to centre but array starts from the centre.
    """ 
    m_old = state.rho_l_old * state.chi_old + p.rho_s * (1 - state.chi_old)
    m_new = state.rho_l     * state.chi     + p.rho_s * (1 - state.chi)
    dJ = V * (m_new - m_old ) / dt

    J = np.zeros_like(state.J) 
    J[0]   = 0.0
    J[1: ] = - np.cumsum(dJ)
    J[-1]  = 0.0

    return J


def calculate_liquid_enthalpy_at_faces(state):
    """ 
    Taking the specific liquid enthalpy at i - 1/2 face to be the specific liquid 
    enthalpy of the ith cell. Append the liquid enthalpy at the wall of the inclusion 
    as the liquid enthalpy of the final cell. The formula to compute the specific 
    liquid enthalpy remains the same.

    h_l = L_w + C_pl (T - T_m) + dP / ρ_l

    Use in calculation of enthalpic flux.
    """
    state.h_l = calculate_liquid_enthalpy(state)
    h_faces = np.append(state.h_l, state.h_l[-1])

    return h_faces


def calculate_weighted_thermal_conductivity_at_faces(state):
    """ 
    The thermal conductivity weighted by liquid fraction, used 
    to calculate heat flux 
    
    k_bar = k_l χ_bar + k_s (1 - χ_bar),
    where χ_bar_{i+1/2} = (χ_{i+1} + χ_{i}) / 2
    """
    chi_bar = (state.chi[1: ] + state.chi[: -1]) / 2
    return p.k_l * chi_bar + p.k_s * (1 - chi_bar)


def calculate_heat_flux(state, dr):
    """ 
    Heat flux through unit area of the faces (J s^{-1})

    q = - k_bar (T_{i+1} - T{i}) / dr,

    for faces sandwiched between cells. The flux at r = 0 should 
    be zero since energy is conserved at the centre of the sphere. 
    The flux at r = R is related to the difference between the 
    temperature at the boundary, T_B, and the temperature of the
    last cell, T[-1].    
    """
    k_bar = calculate_weighted_thermal_conductivity_at_faces(state)

    q = state.q

    q[0]     = 0.0
    q[1: -1] = - k_bar * (state.T[1: ] - state.T[: -1]) / dr
    q[-1]    = - p.k_s * (p.T_B - state.T[-1]) / (dr/2)

    return q


def calculate_liquid_salinity_at_faces(state):
    """ 
    The same idea of the computation of h_faces
    """
    C_faces = np.append(state.C, state.C[-1])
    
    return C_faces


def calculate_liquid_density_at_faces(state):
    """ 
    Again, the same idea as h_faces and C_faces
    """
    rho_l_faces = np.append(state.rho_l, state.rho_l[-1])

    return rho_l_faces


def calculate_radial_speed(state, r_f):
    """ 
    u_r = J / (rho_l_faces  r_f^2) 

    Note the 0th entry, i.e. radial speed at the centre is kept at 
    zero following the boundary condition, also to avoid division 
    by zero.
    """
    rho_l_faces = calculate_liquid_density_at_faces(state)

    u_r = np.zeros_like(state.u_r)
    u_r[0]  = 0.0
    u_r[1: ] = state.J[1: ] / (rho_l_faces[1: ] * r_f[1: ]**2)

    return u_r



####################################################################################
# Overpressure Term, dP
####################################################################################

def calculate_overpressure_term(state, V):
    """ 
    dP = 1/β ln[(M_0 - ρ_s Σ_i (1 - χ) V_i) / (ρ_0 Σ_i exp{β_S(C - C_0) χ V_i })]
    """
    M_0 = np.sum(p.rho_0 * V)
    numerator = M_0 - p.rho_s * np.sum((1 - state.chi) * V)
    denominator = p.rho_0 * np.sum(np.exp(p.beta_S * (state.C - p.C_0)) * state.chi * V )

    return np.log(numerator / denominator) / p.beta



####################################################################################
# MISC
####################################################################################

def calculate_liquid_density(state):
    """ 
    EOS to calculate the density of liquid water

    ρ = ρ_0 exp[β ΔP + β_S (C - C_0)].
    """ 
    return p.rho_0 * np.exp(p.beta * state.dP + p.beta_S * (state.C - p.C_0))


def calculate_liquidus_temperature(state):
    """
    Liquidus temperature of briny cryomagma

    T_L = T_m - m C - Γ ΔP
    """
    return p.T_m - p.m * state.C - p.Gamma * state.dP


def calculate_eutectic_temperature(state):
    """
    Eutectic temperature of briny cryomagma

    T_E = T_m - m C_E - Γ ΔP    
    """
    return p.T_m - p.m * p.C_E - p.Gamma * state.dP


def calculate_dynamic_viscosity(state):
    """
    Dynamic viscosity of water

    η = 
    """
    eta = 1e-10

    return eta

