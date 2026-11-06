####################################################################################
# Main script
####################################################################################

import numpy as np 

from classes import p, d, State
from functions import (
            calculate_liquid_enthalpy, 
            calculate_ice_enthalpy, 
            calculate_enthalpy, 
            calculate_bulk_salinity, 
            calculate_brine_flux,
            calculate_liquid_density, 
            calculate_heat_flux, 
            calculate_overpressure_term, 
            calculate_liquid_enthalpy_at_faces, 
            calculate_liquid_salinity_at_faces, 
            calculate_radial_speed
                       )

from phase_mask_and_inversion import inversion
from Animation import animate_evolutions_profiles

N_iter = 100000      # number of iterations

state = State(
        H = np.zeros(d.N), 
        h_l = np.zeros(d.N), 
        h_s = np.zeros(d.N), 
        S = np.zeros(d.N), 
        rho_l = np.full(d.N, p.rho_b), 
        T = np.full(d.N, p.T_0), 
        C = np.full(d.N, p.C_init), 
        chi = np.ones(d.N),
        dP = 0.0, 
        u_r = np.zeros(d.N + 1), 
        J = np.zeros(d.N + 1), 
        q = np.zeros(d.N + 1), 
        h_faces = np.zeros(d.N + 1), 
        C_faces = np.zeros(d.N + 1), 
        rho_l_faces = np.full(d.N + 1, p.rho_b), 
        rho_l_old = np.zeros(d.N), 
        chi_old = np.zeros(d.N)
              ) 


state.rho_l = calculate_liquid_density(state)
state.h_l = calculate_liquid_enthalpy(state)
state.h_s = calculate_ice_enthalpy(state)
state.H   = calculate_enthalpy(state)
state.S   = calculate_bulk_salinity(state)



# Iteration index        
i = 0
T_history     = np.zeros((N_iter // 100, d.N))  # store temperature profile for plotting
rho_l_history = np.zeros((N_iter // 100, d.N))  # liquid density
H_history     = np.zeros((N_iter // 100, d.N))  # enthalpy
S_history     = np.zeros((N_iter // 100, d.N))  # bulk salinity


while i < N_iter:

    # Update heat fluxes, h_faces, C_faces
    state.q   = calculate_heat_flux(state, d.dr)
    h_faces   = calculate_liquid_enthalpy_at_faces(state)
    C_faces   = calculate_liquid_salinity_at_faces(state)
   
    # Update enthalpy without overpressure term and bulk salinity
    H_flux = state.J * h_faces + d.A * state.q   # enthalpic flux at the faces
    S_flux = state.J * C_faces
    state.H -= d.dt * (H_flux[1: ] - H_flux[: -1]) / d.V
    state.S -= d.dt * (S_flux[1: ] - S_flux[: -1]) / d.V

    # Copies previous values for calculation of brine flux, J
    state.rho_l_old = state.rho_l.copy()
    state.chi_old   = state.chi.copy()

    # Inversion
    state.chi, state.T, state.C = inversion(state)  

    # Obtain dP using the new parameters from inversion(), and update 
    # the enthalpy accounting for the overpressure term
    dP_old = state.dP
    state.dP = calculate_overpressure_term(state, d.V)
    increase_in_dP = state.dP - dP_old 

    state.H += increase_in_dP

    # Recalculate rho_l from EOS
    state.rho_l = calculate_liquid_density(state)

    # Calculate brine flux. comes after new rho_l values are created
    state.J = calculate_brine_flux(state, d.V, d.dt) 

    # Compute u_r using mass conservation 
    state.u_r = calculate_radial_speed(state, d.r_f)

    if i % 100 == 0:
            T_history[i // 100, :]     = state.T.copy()  # store temperature profile for plotting
            rho_l_history[i // 100, :] = state.rho_l.copy() # liquid density 
            H_history[i // 100, :]     = state.H.copy()  # enthalpy
            S_history[i // 100, :]     = state.S.copy()  # bulk salinity

    i += 1

# Plot the temporal evolutions
animation = animate_evolutions_profiles(T_history, rho_l_history, H_history, S_history)


