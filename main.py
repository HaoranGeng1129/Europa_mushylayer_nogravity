####################################################################################
# Main script
####################################################################################
from classes import d, state, h
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
from Animation import plot_profiles

# Compute initial state values
state.rho_l = calculate_liquid_density(state)
state.h_l   = calculate_liquid_enthalpy(state)
state.h_s   = calculate_ice_enthalpy(state)
state.H     = calculate_enthalpy(state)
state.S     = calculate_bulk_salinity(state)

# Iteration index        
i = 0

# Main loop
while i < d.N_iter:

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

    # Calculate lagged parameters 
    state.rho_l = calculate_liquid_density(state)
    state.J = calculate_brine_flux(state, d.V, d.dt) 
    state.u_r = calculate_radial_speed(state, d.r_f)

    if i % 100 == 0:
            h.T_history[i // 100, :]     = state.T.copy()     # temperature
            h.rho_l_history[i // 100, :] = state.rho_l.copy() # liquid density 
            h.H_history[i // 100, :]     = state.H.copy()     # enthalpy
            h.S_history[i // 100, :]     = state.S.copy()     # bulk salinity
            h.dP_history[i // 100]       = state.dP.copy()    # overpressure term
            h.time_history[i // 100]     = (i + 1) * d.dt / (365.25 * 24 * 3600) 

    i += 1

# Plot the temporal evolution animations and overpressure against time
plot_profiles()
