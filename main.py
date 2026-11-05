####################################################################################
# Set up
####################################################################################

import numpy as np 

from classes import p, State
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


N_iter = 100000      # number of iterations

R   = 500.0               # m, radius of the spherical inclusion
M_0 = p.rho_0 * R**3 / 3  # kg, mass of the entire inclusion divided by 4π
N   = 10000               # number of cells in the domain 
dr  = R / N               # radial distance increment

r_f = np.arange(0, N + 1, 1) * dr           # Coordinates of the cell faces
r   = r_f[: -1] + dr / 2                    # Coordinates of the cell centers
A   = r_f ** 2    # area 
V   = (r_f[1:] ** 3 - r_f[:-1] ** 3) / 3    # volume

dt = dr ** 2 /(2 * p.alpha * 10)  # time increment, HAVENT EXPLAINED ORIGIN


state = State(
        H = np.zeros(N), 
        h_l = np.zeros(N), 
        h_s = np.zeros(N), 
        S = np.zeros(N), 
        rho_l = np.full(N, p.rho_b), 
        T = np.full(N, p.T_0), 
        C = np.full(N, p.C_init), 
        chi = np.ones(N), 
        dP = 0.0, 
        u_r = np.zeros(N + 1), 
        J = np.zeros(N + 1), 
        q = np.zeros(N + 1), 
        h_faces = np.zeros(N + 1), 
        C_faces = np.zeros(N + 1), 
        rho_l_faces = np.full(N + 1, p.rho_b), 
        rho_l_old = np.zeros(N), 
        chi_old = np.zeros(N)
              ) 


state.rho_l = calculate_liquid_density(state)
state.h_l = calculate_liquid_enthalpy(state)
state.h_s = calculate_ice_enthalpy(state)
state.H   = calculate_enthalpy(state)
state.S   = calculate_bulk_salinity(state)




# Iteration index        
i = 0
T_history = np.zeros((N_iter // 100, N))  # Store temperature profile for plotting

while i < N_iter:

    # Update heat fluxes, h_faces, C_faces
    state.q   = calculate_heat_flux(state, dr)
    h_faces   = calculate_liquid_enthalpy_at_faces(state)
    C_faces   = calculate_liquid_salinity_at_faces(state)
   
    # Update enthalpy without overpressure term and bulk salinity
    H_flux = state.J * h_faces + A * state.q   # enthalpic flux at the faces
    S_flux = state.J * C_faces
    state.H -= dt * (H_flux[1: ] - H_flux[: -1]) / V
    state.S -= dt * (S_flux[1: ] - S_flux[: -1]) / V

    # Copies previous values for calculation of brine flux, J
    state.rho_l_old = state.rho_l.copy()
    state.chi_old   = state.chi.copy()

    # Inversion
    state.chi, state.T, state.C = inversion(state)
    if i % 100 == 0:
        T_history[i // 100, :] = state.T.copy()  # Store temperature profile for plotting

    # Obtain dP using the new parameters from inversion(), and update 
    # the enthalpy accounting for the overpressure term
    dP_old = state.dP
    state.dP = calculate_overpressure_term(state, V)
    increase_in_dP = state.dP - dP_old 

    state.H += increase_in_dP

    # Recalculate rho_l from EOS
    state.rho_l = calculate_liquid_density(state)

    # Calculate brine flux. comes after new rho_l values are created
    state.J = calculate_brine_flux(state, V, dt) 

    # Compute u_r using mass conservation 
    state.u_r = calculate_radial_speed(state, r_f)

    i += 1







# Plotting the evolution of the temperature profile


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


fig, axis = plt.subplots()
axis.set_xlim(R - 100, R)
axis.set_ylim(260, 280)
axis.set_xlabel("r (m)")
axis.set_ylabel("T (K)")
axis.set_title("Temperature profile in the spherical inclusion")

animated_plot = axis.plot([], [])
 

def update_data(frame):
    """ 
    Updates the inputs for the plot at each frame of the animation.
    """
    animated_plot[0].set_data(r, T_history[frame])

    return animated_plot


# Animation function
animation = FuncAnimation(
                    fig = fig, 
                    func = update_data,
                    frames = len(T_history) ,  
                    interval = 25
                          )




plt.show()
