import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes import p, d

####################################################################################
# Plotting the evolution of the profile
####################################################################################
def animate_evolutions_profiles(T_history, rho_l_history, H_history, S_history):
    """ 
    Creates an animation that displays the temporal evolution of temperature, liquid 
    density, enthalpy, and bulk salinity. 
    """
    fig, axis = plt.subplots(2, 2, figsize = (12, 8))

    axis[0, 0].set_xlim(d.R - 50, d.R)
    axis[0, 0].set_ylim(220, 280)
    axis[0, 0].set_xlabel("r (m)")
    axis[0, 0].set_ylabel("T (K)")
    axis[0, 0].set_title("Temperature")

    axis[0, 1].set_xlim(d.R - 50, d.R)
    axis[0, 1].set_ylim(900, 1200)
    axis[0, 1].set_xlabel("r (m)")
    axis[0, 1].set_ylabel("rho_l (kg/m^3)")
    axis[0, 1].set_title("Liquid Density")

    axis[1, 0].set_xlim(d.R - 50, d.R)
    axis[1, 0].set_ylim(-5e8, 5e8)
    axis[1, 0].set_xlabel("r (m)")
    axis[1, 0].set_ylabel("H (J/m^3)")
    axis[1, 0].set_title("Enthalpy")

    axis[1, 1].set_xlim(d.R - 50, d.R)
    axis[1, 1].set_ylim(0, 5e4)
    axis[1, 1].set_xlabel("r (m)")
    axis[1, 1].set_ylabel("S (g/m^3)")
    axis[1, 1].set_title("Bulk Salinity")

    temperature_plot,     = axis[0, 0].plot([], [])
    liquid_density_plot,  = axis[0, 1].plot([], [])
    enthalpy_plot,        = axis[1, 0].plot([], [])
    bulk_salinity_plot,   = axis[1, 1].plot([], [])

    clock = fig.suptitle("t = 0.00 years")

    def update_data(frame):
        """ 
        Updates the inputs for the plots at each frame of the animation.
        """
        temperature_plot.set_data(d.r_c, T_history[frame])
        liquid_density_plot.set_data(d.r_c, rho_l_history[frame])
        enthalpy_plot.set_data(d.r_c, H_history[frame])
        bulk_salinity_plot.set_data(d.r_c, S_history[frame])

        t_years = (100 * frame + 1) * d.dt / (365.25 * 24 * 3600)
        clock.set_text(f"t = {t_years:.2f} years")

        return temperature_plot, liquid_density_plot, enthalpy_plot, bulk_salinity_plot, clock


    # Animation function
    animation = FuncAnimation(
                        fig = fig, 
                        func = update_data,
                        frames = len(T_history) ,  
                        interval = 25,
                            )
    plt.tight_layout()
    plt.show()

    return animation