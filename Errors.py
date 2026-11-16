import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


####################################################################################
# Plotting fracture time against the radial increment values
####################################################################################

N = np.array([100000, 50000, 30000, 10000, 5000, 3000, 1000])
dr = 1000 / N                      # radial increment (m), exact rather than rounded
t = np.array([6.9012, 6.9025, 6.9058, 6.9251, 6.9310, 7.0160, 7.3872])  # fracture time (yr)

def power_law(x, t_inf, C, p):
    return t_inf + C * x**p       

parameters, covariance_matrix = curve_fit(power_law, dr, t, p0=[6.9, 0.5, 1.0])
t_inf, C, p = parameters
t_inf_error, C_error, p_error = np.sqrt(covariance_matrix.diagonal())

print('t_inf: {:.4f} +/- {:.4f} yr'.format(t_inf, t_inf_error))
print('C:     {:.4f} +/- {:.4f} yr'.format(C, C_error))
print('p:     {:.3f} +/- {:.3f}'.format(p, p_error))

x_fit = np.logspace(-2.1, 0.05, 300)   # smooth curve for plotting

plt.figure(figsize=(5, 4), dpi=200)
plt.plot(dr, t, 'o', label='Simulation runs')
plt.plot(x_fit, power_law(x_fit, t_inf, C, p), 'r',
         label=r'Fit $t_\infty + C \, dr^{p}$, $p$ = %.2f' % p)
plt.axhline(t_inf, color='grey', ls='--', label=r'$t_\infty$ = %.4f yr' % t_inf)
plt.xscale('log')
plt.title('Fracture time against radial increment')
plt.xlabel('Radial increment, dr (m)')
plt.ylabel('Fracture time (years)')
plt.legend()
plt.show()



####################################################################################
# Plotting the errors in fracture time against the radial increment values
####################################################################################

errors = np.abs(t - t_inf) / t_inf

fit_errors = C * x_fit**p / t_inf          # error predicted by the power-law fit above

plt.figure(figsize=(5, 4), dpi=200)
plt.plot(dr, errors, 'o', label='Errors in fracture time')
plt.plot(x_fit, fit_errors, 'r',
         label=r'Fit $C \, dr^{p} / t_\infty$, $p$ = %.2f $\pm$ %.2f' % (p, p_error))

plt.xscale('log')
plt.yscale('log')
plt.title('Errors in fracture time against radial increment')
plt.xlabel('Radial increment, dr (m)')
plt.ylabel('$|t - t_\infty| / t_\infty$')
plt.legend()
plt.show()