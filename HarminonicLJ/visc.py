import numpy as np
import matplotlib.pyplot as plt
from parse_outcar import *
from physchem_funcs import *
from mathstats_funcs import *
from lammps_log_read import extract_properties_from_log
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl

common_path = 'C:/Users/Vitor/Desktop/Simulations/phd/lj'

# 1) I need to run ~5 independent longer simulations. Then, I iterate over them, store the cum_psi_avg for each, and
#    take the mean. If the result is noisy I can fit some function to smooth the noise.

# 2) Run with same system size with LJ clusters. Compare the two S(k) and check for prepeak in the LJ cluster.

# 3) Plot the viscosity of the two systems on the same graph and see how much larger the LJ viscosity cluster are.

# 4) If everything work as expected, compute individual atomic stress for the particle in the center of the cluster.
#    Then, compute the auto and cross correlation with respect to the atomic stress of other particles in the
#    cluster. Compare this correlations with the correlations obtained from a random particle in the bulk liquid.

# 5) PhD

# Plots I need:
# 1) Harmonic LJ potential;
#    + LJ phase diagram from another article to show that it is indeed in the liquid region
# 2) g(r) for no cluster and 2 more clusters to show that it is liquid
# 3) Viscosity for different cluster sizes
# 4) Viscosity
# 5) atomic stress autocorrelation decay for cluster center vs bulk atom
# 6) atomic stress crosscorrelation decay for cluster center vs bulk atom using `n_cluster` nearest neighbors


#50000 with thermo 4 => 12500 entries
cum_psi_matrix = []
n_clusters = 7
k_values = [1, 10, 25, 50, 100, 200]
T_values = [1.0, 1.1, 1.2, 1.3]
for i in T_values:
    n_frames = 45000
    # data = extract_properties_from_log(f'{common_path}/C{i}_3500_log.lammps')
    data = extract_properties_from_log(f'{common_path}/L_T{i}_c4_3500_log.lammps')
    Pxy = data['pxy'].to_numpy()[-n_frames:]
    Pyz = data['pyz'].to_numpy()[-n_frames:]
    Pxz = data['pxz'].to_numpy()[-n_frames:]
    P = np.array([Pxy, Pyz, Pxz])
    V = data['vol'].to_numpy()[-1]

    divisor_list = [1.5]
    n_samples = 10

    eta_avg, eta_std, cum_psi_avg = bootstrap_gk(divisor_list, n_samples, P, box_vol=V, temperature=1.0, dt=0.005,
                                                 dump_freq=4, calc_type='viscosity', n_perc=1, algo=2, cross0=False,
                                                 moving_avg_window=1, units='lj', plot_info=False)
    print(eta_avg)
    cum_psi_matrix.append(cum_psi_avg)

cum_psi_matrix = np.array(cum_psi_matrix)

# One point to make is that I'm trying to replicate the ionic clusters. Thus, the cluster is formed by a center atom
# connected by harmonic bonds to `M` neighboring atoms. These neighboring atoms are not bonded among themselves; all
# bonds involve the cluster atom. This is the simplest possible cluster and also replicates the ionic clusters
# encountered when Th and U are added to molten salts. For the more complex structures where neighboring atoms are
# also bonded, we should expect

#----------------------------------------------------------------------------------------------------------------------#
# Total viscosity as a function of cluster size with increasing number of bonds

# This tell us that as the cluster size increases the viscosity also increases. Or, more generally, as the number of
# bonds increases (i.e. fixed atomic environments or neighbors) the viscosity increases. Also, the time it takes for
# the shear stress to decorrelate with itself also takes longer (the saturation index increases with cluster size).
colors = pl.cm.turbo(np.linspace(0.2, 1, 7))
fig, ax = plt.subplots()
for i, k in enumerate(T_values):
    func = lambda x, a, k: a*(1-np.exp(-k*x))
    idx_max = 800
    x_data = 4*np.arange(0, idx_max, 1)
    popt, _ = curve_fit(func, x_data, cum_psi_matrix[i, :idx_max])
    print(func(x_data[-1], *popt))

    idx_plot = 800
    x_plot = 4*np.arange(0, idx_plot, 1)

    # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label=f'$k = {k}$', lw=2.0)
    # ax.plot(x_plot, cum_psi_matrix[i, :idx_plot], color=colors[i], label=f'$k = {k}$', lw=2.0)
    # if i == 0:
        # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label='No Cluster', lw=2.0)
        # ax.plot(x_plot, cum_psi_matrix[i, :idx_plot], color=colors[i], label='No Cluster', lw=2.0)
    # else:
        # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label=f'{int(3500 / (i + 1))} clusters of size {i + 1}', lw=2.0)
        # if i < 3:
        #     ax.plot(x_plot, cum_psi_matrix[i, :idx_plot], color=colors[i], label=f'{int(3500 / (i + 1))} clusters of size {i + 1}', lw=2.0)
            # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label=f'{int(3500 / (i + 1))} clusters of size {i + 1}', lw=2.0)
        # else:
        #     ax.plot(x_plot, cum_psi_matrix[i, :idx_plot], color=colors[i], label=f'$\, ${int(3500 / (i + 1))} clusters of size {i + 1}', lw=2.0)
            # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label=f' {int(3500 / (i + 1))} clusters of size {i + 1}', lw=2.0)
    ax.plot(x_plot, cum_psi_matrix[i, :idx_plot], color=colors[i], label=f'$T={T_values[i]}$', lw=2.0)
    # ax.plot(x_plot, func(x_plot, *popt), color=colors[i], label=f'$T={T}$', lw=2.0)
# ax.set_ylim([0, 3.0e-9])
ax.set_xlabel('GK Integral Truncation Frame')
ax.set_ylabel(r'$\eta$ (LJ units)')
ax.legend(prop={'size': 24})
# fig.savefig(f'C:/Users/Vitor/Desktop/bresa/Python/scripts/zlab/plots_saved/phd/lj/visc_k_fit.svg', dpi=500)
plt.show()




#----------------------------------------------------------------------------------------------------------------------#
# Total viscosity for N=0 and {10, 25, 50, 100, 200} N=4 clusters

# The purpose of this next plot is to study viscosity by fixing the total number of bonds but varying the size of the
# cluster. I want to study whether the structure of the cluster as a whole
