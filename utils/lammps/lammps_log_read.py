import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.integrate import cumulative_trapezoid


def autocorrelation(x, n_percentile=50):
    """Calculates the autocorrelation along the last axis of a 2-D or 1-D array.
    """
    assert x.ndim in [1,2], f"The dimension of the array has to be either 1 or 2, but this array has dimension {x.ndim}."
    N = x.shape[-1]
    acf_size = int(N * n_percentile/100) # this defines the maximum time delay tau_max.
    x_centered = np.transpose(x.T - x.mean(axis=-1))
    x_padded = np.hstack((x_centered, np.zeros_like(x)))
    x_fft = fft(x_padded)
    acf = ifft(x_fft * x_fft.conjugate()).real[..., :acf_size]

    return acf / np.arange(N, N-acf_size, -1)


def moving_average(x, window):
    """Calculates the moving average of an array over its last axis/dimension.

    Args:
        x (ndarray): Array that we want to calculate the moving average
        window (int): Size of the moving average, or how many elements will be "lumped" together into a single
                      representative number.

    Returns:
        (ndarray) containing the moving average. Note that the size of this array will depend on the window that
        we chose. For instance, for a window 'k' and a 1-D array of size 'N', the final size will be 'N-k'

    """
    x_csum = np.insert(np.cumsum(x, axis=-1), 0, 0, axis=-1)

    return (x_csum[..., window:] - x_csum[..., :-window]) / window


def green_kubo(A, V, T, dt, dump_freq, calc_type, n_perc=20, algo=2, cross0=True, units='vasp'):
    """Applies the Green-Kubo relation to calculate different generalized susceptibilities

    Args:
        A (ndarray): An 1-D or 2-D array that contains the relevant quantity, e.g. off-diagonal stresses for viscosity
                     and heat flux for thermal conductivity on each row.
        V (float): Volume of the simulation box in [A^3].
        T (float): Temperature of the simulation in [K].
        dt (float): Time step of the simulation in [fs].
        dump_freq (int): Frequency which the MD programam writes J_ii in the output file. For VASP dump_fre=1 always,
                         but for LAMMPS it is often not 1.
        calc_type (str): Type of calculation that we want to perfor, either 'thermal_conductivity' or 'viscosity'.
        n_perc (int): Percentage of the array to consider in the autocorrelation. This value should be small for the
                      statistics to be good.
        algo (int): For algo `1` computes the susceptibility for each array in `A` and average them. For algo `2`
                    averages the autocorreltions instead and compute a single susceptibility from it.
        cross0 (bool): If True integrates the autocorrelation until the first index after it crosses zero. If false
                       integrate the entire autocorrelation (this requires a very small n_perc ~5). Overall, if we
                       have a very small simulation (AIMD) it is better with cross0=True because the noice is very
                       large, i.e. the oscillation of the ACF around zero has a large amplitude. On the other hand,
                       if we have a large simulation (LAMMPS), then integrating the entire autocorrelation, i.e.
                       cross0=False, with a small n_perc ~ 5-10 (depending on how long the simulation is) is better.
        units (str): Units of the simulation that calculated `A`, can be either `lammps_metal` or `vasp`.

    Returns:
        psi (float): Generalizes susceptibility (e.g. thermal conductivity or viscosity)
        cumintegral_acf (ndarray): Array containing the values that `psi` can assume depending on where to truncate
                                   the integral.
        avg_acf (ndarray): Average autocorrelation


    OBS1: We take the cumulative integral to construct a nice plot of the viscosity converging to a final value.
          The viscosity is the limit where t->∞ and thus eta[-1]. However, this integral will oscillate around zero
          because the average shear pressure will decorrelate and oscillate around its mean 0 (I center the array
          around its mean in the acf function). Thus, one solution is to consider the value of the integral only
          until the first point where the autocorrelation crosses the zero. We usually take a moving average of
          the shear pressure to smooth its oscillations to avoid underestimating the viscosity in some cases (the
          oscillation would cross the zero line before the "trend").

    """
    if isinstance(A, list) or isinstance(A, tuple):
        A_vstack = np.vstack(A)
    else:
        A_vstack = A.reshape(1,-1)

    dtau = dt * dump_freq
    acf = autocorrelation(A_vstack, n_percentile=n_perc)
    avg_acf = np.mean(acf, axis=0)

    # Compute a susceptibility `psi` for each row in `A` and then averages the susceptibilities
    if algo == 1:
        psi = np.zeros(A_vstack.shape[0])
        acf_cumintegral = cumulative_trapezoid(acf, dx=dtau, initial=0, axis=1)
        for i in range(A_vstack.shape[0]):
            idx_truncate = np.flatnonzero(acf[i] < 0)[0] if cross0 else -1
            psi[i] = acf_cumintegral[i, idx_truncate]
        psi = np.mean(psi)

        if calc_type == 'viscosity':
            psi *= V / (8.61733e-5 * T) # V/k_b*T is the constant for viscosity
            if units == 'vasp':
                psi *= (1e-15 / 160217663400 * 1000 * 1e16) # [mPa.s] (fs to s; A^3/ev to 1/Pa; Pa to mPa; kBar^2 to Pa^2)
            elif units == 'lammps_metal':
                psi *= (1e-12 / 160217663400 * 1000 * 1e10) * 1e-3 # [mPa.s] (fs to s; A^3/ev to 1/Pa; Pa to mPa; Bar^2 to Pa^2)
                # The 1e-3 is to convert dtau from [fs] to [ps];

        elif calc_type == 'thermal_conductivity':
            psi *= V / (8.61733e-5 * T**2) # V/k_b*T^2 is the constant for thermal conductivity
            if units == 'lammps_metal':
                psi *= (1.60218e-19 / 1e-12) / 1e-10 * 1e-3 # [(eV/ps)/A.K * fs] to [(J/s)/m.K]

    # Compute the susceptibility from the average autocorrelation of each row in `A`
    elif algo == 2:
        acf_cumintegral = cumulative_trapezoid(avg_acf, dx=dtau, initial=0)

        if calc_type == 'viscosity':
            acf_cumintegral *= V / (8.61733e-5 * T)
            if units == 'vasp':
                acf_cumintegral *= (1e-15 / 160217663400 * 1000 * 1e16)
            elif units == 'lammps_metal':
                acf_cumintegral *= (1e-12 / 160217663400 * 1000 * 1e10) * 1e-3 # The 1e-3 is to convert dtau from [fs] to [ps]

        elif calc_type == 'thermal_conductivity':
            acf_cumintegral *= V / (8.61733e-5 * T**2)
            if units == 'lammps_metal':
                acf_cumintegral *= (1.60218e-19 / 1e-12) / 1e-10 * 1e-3

        idx_truncate = np.flatnonzero(avg_acf < 0)[0] if cross0 else -1
        psi = acf_cumintegral[idx_truncate]

    return psi, acf_cumintegral, avg_acf


def bootstrap_gk(divisor_list, n_samples, A, box_vol, temperature, dt, dump_freq, calc_type, n_perc=20, algo=2,
                 cross0=True, moving_avg_window=1, units='vasp', plot_info=False):
    """Bootstrap the trajectory window over a long trajectory and calculate a generalized susceptibility using GK.

       OBS1: The provided time step `dt` must be in units of [fs]
    """
    A_mov_avg = moving_average(A, moving_avg_window)
    n_frames = A_mov_avg.shape[1]
    psi_matrix = np.zeros((len(divisor_list), n_samples))

    for i, divisor in enumerate(divisor_list):
        sample_size = int(n_frames // divisor)
        for j in range(n_samples):
            starting_frame = np.random.randint(0, n_frames - sample_size)
            psi_matrix[i,j], cum_psi, acf = green_kubo(A_mov_avg[:, starting_frame:(starting_frame+sample_size)],
                                                       box_vol, temperature, dt, dump_freq, calc_type,
                                                       n_perc=n_perc, algo=algo, cross0=cross0, units=units)
            if plot_info:
                if j % int(n_samples/2) == 0:
                    fig, ax = plt.subplots(2,1)
                    ax[0].plot(acf, color='navy')
                    ax[0].axhline(0, color='black', ls='--')
                    ax[0].set_xlabel('Time Delay [units of dt*dump_freq]')
                    ax[0].set_ylabel('Average ACF')
                    ax[1].plot(cum_psi, color='orangered')
                    ax[1].set_xlabel('Integral truncation index')
                    ax[1].set_ylabel('Susceptibility')
                    if cross0:
                        cross0_idx = np.flatnonzero(acf < 0)[0]
                        ax[0].scatter(cross0_idx, acf[cross0_idx], color='red', s=15)
                        ax[1].axvline(cross0_idx, color='black', ls='--')
                        ax[1].scatter(cross0_idx, cum_psi[cross0_idx], color='forestgreen', s=15)
                    plt.show()

    return np.mean(psi_matrix).round(3), np.std(psi_matrix).round(3)


def extract_properties_from_log(log_file_path):
    """Create a dataframe with the properties in lammps log
    Args:
        log_file_path (str): Path of the log file.
    Returns:
        (DataFrame): A pandas dataframe whose columns are the thermodynamic properties in the log file.
    """
    data = []
    df_head = None

    with open(log_file_path, 'r') as f:
        for line in f:
            if 'thermo_style' in line:
                df_head = line.split()[2:]
                if '#' in df_head:
                    df_head = df_head[:df_head.index('#')]

            if df_head is not None:
                # Only the lines with the same length as df_head and with purely numeric entries are captured
                if len(df_head) == len(line.split()):
                    try:
                        properties = list(map(float, line.split()))
                        data.append(properties)
                    except:
                        continue

    df = pd.DataFrame(data, columns=df_head)

    return df


# if __name__ == "__main__":
#     # argv[1] -> Log file path
#     # argv[2] -> Number of the last `N` frames to consider
#     # argv[3] -> Temperature
#     # argv[4] -> Calculation type
#     data = extract_properties_from_log(sys.argv[1])
#     n_frames = int(sys.argv[2])
#     temp = float(sys.argv[3])
#
#
#     if sys.argv[4] == 'density':
#         rho = data['density'].to_numpy()
#         first_npt_idx = np.flatnonzero(np.diff(rho) != 0)[0] + 1
#         rho_npt = rho[first_npt_idx:]
#         print(f'{temp}K:', f'{np.mean(rho_npt[-n_frames:]):.3f}', f'{np.std(rho_npt[-n_frames:]):.3f}')
#
#     elif sys.argv[4] == 'viscosity':
#         Pxy = data['pxy'].to_numpy()[-n_frames:]
#         Pyz = data['pyz'].to_numpy()[-n_frames:]
#         Pxz = data['pxz'].to_numpy()[-n_frames:]
#         P = np.array([Pxy, Pyz, Pxz])
#         V = data['vol'].to_numpy()[-1]
#
#         n_used_frames = Pxy.shape[0]
#         divisor_list = [1.5, 1.7, 2]
#         n_samples = 300
#
#         eta_avg, eta_std = bootstrap_gk(divisor_list, n_samples, P, box_vol=V, temperature=temp, dt=1, dump_freq=4,
#                                         calc_type='viscosity', n_perc=20, algo=2, cross0=False, moving_avg_window=1,
#                                         units='lammps_metal', plot_info=False)
#
#         print(f'{temp}K:', f'{eta_avg:.3f}', f'{eta_std:.3f}')
#
#     elif sys.argv[4] == 'thermal_conductivity':
#         J_xx = data['v_Jx'].to_numpy()[-n_frames:]
#         J_yy = data['v_Jy'].to_numpy()[-n_frames:]
#         J_zz = data['v_Jz'].to_numpy()[-n_frames:]
#         J = np.array([J_xx, J_yy, J_zz])
#         V = data['vol'].to_numpy()[-1]
#
#         n_used_frames = J_xx.shape[0]
#         divisor_list = [1.3, 1.5]
#         n_samples = 100
#
#         k_avg, k_std = bootstrap_gk(divisor_list, n_samples, J, V, temp, dt=1, dump_freq=4, calc_type='thermal_conductivity',
#                                     n_perc=5, algo=2, cross0=False, units='lammps_metal', plot_info=True)
#
#         print(f'{temp}K:', f'k = {k_avg:.3f}', f'+/- {k_std:.3f} [W/m.K]')

## Example
# common_path = 'C:/Users/Vitor/Desktop/test_lammps2.txt' # path of the lammps log file
# data = extract_properties_from_log(common_path)
# print(data.head())
# print(data.shape)
# quit()

## Density
# rho = data['density'].to_numpy()
# first_npt_idx = np.count_nonzero(rho == rho[0])
# rho_npt = np.copy(rho[first_npt_idx:])
# print(f'{temp}:', f'{np.mean(rho_npt[-n_frames:]):.3f}', f'{np.std(rho_npt[-n_frames:]):.3f}') # 150

## Viscosity
# Pxy = data['Pxy'].to_numpy()
# Pyz = data['Pyz'].to_numpy()
# Pxz = data['Pxy'].to_numpy()

## Thermal Conductivity
# data = extract_properties_from_log('C:/Users/Vitor/Downloads/log_virial.lammps')
# n_frames = 50000
# temp = 1100
# J_xx = data['v_Jx'].to_numpy()[-n_frames:]
# J_yy = data['v_Jy'].to_numpy()[-n_frames:]
# J_zz = data['v_Jz'].to_numpy()[-n_frames:]
# # V = data['vol'].to_numpy()[-1]
# V = 144219
#
#
# n_used_frames = J_xx.shape[0]
# divisor_list = [1.2, 1.3]
# n_samples = 100
# J = np.array([J_xx, J_yy, J_zz])
# k_avg, k_std = bootstrap_gk(divisor_list, n_samples, J, V, temp, dt=1, dump_freq=4, calc_type='thermal_conductivity',
#                             n_perc=3, algo=2, cross0=False, units='lammps_metal', plot_info=True)
#
# print(f'{temp}K:', f'k = {k_avg:.3f}', f'+/- {k_std:.3f} [W/m.K]')


