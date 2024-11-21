import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from convert_units import *
from scipy.optimize import curve_fit
from mathstats_funcs import *
from scipy.signal import correlate
from scipy.integrate import cumulative_trapezoid
from ase.data import atomic_masses, atomic_numbers


def atomic_to_weight(atoms):
    """
        Converts from atomic to weight fraction. Receives a dict that contains each atomic type and its percentage
        molar concentration
    """

    total = sum([atom_concentration / atomic_masses[atomic_numbers[atom_type]]
                 for atom_type, atom_concentration in atoms.items()])
    weight_fraction = {}
    for atom_type, atom_concentration in atoms.items():
        weight_fraction[atom_type] = round(100 * (atom_concentration / atomic_masses[atomic_numbers[atom_type]]) / total, 2)

    return weight_fraction

# print(atomic_to_weight({'Ni': 78.16, 'Cr': 21.84})) # {'Ni': 80.16, 'Cr': 19.84}

def weight_to_atomic(atoms):
    """
        Convert from weight to atomic fraction. Receives a dict that contains each atomic type and its percentage
        molar concentration
    """
    total = sum([atom_concentration / atomic_masses[atomic_numbers[atom_type]]
                 for atom_type, atom_concentration in atoms.items()])

    atomic_fraction = {}
    for atom_type, atom_concentration in atoms.items():
        atomic_fraction[atom_type] = round(100 * (atom_concentration / atomic_masses[atomic_numbers[atom_type]]) / total, 2)

    return atomic_fraction

# print(weight_to_atomic({'Ni': 80, 'Cr': 20})) # {'Ni': 77.99, 'Cr': 22.01}


def calculate_mass(atomic_system, units='g'):
    """
        Calculates the mass of the atomic system in grams. Receives a dict containing the atomic species or type as
        the key and how many atoms of this type the system has as the value. For instance, {'Bi': 50, 'Ca': 20} means
        that there is 50 Bi atoms and 20 Ca atoms in our system.
    """
    N_a = 6.02214e23
    mass = sum([n_atoms/N_a * atomic_masses[atomic_numbers[atom_type]] for atom_type, n_atoms in atomic_system.items()])

    return mass if units == 'g' else g_to_kg(mass)


def number_of_mols(mass, atomic_system):
    """Finds how many mols there are in 1 gram of the atomic system"""
    N_a = 6.02214e23
    sys_natoms = sum(atomic_system.values())
    sys_mass = calculate_mass(atomic_system, units='g')
    natoms_per_gram = sys_natoms / sys_mass # [n_atoms/g]
    return mass * natoms_per_gram / N_a # [n_mols]


def calculate_density(atomic_system, box_volume):
    """
        Calculates the density in [g/cm³] of a system given the atoms and the volume of the cell/box.

        atoms => Dicitonary containing the atom type as keys and how many of this atom there is in the box
        volume => A list or array that contains the box volume in [Å³]
    """
    box_volume = np.array(box_volume)
    mass = calculate_mass(atomic_system, units='g')
    return mass / (box_volume * 1e-24) # [g/cm3]

# density = calculate_density(atomic_system={'F': 50 , 'Li':23 , 'Na': 6, 'K': 21}, box_volume=10.5**3)
# print(density)
# quit()

def calculate_molar_mass(atomic_system, n_molecules, units='g'):
    """
        Calculates the molar mass in g/mol

        OBS: It does not make sense calculate the molar mass of an ionic system with different ions. Only molecules or
             single atoms/ions
    """
    N_a = 6.02214e23
    m = calculate_mass(atomic_system, units=units)
    return m * N_a / n_molecules if units == 'g' else (m * N_a / n_molecules) / 1000


def calculate_number_density(atomic_system, volume):
    """
        Calculates the atomic number density (number of atoms per unit volume) in units of [1/Å³]
    """
    volume = np.array(volume)
    n_atoms = sum(atomic_system.values())

    return n_atoms / volume


def calculate_internal_energy(atomic_system, E0, EK, type='specific', units='J', num_molecules=None):
    """
        Calculates the internal energy of the system using U = E0 + EK in joules or eV or specific internal energy in
        [J/kg] or [J/g].

        OBS1: Apparently E0 is the energy at 0 K, so we add 'EK' to get the internal energy at the temperature that
              we are running.

        OBS2: When calculating the internal energy per unit mol we must divide by the total number of molecules in
              the system and not the total number of atoms. This is true even for ionic systems.
    """
    U = E0 + EK
    U = ev_to_joules(U) if units == 'J' else U
    if type == 'specific':
        m = calculate_mass(atomic_system, units='kg')
        U /= m
    elif type == 'molar':
        U /= sum(atomic_system.values())  # Finding energy per atom
        U /= num_molecules
        U *= 6.022e23

    return U



def calculate_enthalpy(atomic_system, E0, EK, p, V, type='specific', units='J', num_molecules=None):
    """Calculates the enthalpy using H = U + pV in [J], where 'U' is the internal energy, 'p' is the pressure and
       'V' is the volume. If we we wish to calculate specific enthalpy, h = H/m the units will be [J/kg]

       OBS1: When calculating the internal energy per unit mol we must divide by the total number of molecules in
             the system and not the total number of atoms. This is true even for ionic systems. For instance, when
             using the entalphy to find the specific heat capacity at constant pressure for a simulation of LiF-NaF
             whose simulation box contained 50 F, 30 Li, and 20 Na, to match experimental results I must divide the
             entalphy by 50 because there are 50 "molecules" in the system, and not 100. This is odd given that this
             is an ionic system, but that is how it should be reported. In theory both results are be correct, they
             just mean different things. they just mean different things. The first is per atom and the second per
             "molecule", and it seems like the community reports this value per "molecule"

    """
    p = kbar_to_pascal(p)
    V = angstrom3_to_meter3(V)
    U = calculate_internal_energy(atomic_system, E0, EK, units=units, type='full')
    H = (U + p*V)

    if type == 'specific':
        m = calculate_mass(atomic_system, units='kg')
        H /= m # entalphy per unit mass
    elif type == 'molar':
        # H /= sum(atomic_system.values()) # Finding the average entalphy per atom
        H /= num_molecules
        H *= 6.022e23 # entalphy per mol


    return H


def calculate_density_curve(func_to_fit, T, density_avg, n=10000):
    """

    """
    popt, pcov = curve_fit(func_to_fit, T, density_avg)
    perr = np.sqrt(np.diag(pcov))
    T_array = np.linspace(min(T), max(T), n)
    density = func_to_fit(T_array, *popt)

    return np.hstack((T_array[:,np.newaxis], density[:,np.newaxis])), popt, perr


def calculate_heat_capacity(h_func, T, h_avg, n=10000, order=2, dhdT_func=None):
    """
        Calculates the heat capacity at constant volume 'cv' or pressure 'cp', where cv = du/dT and cp = dh/dT, where
        'U' is the internal energy, 'H' is the enthalpy, and 'T' is the temperature. Both are simply the slope of
        the u vs T or the h vs T plot. This function returns a temperature array, the specific heat at each point of
        that array, and the optimized parameters for H_func.

        Args:
        T => Is the temperature array that tell us the temperature range of our simulation;
        h_avg => Array containing the averaged specific/molar entalphy at each temperature.
        h_func => Is the functional form that we gonna fit h_avg with respect to temperature
        dhdT_func => Is the analytical derivative of H_func.

        Returns:
            np.hstack(...) => A (n,2) matrix containing the points (T, cp)
            popt => The fitting coefficients of the entalphy vs temperature
            perr => The fitting error

        The heat capacity calculation can be done in two ways:
             1) The first approach use both a h_func and its anylitcal derivative dhdT_func. First, we fit h_func
                with 'T' and 'h_avg' to find the optimized paremeters for h_func. Then, the cp curve will simply be
                dhdT_func(T) using the fitted parameters of h_func. To plot we do ax.plot(T, dhdT_func(T)).
             2) The second is more general because we use only H_func and it yields virtually the same result. First,
                we fit H_funct with 'T' and 'h_avg' to find the optimized paremeters for h_func. Next, create a
                temperature array T_array = np.linspace(T.min(), T.max(), n) where 'n' tell us how many elements this
                array will have (more elements, better result). Then, we do np.diff(h_func(T)) and divide this result
                by dT = (T_array[1] - T_array[0]). The result will be an approximation of the derivative of the
                function. Thus, smaller the 'dT' (higher the 'n'), more closely it will match the real derivative.
                Finally, we can plot ax.plot(T_array[:-1] + dT/2, np.diff(h_func(T))/dt)
                OBS: We could also do a second order accurate derivative by doing h_func(T)[2:] - h_func(T)[:-2]

        OBS1: We can also use internal energy instead of entalphy to calculate 'c_p'. Our simulation is always at 0
              pressure (VASP default) so H = U + p*V simplifies H = U. The difference in the result if we use the
              instant pressure and volume against 0 pressure is actually very small (~1%)

        OBS2: It is always better to find specific heat from NpT simulations. In NVT there is the energy of the
              thermostat, but when I don't add that I get a small

        OBS3: In LAMMPS, we usually report 'c_v' and we calculate it by doing a linear fitting to the total energy
              versus temperature. For 'c_p' we must fit the entalphy. This method yields better results than looking
              at the standard deviation of the energy or entalphy. I know this approach is valid for liquids at
              high temperatures (>500 K). For solids I'm not 100% sure, but I think it is as long as all relevant
              interactions are being considered and the potential energy part of the total energy is accurate.
    """
    popt, pcov = curve_fit(h_func, T, h_avg)
    perr = np.sqrt(np.diag(pcov))
    T_array = np.linspace(T.min(), T.max(), n)
    h_array = h_func(T_array, *popt)
    cp_array = []

    if dhdT_func is None:
        dT = T_array[1] - T_array[0]
        if order == 1:
            cp_array = np.diff(h_array) / dT
            T_array = T_array[:-1] + dT/2
        elif order == 2:
            cp_array = (h_array[2:] - h_array[:-2]) / (2*dT)
            T_array = T_array[1:-1]
    else:
        # This assumes that h_func is a "full" polynomial, so the last popt which is a constant dies when we take dH/dT
        cp_array = dhdT_func(T_array, *popt[:-1])

    cp_array = np.round(cp_array, 2)


    return np.hstack((T_array[:,np.newaxis], cp_array[:,np.newaxis])), popt, perr


def calculate_heat_capacity2(atoms, E0, EK, T, used_frames):
    """ Calculate specific heat capacity using the alternative way C_v = var(E)/(k_b * T²) or C_p = var(H)/(k_b * T²).
        This should yield a very similar result to the traditional way which is taking the derivative.

        OBS1: It is very important that we take the variance of the total energy or entalphy 'U' or 'H', and not their
              specific or molar versions. If we take the variance of the specific energy or entalphy the result will be
              wrong (variance is not invariant under multiplication by a constant).

        OBS1: We can also use internal energy instead of entalphy to calculate 'c_p'. Our simulation is always at 0
              pressure (VASP default) so H = U + p*V simplifies H = U. The difference in the result if we use the
              instant pressure and volume against 0 pressure is actually very small (~1%)

        OBS2: Using this formula with 'E' from VASP yields a very wrong result. 'E' considers the kinetic (negligibe)
              and potential (very high) energies of the thermostat. Thus, E = F + EK + SP + SK. Turns out that if
              we consider the thermostat we get very wrong specific heat capacity values (it will be very high). The
              best is to use E0 + EK as the total energy, and since in NpT SK = SP = 0, it is more safe to use E0 + EK
              in NpT. Using E0 + EK in NVT seems to underestimate the heat capacity.
    """
    U = calculate_internal_energy(atoms, E0, EK, type='full') # [J]
    U_var = np.var(U[:, -used_frames:], axis=1) # [J²]
    k_b = 1.38e-23 # [J/K]
    m = calculate_mass(atoms, units='kg') # [kg]
    return U_var / (k_b * T**2) * 1/m # [J/kg.K]


def calculate_thermal_expansion(T, rho, drho_dT=None):
    """
        Calculates the volumetric thermal expansion through the formula alpha = -1/ρ (dρ/dT) where dρ/dT is calculated
        at a fixed pressure.

        rho => An array containing the density curve as a function of temperature
        T => An array containing temperature points

        Returns:
            np.hstack(...) => A (n,2) matrix containing the pair (T, alpha)
            popt => The fitting parameters of a line passing through (T, alpha)

    """
    rho = np.array(rho)
    T = np.array(T)

    if drho_dT is None:
        dT = T[1] - T[0]
        drho_dT = (rho[2:] - rho[:-2]) / (2*dT)
        T = T[1:-1]
        rho = rho[1:-1]
    alpha = -1/rho * drho_dT
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, T, alpha)

    return np.hstack((T[:,np.newaxis], alpha[:,np.newaxis])), popt


def calculate_msd(traj_file_path, spacing='linear'):
    """Calcualtes the mean squared displacement MSD(t) = 〈|r(t) - r(0)|²〉

    Args:
        traj_file_path (str): Path to the trajectory file.
        spacing (str): How the time points where the MSD is calculated are chosen, can be linear or log.

    Returns:
        (ndarray): (M,2) array where the first column is the time points where the MSD was calculated and the second
                   column is the MSD.
    """

    pass



def calculate_diffusion_coefficient(msd, first_idx=0, last_idx=None, dim=3):
    """Calculates the diffusion coefficient 'D' in [cm²/s] through Einstein's relation D = 1/6 lim t->∞ d(MSD)/dt.

    The diffusion coefficient is given by the slope of the mean squared displacement MSD(t) = 〈|r(t) - r(0)|²〉 as the
    time goes to infinity (which is a straight line y=ax) divided by 2*d where 'd' is the dimension of the system.

    OBS1: Note that MSD should always start from zero (because at the initial time the displcament is clearly zero),
          therefore we can drop the 'b' coefficient in the linear fit

    Args:
        msd (ndarray): A numpy array of shape (N,2) where the first column is the times that we calculated the msd(t)
                       and the second column the MSD at these times. Thus, this array represents MSD(t).
        first_idx (int): Index of the first time point to fit the straight line representing the diffusion coefficient.
        last_idx (int): Index of the last time point to fit the straight line representing the diffusion coefficient.
        dim (int): Dimension of the system.

    Returns:
        (float, float)

    """
    if last_idx is None:
        last_idx = msd.shape[0]-1
    popt, pcov = curve_fit(lambda x, a: a*x, msd[first_idx:last_idx+1, 0], msd[first_idx:last_idx+1, 1])
    msd_slope = popt[0]
    D = (msd_slope / 2*dim)
    D /= 10 # Converting from [Å²/fs] to [cm²/s]

    return D, msd_slope


def fit_diffusion_curve(D, T, n=1000):
    """Fits the diffusion coefficients to an exponential.

    D => An array (n,) containing the diffusion coefficient at temperature T
    T => An array (n,) contining temperature points

    Return:
        np.hstack(...) => The diffusion as a function of temperature or the pair (T, diffusion_coeff)
        D0 => The diffusio coefficient as T goes to infinity
        E_a => The activation energy for the diffusion process in [eV]

    OBS: We usually plot it as D x 1000/T but with the y-axis in log scale

    """
    T_array = np.linspace(T.min(), T.max(), n)

    D_func = lambda T, D0, E_a: D0 * np.exp(-E_a / (8.61733e-5 * T))
    popt, pcov = curve_fit(D_func, T, D)
    D0, E_a = popt

    return np.hstack((T_array[:,np.newaxis], D_func(T_array, *popt)[:,np.newaxis])), D0, E_a


def calculate_relaxation_time(kvecs, fs_kt,  method=2, silent=True, plot_fitting=False):
    """Finds the relaxation time given a intermediate scattering function.

     The idea is to find value where F_s(k,t) = F_s(k,0)/e (or F_s(k,t) = 1/e ~ 0.36 if F_s(k,0) = 1) for each k-vector.

        Method 1: Finds the closest time point where F_s(k,t) = 1/e and defines it as the relaxation time. This is very
                  simple and crude method. If we have a lot of time points then it will be a very good approximation,
                  otherwise it won't be so accurate.

        Method 2: Now for each k-vector we fit its F(k',t) to a spline up to the first negative time index and find the
                  time point on this spline where it drops to 0.36 of its value at t=0.

        Args:
            kvecs (np.array) => (N,) array containing the k-vectors used.
            fs_kt (np.array) => (n_bins, N+1) matrix representing the intermediate scattering function. The first
                                column fs_kt[:,0] represents the sampled simulation times and the other 'N' columns
                                represents the fs_kt for each k-vector.
            silent (bool) => Controls whether to print or not the value of F(k,t) where the relaxation time was
                             calculated.
            plot_fitting (bool) => Controls whether to show the fitting of the spline/exponential w.r.t F_s(k,t).

        Return:
            tau (np.array) => (N,) array containing the relaxation time for each mode or k-vector of F_s(k,t).


        OBS1: The diffusion coefficient is simply the slope of 1/tau vs |k|²:
              popt, pcov = curve_fit(lambda x, a: a * x, kvecs**2, 1/tau)
              D = popt[0] / 10 # Converting from [Å²/fs] to [cm²/s]
    """
    tau = np.zeros_like(kvecs)

    ## Method 1:
    if method == 1:
        for j in range(kvecs.shape[0]):
            closest_idx, closest_value = closest_to_n(fs_kt[:, j + 1], 1 / np.e)
            tau[j] = fs_kt[closest_idx, 0]

            if not silent:
                print(closest_value)

    ## Method 2:
    elif method == 2:
        for j in range(kvecs.shape[0]):
            # Finding the first point where F(k,t) is negative
            negative_indices = np.flatnonzero(fs_kt[:,j+1] < 0)
            if len(negative_indices) == 0:
                max_time_idx = fs_kt[:,j+1].shape[0]
            else:
                max_time_idx = negative_indices[0]

            # Fitting F_s(k,t) to an exponential
            # func = lambda t, tau1, tau2, f1, f2, beta: f1*np.exp(-t / tau1) + f2*np.exp((-t / tau1)**beta)
            # func = lambda t, tau, beta: np.exp((-t / tau1)**beta)
            # popt, pcov = curve_fit(func, fs_kt[:max_time, 0], fs_kt[:max_time, j + 1], maxfev=10000)
            # tau[j] = popt[0]

            # Fitting F_s(k[j],t) to a spline
            spline = UnivariateSpline(fs_kt[:max_time_idx, 0], fs_kt[:max_time_idx, j+1], k=3, s=0)
            time = np.linspace(0, fs_kt[max_time_idx-1, 0], 100000)
            closest_idx, closest_value = closest_to_n(spline(time), spline(0)/np.e) # Or closest_to_n(spline(time), 1/np.e)
            tau[j] = time[closest_idx]

            if not silent:
                print(closest_value)

            if plot_fitting:
                fig, ax = plt.subplots()
                ax.plot(fs_kt[:max_time_idx, 0], fs_kt[:max_time_idx, j+1], color='navy', label='Truth')
                # ax.plot(fs_kt[:max_time, 0], func(fs_kt[:max_time, 0], *popt), color='orangered', ls='--', label='Exponential Fit')
                ax.plot(fs_kt[:max_time_idx, 0], spline(fs_kt[:max_time_idx, 0]), color='forestgreen', ls='--', label='Spline')
                ax.legend()
                plt.show()

    return tau


def calculate_diffusion_vacf(v_acf):
    """
        Calculates the diffusion coefficient form the integration of the velocity autocorrelation. This is simply the
        Green-Kubo relation for diffusion.

        Args:
            v_acf (np.array) => The velocity autocorrelation

        Returns:
            (float) => Diffusion coefficient

        OBS1: The integration step has to be the same as the time difference between two consecutive points where the
              v_acf was evaluated. In other words, has to be set to the 'time_interval' flag in LiquidLib
    """
    dt = v_acf[1,0] - v_acf[0,0] # Note that this is simply the 'time_interval' in LiquidLib
    last_idx = np.flatnonzero(v_acf[:, 1] < 0)[0]

    return np.trapz(v_acf[:last_idx, 1], dx=dt) / 3


def calculate_coordination_number(r, rdf, n_count, box_volume, r1=None, search_rdf_r1=False, verbose=True):
    """Calculates the first coordination number through integration of the radial distribution function g(r).

        The formula used is n1 = ∫g(r) ρ4πr²dr from r=[r0, r1], where r0=0 and 'r1' is the is first local minima or
        valley. This integrates the first peak of the pair distribution function up to 'r1'. 'ρ4πr²dr' is a normalizing
        factor where ρ=N/V is the number density of the particles that we are counting and 4πr²dr is the volume of a
        infinitesimal spherical shell.

        Args:
            r (ndarray): The r's that the pair distribution function was calculated
            rdf (ndarray): Radial distribution function at 'r'
            n_count (int): Number of the atoms 'B' that we wish to count when 'A' is taken as reference.
            box_volume (float) => Volume of the simulation box
            r1 (int) => If not None the coordination number will be integrated up to this point.
            search_rdf_r1 (bool): If True we look for the g(r) first minima around the spline first minima
            verbose (bool): If True prints the location of the first minima, i.e. up to where we integrated the g(r),
                            and plots the fitted spline.

        Returns:
            (float): The first coordination number 'n1'

        OBS1: If we have the pair distribution function of two atoms A-B, the coordination number for 'A' will be
              different from the coordination number for 'B' even though the g(r) of A-B is the same as B-A. The
              number of 'A' and 'B' atoms in the box will probably be different, thus yielding different number
              densities and thus coordination numbers. This is intuitive because the number of 'B' atoms around 'A'
              is not necessarily the same as the number of 'A' atoms surrounding 'B'.

    """

    if r1 is None:
        rdf_spline = UnivariateSpline(r, rdf, k=3, s=0.08)
        r_spline = np.linspace(r.min(), r.max(), 10000)
        idxs_r_mins_spline, _ = find_local_mins(rdf_spline(r_spline))
        r_candidates = r_spline[idxs_r_mins_spline]

        # We return the first minimum after the first peak. Doing so will avoid finding noisy local minimas before the
        # first peak.
        r_peak = r[rdf.argmax()]
        r1_spline = r_candidates[r_candidates > r_peak][0] # Get first minima beyond the first peak
        idx_r1, r1 = closest_to_n(r, r1_spline)

        # Sometimes 'idx_r1' doesn't correspond to the first minima of the g(r). Thus, I can perform a search and look
        # at 'search_depth' neighbors to the left and right of g(r1) to see if there is a smaller value. Then, I keep
        # the index of the smallest value as idx_r1.
        if search_rdf_r1:
            search_window = 2
            idx_r1 += np.argmin(rdf[idx_r1-search_window:idx_r1+search_window+1]) - search_window

        if verbose:
            print(f'The first local minimum of the spline g(r) was found at r = {r1_spline:.2f} A, closest point in g(r) is r = {r1:.2f} A')
            x = np.linspace(0, max(r), 10000)
            fig, ax = plt.subplots()
            ax.plot(x, rdf_spline(x), color='navy', lw=1.5)
            ax.axvline(r1_spline, ls='--', color='orangered', label='r1 spline')
            ax.axvline(r[idx_r1], ls='--', color='forestgreen', label='r1 g(r)')
            ax.legend(loc='best')
            fig.show()

    else:
        idx_r1, _ = closest_to_n(r, r1)

    dr = np.mean(np.diff(r))
    r_midpoint = (r[:-1] + r[1:]) / 2
    rdf_midpoint = (rdf[:-1] + rdf[1:]) / 2
    rho = n_count / box_volume

    integral = np.sum((rdf_midpoint * r_midpoint**2 * dr)[:idx_r1])

    return 4 * np.pi * rho * integral


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
                       have a very small simulation (AIMD) it is better with cross0=True because the noise is very
                       large, i.e. the oscillation of the ACF around zero has a large amplitude. On the other hand,
                       if we have a large simulation (LAMMPS), then integrating the entire autocorrelation, i.e.
                       cross0=False, with a small n_perc ~5-10 (depending on how long the simulation is) is better.
        units (str): Units of the simulation that calculated `A`, can be either `lammps_metal` or `vasp`.

    Returns:
        psi (float): Generalizes susceptibility (e.g. thermal conductivity or viscosity)
        cumintegral_acf (ndarray): Array containing the values that `psi` can assume depending on where to truncate
                                   the integral.
        avg_acf (ndarray): Average autocorrelation

    OBS1: Since the average shear pressure is noisy, we usually do a moving average to smooth it. This avoids
          crossing the zero line and stop couting the viscosity too soon in case of a more violent oscillation
          of the average shear pressure before reaching zero.
    OBS2: If we calculate the viscosity using the average acf of Pxy, Pyz, and Pzx we get consistently smaller
          results compared to calculating the viscosity for each off-diagonal element and then at the end find
          the average of the viscosities. This is because in the first approach the autocorrelation will go to
          zero faster, so its integral will be smaller. This happens because when we take the mean of the three
          acfs we are essentially "killing" some of the correlations. For instance:
          Method 1:  [2.38850899 1.8175449  1.45543336 1.24776136 1.2217034  1.06035976]
                     [0.50880727 0.27966025 0.14824042 0.21355852 0.2486301  0.16956756]
          Method 2:  [2.44147078 1.84896881 1.53931232 1.33804634 1.24311352 1.08987762]
                     [0.47576769 0.26144727 0.17663637 0.22798584 0.20176884 0.19405260]
    OBS3: We take the cumulative integral to construct a nice plot of the viscosity converging to a final value.
          The viscosity is the limit where t->∞ and thus eta[-1]. However, this integral will oscillate around zero
          because the average shear pressure will decorrelate and oscillate around its mean 0 (I center the array
          around its mean in the acf function). Thus, one solution is to consider the value of the integral only
          until the first point where the autocorrelation crosses the zero. We usually take a moving average of
          the shear pressure to smooth its oscillations to avoid underestimating the viscosity in some cases (the
          oscillation would cross the zero line before the "trend").
    OBS4: For thermal conductivity calculation of liquids it seems like we must include the kinetic energy in the
          per atom stress tensor command. I performed two runs to test it, and when the kinetic energy is added the
          thermal conductivity is ~60% of what it was when considering only the virial. Thus, the effect of adding
          the kinetic part of the per atom stress tensor is that the heat flux decorrelates with itself faster.
    OBS5: We shuld use n_perc=1-3 becuase the decorrelation time is much smaller than the total sampled time.

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
            if units == 'lj':
                psi *= V / (1 * T)  # k_b is 1 here
            else:
                psi *= V / (8.61733e-5 * T) # V/k_b*T is the constant for viscosity
                if units == 'vasp':
                    psi *= (1e-15 / 160217663400 * 1000 * 1e16) # [mPa.s] (fs to s; A^3/ev to 1/Pa; Pa to mPa; kBar^2 to Pa^2)
                elif units == 'lammps_metal':
                    psi *= (1e-12 / 160217663400 * 1000 * 1e10) * 1e-3 # [mPa.s] (ps to s; A^3/ev to 1/Pa; Pa to mPa; Bar^2 to Pa^2)
                    # The 1e-3 is to convert dtau from [fs] to [ps]

        elif calc_type == 'thermal_conductivity':
            psi *= V / (8.61733e-5 * T ** 2) # V/k_b*T^2 is the constant for thermal conductivity
            if units == 'lammps_metal':
                psi *= (1.60218e-19 / 1e-12) / 1e-10 * 1e-3 # [(eV/ps)/A.K * fs] to [(J/s)/m.K]

    # Compute the susceptibility from the average autocorrelation of each row in `A`
    elif algo == 2:
        acf_cumintegral = cumulative_trapezoid(avg_acf, dx=dtau, initial=0)

        if calc_type == 'viscosity':
            if units == 'lj':
                acf_cumintegral *= V / (1 * T)
            else:
                acf_cumintegral *= V / (8.61733e-5 * T)
                if units == 'vasp':
                    acf_cumintegral *= (1e-15 / 160217663400 * 1000 * 1e16)
                elif units == 'lammps_metal':
                    acf_cumintegral *= (1e-12 / 160217663400 * 1000 * 1e10) * 1e-3 # The 1e-3 is to convert dtau from [fs] to [ps]

        elif calc_type == 'thermal_conductivity':
            acf_cumintegral *= V / (8.61733e-5 * T**2)
            if units == 'lammps_metal':
                acf_cumintegral *= (1.60218e-19/1e-12) / 1e-10 * 1e-3

        idx_truncate = np.flatnonzero(avg_acf < 0)[0] if cross0 else -1
        psi = acf_cumintegral[idx_truncate]

    return psi, acf_cumintegral, avg_acf

# x = np.linspace(-100,100,100)
# var = 10000
# gauss = lambda x, var: 1/(var*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*var))
# fig, ax = plt.subplots()
# ax.plot(x, gauss(x, var), color='navy')
# ax.plot(x, gauss(x, 5), color='red')
# ax.plot(x, gauss(x, 10), color='orange')
# ax.plot(x, gauss(x, 100), color='black')
# plt.show()

def bootstrap_gk(divisor_list, n_samples, A, box_vol, temperature, dt, dump_freq, calc_type, n_perc=5, algo=2,
                 cross0=True, moving_avg_window=1, units='vasp', plot_info=False):
    """Bootstrap the trajectory window over a long trajectory and calculate a generalized susceptibility using GK.

       OBS1: In VASP often `A` is the result of a moving average with 3-4 frames.
       OBS2: We should often use n_perc=5 because the decorrelation time is much smaller than the total simulation time.
    """
    A_mov_avg = moving_average(A, moving_avg_window)
    n_frames = A_mov_avg.shape[1]
    psi_matrix = np.zeros((len(divisor_list), n_samples), dtype=float)
    cumpsi_matrix = []

    for i, divisor in enumerate(divisor_list):
        sample_size = int(n_frames // divisor)
        for j in range(n_samples):
            starting_frame = np.random.randint(0, n_frames - sample_size)
            psi_matrix[i,j], cum_psi, acf = green_kubo(A_mov_avg[:, starting_frame:(starting_frame+sample_size)],
                                                       box_vol, temperature, dt, dump_freq, calc_type,
                                                       n_perc=n_perc, algo=algo, cross0=cross0, units=units)
            cumpsi_matrix.append(cum_psi)
            if plot_info:
                if j % int(n_samples/2) == 0:
                    fig, ax = plt.subplots(2,1)
                    ax[0].plot(acf, color='navy')
                    ax[0].axhline(0, color='black', ls='--')
                    ax[0].set_xlabel('Time Delay [units of dt*dump_freq]')
                    ax[0].set_ylabel('Average ACF')
                    ax[1].plot(cum_psi, color='orangered')
                    ax[1].set_xlabel('Integral Truncation Index')
                    ax[1].set_ylabel('Susceptibility')
                    if cross0:
                        cross0_idx = np.flatnonzero(acf < 0)[0]
                        ax[0].scatter(cross0_idx, acf[cross0_idx], color='red', s=15)
                        ax[1].axvline(cross0_idx, color='black', ls='--')
                        ax[1].scatter(cross0_idx, cum_psi[cross0_idx], color='forestgreen', s=15)
                    plt.show()

    return np.mean(psi_matrix).round(3), np.std(psi_matrix).round(3), np.array(cumpsi_matrix).mean(axis=0)


def calculate_lattice_constant(volumes, energies, at_number=1, eos_type='birchmurnaghan', plot_curve=True, cubic=False):
    """Calculate the equilibrum lattice. To do that, we perform several VASP runs with IBRION=-1 using different
       lattice parameters. Then, we extract the final converged energy of each run and fit an equation of state to
       Energy vs Volume. The equilibrum volume is then the volume where the energy is a minimum.

       Args:
           at_number (int): Volume scaling factor, which depends on how many atoms there was in our VASP cell. For
                            instance, for FCC lattices we can use just a primitive unit cell with a single atom or the
                            conventional unit cell with 4 atoms. If we choose the former, we need to multiply the
                            volume by 4 before taking the cubic root since the volume that VASP returns is the volume
                            of a single atom.

       Returns:
            (V0, E0, B0) => Tuple containing the equilibrium volume, equilibrium energy and bulk modulus.
    """
    from ase.eos import EquationOfState
    from ase.units import GPa

    eos = EquationOfState(volumes, energies, eos=eos_type)
    V0, E0, B0 = eos.fit()

    print(f'V0: {V0:.2f} Angs^3')   # Equilibrium Volume
    print(f'E0: {E0:.2f} eV')       # Minimum energy
    print(f'B0: {B0/GPa:.2f} GPa')  # Bulk Modulus

    if cubic:
        print(f'a: {(at_number*V0)**(1/3):.3f} Angs')

    if plot_curve:
        eos.plot()
        plt.show()

    return V0, E0, B0


def calculate_bulk_modulus(stiffness_tensor, type='Voigt', units='kbar'):
    """Calculates the bulk modulus from the relastic stiffness tensor.

       Ref: J. Am. Ceram. Soc., 96 [12] 3891–3900 (2013)

        Args:
            stiffness_tensor (np.array): A (3,3) symmetric matrix or the full (6,6) matrix, but only the first (3,3)
                                         quadrant is used for bulk modulus.
            type (str): Whether to calculate the standard Voigt, Reuss, or Hill bulk modulus, which is the average
                        value of the Voigt and Reuss bulk moduli.
            units (str): Units of the input sitiffness tensor.

        Returns:
            float: Bulk modulus in units of GPa

        OBS1: I often organize the input like this:
        D = [XX-XX, YY-YY, ZZ-ZZ, XY-XY, YZ-YZ, ZX-ZX]
        S = [XX-YY, XX-ZZ, YY-ZZ, XX-XY, XX-YZ, XX-ZX, YY-XY, YY-YZ, YY-ZX, ZZ-XY, ZZ-YZ, ZZ-ZX, XY-YZ, XY-ZX, YZ-ZX]
        stiffness_matrix =  np.array([[ D[0], S[0],  S[1],  S[3],  S[4],  S[5] ],
                                      [ S[0], D[1],  S[2],  S[6],  S[7],  S[8] ],
                                      [ S[1], S[2],  D[2],  S[9], S[10], S[11] ],
                                      [ S[3], S[6],  S[9],  D[3], S[12], S[13] ],
                                      [ S[4], S[7], S[10], S[12],  D[4], S[14] ],
                                      [ S[5], S[8], S[11], S[13],  S[14], D[5] ],])
        But since we only use the first three elements of 'S' in the calculations, we can ignore the other elements
        and set them to zero (they will be basically zero anyway, and in theory they should be zero).
        full_stiffness_matrix =  np.array([[ D[0], S[0], S[1],    0,     0,  0 ],
                                   [ S[0], D[1], S[2],    0,     0,  0 ],
                                   [ S[1], S[2], D[2],    0,     0,  0 ],
                                   [   0,     0,     0, D[3],    0,  0 ],
                                   [   0,     0,     0,    0, D[4],  0 ],
                                   [   0,     0,     0,    0,    0, D[5] ],
                                   ])
    """
    C = np.copy(stiffness_tensor)
    if units == 'kbar':
        C /= 10  # kBar to GPa

    if type == 'Voigt':
        B_v = 1/9 * (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[0,2] + C[1,2]))
        return B_v

    elif type == 'Reuss':
        S = np.linalg.inv(C) # Compliance matrix
        B_r = 1 / (S[0,0] + S[1,1] + S[2,2] + 2*(S[0,1] + S[0,2] + S[1,2]))
        return B_r

    elif type == 'Hill':
        S = np.linalg.inv(C)  # Compliance matrix
        B_r = 1 / (S[0,0] + S[1,1] + S[2,2] + 2*(S[0,1] + S[0,2] + S[1,2]))
        B_v = 1/9 * (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[0,2] + C[1,2]))
        B_h = (B_v + B_r) / 2
        return B_h

    else:
        raise Exception('Please choose either Voigt, Reuss, or Hill as type')


def calculate_shear_modulus(stiffness_tensor, type='Voigt', units='kbar'):
    """Calculate the shear modulus G from the full 6x6 stiffness tensor

       Ref: J. Am. Ceram. Soc., 96 [12] 3891–3900 (2013)

        Args:
            stiffness_tensor (np.array): A (3,3) symmetric matrix or the full (6,6) matrix, but only the first (3,3)
                                         quadrant is used for bulk modulus.
            type (str): Whether to calculate the standard Voigt, Reuss, or Hill bulk modulus, which is the average
                        value of the Voigt and Reuss bulk moduli.
            units (str): Units of the input sitiffness tensor.

        Returns:
            float: Bulk modulus in units of GPa
    """
    C = np.copy(stiffness_tensor)
    if units == 'kbar':
        C /= 10  # kBar to GPa

    if type == 'Voigt':
        G_v = 1/15 * (C[0,0] + C[1,1] + C[2,2] - C[0,1] - C[0,2] - C[1,2]) + 1/5 * (C[3,3] + C[4,4] + C[5,5])
        return G_v

    elif type == 'Reuss':
        S = np.linalg.inv(C)
        G_r = 15 / ( 4*(S[0,0] + S[1,1] + S[2,2] - S[0,1] - S[0,2] - S[1,2]) + 3*(S[3,3] + S[4,4] + S[5,5]) )
        return G_r

    elif type == 'Hill':
        G_v = 1/15 * (C[0,0] + C[1,1] + C[2,2] - C[0,1] - C[0,2] - C[1,2]) + 1/5 * (C[3,3] + C[4,4] + C[5,5])
        S = np.linalg.inv(C)
        G_r = 15 / ( 4*(S[0,0] + S[1,1] + S[2,2] - S[0,1] - S[0,2] - S[1,2]) + 3*(S[3,3] + S[4,4] + S[5,5]) )
        return (G_v + G_r) / 2

    else:
        raise Exception('Please choose either Voigt, Reuss or Hill as type')


def calculate_young_modulus(stiffness_tensor):
    """Calculates the Young modulus `E` from the sitffness tensor

       Ref: J. Am. Ceram. Soc., 96 [12] 3891–3900 (2013)

       Args:
            stiffness_tensor (np.array): full (6,6) stiffness tensor matrix

       Returns:
           float: Young's modulus `E`

    """
    B_h = calculate_bulk_modulus(stiffness_tensor, type='Hill')
    G_h = calculate_shear_modulus(stiffness_tensor, type='Hill')

    return (9*B_h*G_h) / (3*B_h + G_h)


def calculate_poisson_ratio(stiffness_tensor):
    """Calculates the Poisson's ratio `v` from the sitffness tensor

       Ref: J. Am. Ceram. Soc., 96 [12] 3891–3900 (2013)

       Args:
            stiffness_tensor (np.array): full (6,6) stiffness tensor matrix

       Returns:
           float: Poisson's ratio `v`
    """

    B_h = calculate_bulk_modulus(stiffness_tensor, type='Hill')
    G_h = calculate_shear_modulus(stiffness_tensor, type='Hill')

    return (3*B_h - 2*G_h) / (6*B_h + 2*G_h)
