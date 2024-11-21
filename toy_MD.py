import numpy as np
from physchem_funcs import calculate_mass
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.spatial.distance import pdist, squareform

def box_dimension_estimator(atomic_system, density, density_unit='g/cm3', decimal_digits=6):
    """First approximation to the cubic box/cell size given a density and a system.

    The idea is very simple, since ρ = m_system / V_box, then V_box = m_system / ρ.

    Args:
        atomic_system (dict): A dictionary containing the atom type and the number of the corresponding atoms that we
                              wish to simulate, e.g. atoms={'Na': 32, 'Cl': 32}
        density (float): The density of the molecule at the temperature that we wish to run MD
        density_unit (str): The units we are using. Can be either 'g/cm3' or 'kg/m3'
        decimal_digits (int): Number of decimal digits of the box length.

    Returns:
        (float): estimated box length [Angstrom]
    """
    if density_unit == 'kg/m3':
        density /= 10**3 # Turns density to [g/cm^3]

    box_volume = calculate_mass(atomic_system) / density # [cm^3]
    L = box_volume**(1/3) * 1e8 # [Angstrom]

    return np.around(L, decimal_digits)


def create_neutral_atomic_system(atom_charges, target_num_atoms):
    """Create a random neutral atomic system given the atoms in atom_charges.

    Args:
      atom_charges (dict): Contains the atom type as key and its valence as values
      target_num_atoms (int): Targeted number of atoms to have in the system

    Returns:
      (dict): A dictionary where the keys are the atom types in the system and the corresponding values the number of
              each atom type in the system.

    # OBS: Sometimes it will be impossible to have the exact number 'taget_num_atoms' in the system due to the random
    #      way in which the atomic system was created. This isn't a problem because the box volume is adjusted based
    #      on the atomic system to yield the exact density that we want (which is what matters).

    """
    total_charge = 0
    atom_types = set(atom_charges.keys())
    atomic_system = {el: 0 for el in atom_types}

    anions = list(filter(lambda an: atom_charges[an] < 0, atom_types))
    cations = list(atom_types.difference(anions))
    atom_types = list(atom_types)

    max_anion_charge = min({atom_charges[anion] for anion in anions}) # anion is negative, so we invert max/min
    max_cation_charge = max({atom_charges[cation] for cation in cations})

    while True:
        # Picking a random atom
        if total_charge == 0:
            atom = random.choice(atom_types)
        else:
            atom = random.choice(anions) if total_charge > 0 else random.choice(cations)
        atomic_system[atom] += 1
        total_charge += atom_charges[atom]

        # We want the charges to always be at most as negative as the most positive cation or as positive as the most
        # negative anion. The reason for this is because when the number of atoms in atomic_system is close to the
        # targeted number of atoms, we can always fix the total charge without having to put a lot of atoms.
        if total_charge > 0 and total_charge > abs(max_anion_charge):
            # Recall that anion charges are negative, so min() charge is the most negative one.
            while total_charge > abs(max_anion_charge):
                balance_atom = random.choice(anions)
                atomic_system[balance_atom] += 1
                total_charge += atom_charges[balance_atom]

        elif total_charge < 0 and abs(total_charge) > max_cation_charge:
            while abs(total_charge) > max_cation_charge:
                balance_atom = random.choice(cations)
                atomic_system[balance_atom] += 1
                total_charge += atom_charges[balance_atom]

        # Checking if the number of atoms in the system is close to the targeted number of atoms.
        current_num_atoms = sum(atomic_system.values())
        if current_num_atoms >= (target_num_atoms - 1):
            if total_charge != 0:
                candidates = list(filter(lambda at: atom_charges[at] == -total_charge, atom_types))
                # If the anion or cation charges are not "continuous" e.g. {-1,-3} then there is a chance that the
                # total charge will be +2 and no single atom will neutralize it, so candidates will be empty.
                if candidates:
                    balance_atom = random.choice(candidates)
                    atomic_system[balance_atom] += 1
                    total_charge += atom_charges[balance_atom]

        if total_charge == 0 and current_num_atoms >= (target_num_atoms-1):
            break

    return atomic_system


def generate_coordinates(n_atoms, box_length, min_distance=2.0, max_iter=30, max_tries=25, decimal_digits=6,
                         scale_coords=True, save_txt=False):
    """
    Generate random atomic coordinates inside a cubic box respecting the min_distance constraint.

    Args:
        n_atoms (int) => Number of atom in the system and thus the number of coordinates that will be generated.
        box_length (float) => The parameter of the cubic box where the simulation takes place, where the volume of the
                              box is equal to (box_length)**3
        min_distance (float): Minimum absolute distance between any two atoms in Angstroms.
        max_iter (int): Maximum number of tries to place an atom in a specific grid.
        max_tries (int): Maximum number of tries to rerun the algorithm and try to fit the atoms that are missing to
                         reach the desired number of atoms.
        decimal_digits (int) => Controls the number of decimal digits of the coordinates
        scale_coords (bool): Scales distances by the box length such that all coordinates are now between [0,1].
        save_txt (bool) => If set to 'True', a text file will be created with the coordinates

    Returns:
        (np.array): A (N,3) np.array where 'N' is the total number of atoms in the system.
    """

    N = int(box_length // min_distance)
    grid_spacing = box_length / N
    grid = np.arange(0, box_length, grid_spacing)

    if N ** 3 < n_atoms:
        raise ValueError(
            f'Density is too high, there is no way or it will take a very long time to place {n_atoms} atoms '
            f'in a box length of {box_length:.2f} given that each atom must be {min_distance} apart')

    # Creating a grid that when indexed by (i,j,k) returns the base coordinate at that point
    mesh_grid = np.array(np.meshgrid(grid, grid, grid)).T
    # Swapping axes such that meshgrid[a,b,c] corresponds to gridspacing * [a,b,c]
    mesh_grid = np.swapaxes(mesh_grid, 0, 2)
    mesh_grid = np.swapaxes(mesh_grid, 0, 1)
    occupied_mesh_grid = np.zeros_like(mesh_grid)  # Will store the absolute position of each atom in the grid

    all_indices = np.indices((N, N, N)).T.reshape(-1, 3)
    np.random.shuffle(all_indices)
    available_indices = list(map(tuple, all_indices))
    occupied_indices = []

    aux_neighbors_idx = list(product((0, -1, 1), (0, -1, 1), (0, -1, 1)))
    del aux_neighbors_idx[0]  # Removing (0,0,0) since it is useless
    aux_neighbors_idx = np.array(aux_neighbors_idx)

    rng = np.random.default_rng()
    total_tries = 0

    while len(occupied_indices) < n_atoms:
        if not available_indices:
            # Refill available_indices with the indices that are not occupied after a full sweep over all indices.
            available_indices = set(map(tuple, all_indices.tolist())).difference(occupied_indices)
            total_tries += 1
            # This increase of max_iter speeds the code because we are now trying harder to fit the remaining atoms
            # in the box for a given index, so we don't need to calculate the neighbors and the whole cycle again.
            max_iter += 2
            if total_tries > max_tries:
                raise Exception(f'Exceeded maximum iterations, only {len(occupied_indices)} were placed')

        idx = available_indices.pop()
        neighbor_list = np.array(idx) + aux_neighbors_idx

        # Applying PBC to the neighbors
        neighbor_list[neighbor_list == -1] = N - 1
        neighbor_list[neighbor_list == N] = 0

        # Getting only the coordinates of the occupied neighbors from the neighbor list since we only need to check
        # distance for them (there is no reason to check distance of an empty grid).
        i_neighbors, j_neighbors, k_neighbors = neighbor_list[:, 0], neighbor_list[:, 1], neighbor_list[:, 2]
        neighbors_coords = occupied_mesh_grid[tuple(i_neighbors), tuple(j_neighbors), tuple(k_neighbors)]
        occupied_neighbors_coords = neighbors_coords[np.any(neighbors_coords, axis=1)] # Empty grid have coords (0,0,0)

        for _ in range(max_iter):
            candidate_coord = mesh_grid[idx] + rng.uniform(0, grid_spacing, size=3)
            distances = np.linalg.norm(to_mic(candidate_coord - occupied_neighbors_coords, box_length), axis=1)
            if np.all(distances > min_distance):
                occupied_mesh_grid[idx] = candidate_coord
                occupied_indices.append(tuple(idx))
                break

    occupied_mesh_grid = occupied_mesh_grid.reshape(-1, 3)
    coords = occupied_mesh_grid[np.any(occupied_mesh_grid, axis=1)]
    np.random.shuffle(coords)

    if scale_coords:
        coords /= box_length

    if save_txt:
        np.savetxt(f'{n_atoms}_positions.txt', coords, fmt=f'%.{decimal_digits}f')

    return coords


def wrap_coords(r, box_length, zero_origin=False):
    """
    Wrap positions inside simulation box

    Args:
      r (np.array): the positions of each atom, which is a coordinate matrix (n_atoms, 3)
      box_length (float): box side length
      zero_origin (bool): If True indicates that the box goes from [0, L] in all directions instead of [-L/2, L/2]
    Returns:
      np.array: r in box
    """
    # shifting coordinates to [0,L]
    if not zero_origin:
        r = r + box_length / 2

    # Unwrapping
    # r = np.where(abs(r) > box_length, r % box_length, r)
    r %= box_length

    return r - box_length / 2 if not zero_origin else r


def to_mic(r_ij, box_length):
    """Impose minimum image convention on the vector r_ij = r_i - r containing the distances between atom 'i' and all
       other atoms in the system.

    The idea behind PBC is that there are infinite identical copies of the simulation box to emulate a bulk material.
    In that case, if we have 100 atoms in our system and we wish to calculate the distance between a reference atom
    and the other 99 atoms in the box, then we must consider the closest "copy" of each of the 99 atoms to this
    reference atom. This is the minimum image convention (MIC), i.e. we consider the smallest distance between the
    reference atom and the "infinite" copies of the other atoms.

    Args:
        r_ij (np.array): A (n_atoms, 3) matrix r_ij = r_i - r or a (n_atoms, n_atoms, 3) 3-D array containing all r_i-r.
                       To better understand, for the 2-D array case, each row stores (x_i - x_j, y_i - y_j, z_i - z_j)
                       or (Δx, Δy, Δz).
        box_length (float): Length of cubic cell

    Returns:
      np.array: r_ij under MIC

    OBS1: If we have a non-cubic box then we must first transfor the non-cubic box into an cubic box
    OBS2: We can use abs(r_ij) as well because r_ij stores the coordinate differences between atoms 'i' and 'j
          (Δx, Δy, Δz) and the euclidean distance doesn't depend on the sign of (Δx, Δy, Δz) but only its
          norm as we use their squared values d = sqrt(Δx² + Δy² + Δz²)
    """
    # If r_ij/L is < 0.5 then it is already the minimum distance, but if it is > 0.5 we subtract a box length from it.
    # This works because np.round() will round x > 0.5 to 1 and x <= 0.5 to 0. The idea is that the maximum allowed
    # distance along each dimension has to be at most L/2.
    return r_ij - box_length * np.round(r_ij/box_length) # abs(r_ij) - box_length * np.round(abs(r_ij)/box_length)


def kinetic_energy(vel, mass):
  """Calculate total kinetic energy.

  Args:
      vel (ndarray): Particle velocities, shape (n_atoms, 3).
      mass (float): Particle mass.
  Returns:
      (float): Total kinetic energy.
  """
  return mass/2 * np.sum(vel**2)



def rc_potential_energy(distance_table, rc, epsilon=1, sigma=1):
    """Calculate total potential energy using cut-and-shifted Lennard-Jones potential.

    Args:
        distance_table (ndarray): Distance table of shape (n_atoms, n_atoms) containing the distance betwen all pair of
                                  atoms
        rc (float): Cutoff radius for the potential energy
        epsilon (float): Potential energy parameter
        sigma (float): Potential distance parameter

    Returns:
        (float): total potential energy

    OBS1: Since the potential form is expensive computationally, we should first remove all the distances that we
          won't need. If we remove it after applying the potential we are wasting resources (always treat the input
          data first; don't do unecessary calculations).
    """
    Vc = 4 * epsilon * ((sigma/rc)**12 - (sigma/rc)**6)
    d_ij = np.triu(distance_table, 1) # d_ij is a upper triangular matrix
    d_i = d_ij[(d_ij != 0) & (d_ij < rc)] # Now d_i is a vector containig only the permitted distances
    return np.sum(4 * epsilon * ((sigma/d_i)**12 - (sigma/d_i)**6) - Vc)


def rc_force_vectorized(pos, box_length, rc, epsilon=1, sigma=1):
    """Compute the forces acting on each atom given a LJ potential.

    Args:
        pos (np.array) : particle positions, shape (n_atoms, 3)
        box_length (float) : side length of cubic box
        rc (float) : cutoff radius
        epsilon (float) : potential parameter
        sigma (float) : potential parameter

    Returns:
      (ndarray): total force vector (n_atom, 3)

    OBS1: Displacement matrix and distance vectors are two different things. Displacement is simply r_ij = r_i - r_j
          where 'r_i' and 'r_j' are the coordinate vectors of atoms 'i' and 'j', which in that case it is a vector.
          The distance vector is the euclidean distance of the displacement vector d = np.sqrt(np.sum(r_ij**2)) and
          it is a number. If we are doing in a vectorized way, the displacement matrix will be a 3-D array (an array
          of matrices representing the displacement matrix of each atom 'i') and the distance vector will be a matrix
          where each row is the distance vector for each atom 'i';
    """
    F_ij = np.zeros_like(pos)
    r_ijk = pos[:,np.newaxis,:] - pos # its a faster r_ijk = np.array([pos[idx] - pos for idx in range(pos.shape[0])])
    r_ijk = to_mic(r_ijk, box_length) # Finding the diplacement matrices in the MIC for each atom 'i'
    d_ij = np.linalg.norm(r_ijk, axis=2) # Finding the distance vectors in the MIC for each atom 'i'

    # Filtering the distances between 0 and the cutoff radius
    idx = np.nonzero((d_ij!=0) & (d_ij<rc))
    d_ij = d_ij[idx].reshape(-1,1)

    d_counter = 0
    for i, dist_table_i in enumerate(r_ijk): # matrix_i is the distance table for particle 'i'
        idx_target = idx[1][idx[0] == i] # contains the indices of the atoms where the distance from 'i' is less than cutoff
        n_targets = len(idx_target) # stores how many atoms with respect to 'i' we need to compute the forces (d_ij < r_c)
        r_i = dist_table_i[idx_target]
        d_i = d_ij[d_counter:d_counter+n_targets]
        F_ij[i] = np.sum(24 * epsilon / d_i * (2*(sigma/d_i)**12 - (sigma/d_i)**6) * (r_i/d_i), axis=0)
        d_counter += n_targets

    return F_ij

def construct_displacement_table(r_ij, box_length):
    """Constructs a displacement table in the minimum image convention given a coordinate matrix and the simulation
       box length length

    Args:
        r_ij (ndarray): A coordinate matrix of shape (n_atoms, dim=3)
        box_length (float): Length of the cubic box

    Returns:
        r_ijk (np.array): A 3-D array of shape (n_atoms, n_atoms, dim=3) containing the displacement table of all
                          atoms in the minimum image convention. For instance, r_ijk[i] is the displcament table
                          of the i-th atom.
    """
    return to_mic(r_ij[:,np.newaxis,:] - r_ij[np.newaxis,:,:], box_length)


def construct_distance_table(r, box_length):
    """Constructs the distance table of the system given the matrix 'r' containing the coordinates of each atom

    Args:
        r (np.array): A matrix with shape (n_atoms, 3) that contains the 'x' 'y' 'z' coordinates of each atom
        box_length (float): Length of the simulation box
    Returns:
        d_ij (np.array): A matrix with shape (n_atoms, n_atoms) that contains the distance between each pair of atoms
                         'i' and 'j'. The matrix is symmetric (d_ij = d_ji) and all elements of the main diagonal are 0.

    OBS1: If we import pdist and squareform from scipy.spatial.distance and return squareform(pdist(r, "euclidean"))
          it will be 5x faster, but the output won't be in the upper triangular form (still worth it) and we cannot
          put the distance vector under MIC.
    """

    # Method 0: 1000 it/s in a 500 x 3 coordinates matrix, but I cannot use MIC
    # d_ij = squareform(pdist(r))

    # Method 1: 56 it/s in a 500 x 3 coordinates matrix
    # d_ij = np.zeros((r.shape[0], r.shape[0]))
    # for i in range(r.shape[0]):
    #     d_ij[i] = np.linalg.norm(to_mic(r[i] - r, box_length), axis=1)
    # return d_ij

    # Method 2: 440 it/s in a 100 x 3 matrix; 65 it/s in a 500 x 3 matrix; 25 it/s in a 1000 x 3 matrix
    # d_ij = np.zeros((r.shape[0], r.shape[0]))
    # for i in range(r.shape[0] - 1):
    #     d_ij[i,i+1:] = np.linalg.norm(to_mic(r[i] - r[i+1:, :], box_length), axis=1)
    # return d_ij

    # Method 3: 3497 it/s in a 100 x 3 matrix; 66 it/s in a 500 x 3 matrix; 16 it/s in a 1000 x 3 matrix
    return np.linalg.norm(to_mic(r[:,np.newaxis,:] - r[np.newaxis,:,:], box_length), axis=2)


def initialize_positions_cube(n_atoms, box_length):
    """Initializes the positions of 'n' atoms in a cubic grid where each axis goes from [-L/2, L/2]

    Args:
        n_atoms (int): The number of atoms in the system.
        box_length (float): The length of the cubic box.

    Returns:
        (ndarray): An array (n_atoms, 3) containing the x, y, z coordinates of each atom.

    OBS0: Note that (n)**1/3 has to be an integer, otherwise we cannot construct a perfect cubic lattice
    OBS1: r = np.array(tuple(product(pos,pos,pos))), which uses product function from itertools module is 10% faster
          than the way we implemented. Another option is np.array(np.meshgrid(pos, pos, pos)).T.reshape(-1, 3), which
          scales better with increasing 'N', and it becomes much faster for N > 4.
    """
    N = int(round(n_atoms**(1/3)))
    offset = (box_length / N) / 2 # accounting for periodic boundary conditions
    pos = [x+offset for x in range(0,N)] # creating a list contaning the grid points along an axis
    r = np.array([(x, y, z) for x in pos for y in pos for z in pos]) # taking the cartesian product pos x pos x pos
    return r - box_length/2


def initialize_velocities(n_atoms=10, T=0.728, m=1, seed=103):
    """Randomly initializes the velocity of 'n' atoms as specified in Frenkel & Smit p. 66

    Args:
        n_atoms (int): The number of atoms in the system.
        T (float): Desired temperature of simulation, necessary to scale all velocities to match the temperature.
        m (float): Mass of the particles that we are simulating.
        seed (int): The seed that will feed numpy random generator.

    Returns:
        np.array: An array (n_atoms,3) containing the velocity componentes v_x, v_y, v_z for each atom.

    OBS1: The net momentum of the box has to be zero, i.e. np.all(np.isclose(v.sum(axis=0), 0)) == True
    OBS2: We are working with reduced units for temperature, that is why we scale the velocities so m〈v²〉 = T* (we
          are considering k_bT = T*). The overall goal is to make the ensemble average kinetic energy of the system
          match the expected value from the equipartition theorem 〈K〉 = 3/2 k_b T. If we look into individual velocities
          components, then using 'x' as example its m〈v²_x〉/2 = 1/2 k_b T thus m〈v²_x〉 = k_b T and that is what we do
          for the scaling factor np.sqrt(T / (np.mean(v**2) * m)) since m * np.mean((v * scaling_factor)**2) = T.
    """
    np.random.seed(seed)
    v = np.random.random((n_atoms, 3)) - 0.5 # Initializing the velocity between [-0.5, 0.5]
    v -= np.sum(v, axis=0) / n_atoms # Shifting the velocities so that the net momentum is 0 in all directions
    scaling_factor = np.sqrt(T / (np.mean(v**2) * m))
    v *= scaling_factor # Scaling velocities so that m〈v²〉 = T

    return v


def verlet_pos(r, v, F, m, dt):
    """Velvet integration of Newton's equation of motion for position.
    """
    return r + v * dt + F/(2*m) * (dt**2)


def verlet_vel(v, F, F_next, m, dt):
    """Velvet integration of Newton's equation of motion for velocity.
    """
    return v + (F + F_next)/(2*m) * dt


def calculate_pair_distribution(dists, n_atoms, n_bins, box_length):
    """ Calculate the pair distribution function g(r). Note that it doesn't make sense to calculate the g_r over the
        whole box length because then we would be counting images. Therefore, n_bins * dr ≤ box_length/2

    Args:
        dists (np.array): 1-D array of pair distances of shape (n_atoms*(n_atoms-1)/2, )
        n_atoms (int): number of atoms
        n_bins (int): number of bins in the histogram
        box_length (float): Length of cubic box

    Returns:
        (ndarray): array of shape (nbins,) representing the pair correlation g(r)

    OBS1: We multiply the histogram by 2 / n_atoms. The '2' is there because we are feeding this function only the
          upper triangular matrix of the distance table, so we multiply the histogram by '2' to count the distances
          represented by the lower triangular part (the distance table is symmetric). By doing so, we binned all pair
          distances r_ij except when i=j. Then, we divide by the total number of atoms to find the average radial
          distribution.

    OBS2: For liquids, at large distances 'r' the g(r) will decay to 1. Mathematically, the particle count in the
          spherical shell defined by this large 'r' given by 'avg_histogram' will be the same as the particle count
          considering a structureless ideal gas with the same bulk density given by 'histogram_normalization'. Thus,
          both counts will be the roughly same and therefore g(r) ~ 1 at large 'r'.

    """
    number_density = n_atoms / (box_length ** 3)

    # Creating a histogram with the pair distances
    dr = (box_length / 2) / n_bins
    histogram, bins_edge = np.histogram(dists, bins=n_bins, range=(0, n_bins * dr))
    avg_histogram = (2*histogram) / n_atoms

    # Each spherical shell has the same number density as the bulk density of the system, so it normalizes the g(r) to
    # make it decay to 1 at long distances 'r'.
    volume_spherical_shells = 4/3 * np.pi * np.diff(bins_edge**3) # Volume of spherical shells delimited by bins edges
    histogram_normalization = volume_spherical_shells * number_density # Number of atoms in each shell

    return avg_histogram / histogram_normalization


def generate_kvecs(max_n, box_length):
  """Generates k-vectors commensurate with a cubic box. Considers only k-vectors in the all-positive octant of
     reciprocal space. As we increase the number of k-points along each axis we use more k-vectors.

  Args:
      max_n : max_n + 1 is the number of k-points along each axis (maximum value for nx, ny, nz).
      box_length : side length of cubic cell

  Returns:
      (ndarray): array of shape (nk, ndim) representing a collection of k-vectors

  OBS1: The effect of increasing the number of k-points is to use k-vectors with greater magnitude. Note that the
        spacing between successive k-vectors |Δk| is independent of the number of k-points, For instance, the first
        six magnitudes will always be [0, 1.48457342  2.09950386  2.57135658  2.96914683  3.31960708]. If max_n=5
        the magnitude of the last k-vectors will be 12.85678292, but if max_n=6 it will be 15.42813951.
  """
  k = 2 * np.pi / box_length * np.linspace(0, max_n, max_n+1)
  k_vecs = np.array(np.meshgrid(k,k,k)).T.reshape(-1,3) # [(x,y,z) for x in k for y in k for z in k]
  k_vecs[:, [1,2]] = k_vecs[:, [2,1]]
  k_vecs[:, [0,1]] = k_vecs[:, [1,0]]
  return k_vecs


def calculate_rho_k(k_vecs, pos):
    """Calculate the fourier transform of particle density 'ρ', ρ_k = ∑_j exp(-ik·r_j) for j=[1,N_atoms] where
       k·r_j is the dot product between all k-vectors (i.e. a matrix where each row is a k_vector) and 'r_j'. Note
       that we use Euler identity exp(-ik·r_j) = cos(k·r_j) + i*sin(k·r_j)

    Args:
        k_vecs (np.array): Array of k-vectors, shape (n_kvecs, dim=3).
        pos (np.array): Particle positions, shape (n_atoms, dim=3).

    Returns:
        (ndarray): array of shape (n_kvecs,) representing the Fourier transformed density rho_k
    """
    # 'kr' has shape (n_kvecs, n_atoms) where each column 'j' contains np.sum(k_vecs * pos[j], axis=1), for instance
    # kr[:,j] = [np.dot(k_vecs[0],         pos[j]),
    #            np.dot(k_vecs[1],         pos[j]),
    #            ...
    #            np.dot(k_vecs[n_kvecs-1], pos[j])]
    kr = k_vecs @ pos.T
    return np.sum(np.cos(kr) - 1.0j * np.sin(kr), axis=1)


def calculate_structure_factor(kvecs, pos):
    """Calculate the structure factor S(k) = 1/N_atoms 〈ρ_k, ρ_-k〉, where 'ρ' is the particle density (N_particles / V)
       and the ensemble average is over the frames sampled during MD.

    Args:
        kvecs (ndarray): Array of k-vectors, shape (n_kvecs, dim=3)
        pos (ndarray): Particle positions, shape (n_atoms, dim=3)
    Returns:
      (ndarray): Array of shape (n_kvecs,) representing the structure factor s(k)

    OBS1: We should always plot the returned value from [1:] because s_k[0] is always equal to n_atoms. Also, we should
          plot using the magnitude of the k-vector because there will be different k-vectors with the same magnitude
          but different orientations. Since liquids are isotropic, their direction won't matter. .
    """
    # We get only the real part because the imaginary will be 0 since rho_k(kvecs, pos) * rho_k(-kvecs, pos) is just
    # the magnitude of rho_k.
    return (calculate_rho_k(kvecs, pos) * calculate_rho_k(-kvecs, pos)).real / pos.shape[0]


def autocorrelation_time(a):
    """Computes the autocorrelation time of a time series.

    Args:
        a (ndarray): Time series

    Returns:
      (float): autocorrelation time of 'a'
    """
    N = a.shape[0] # number of time points (size of the time series)
    mean = np.mean(a)
    std = np.std(a)
    C = []

    # Calculate the autocorrelation function until it becomes <= 0 (this value is not included). We are considering
    # that, given that acf[k] is the first point crossing the x-axis, if we overlap a[:N-k] and a[k:] we can consider
    # these time series independent.
    for t_i in range(N): #                    sliding array      fixed array
        acf = 1 / (std**2 * (N - t_i)) * sum((a[:N-t_i] - mean) * (a[t_i:] - mean))
        if acf <= 0: break # Breaks the loop because the cutoff value was reached
        C.append(acf)

    return 1 + 2 * sum(C[1:])


def calculate_velocity_autocorrelation(v0, v_i):
    """Calculate the autoccorelation between the velocity at t=t' and the velocity at the first time point t=0.

    Args:
        v0 (ndarray): Velocity matrix (n_atoms, ndim=3) containing the initial velocities
        v_i (ndarray): Velocity matrix (n_atoms, ndim=3) containing the velocities at the i-th frame.

    Returns:
        (float): Velocity autocorrelation between t=0 and t=t'
    """
    return np.sum(v0 * v_i) / v0.shape[0]


def calculate_diffusion_constant(v_acf, dt=0.032):
    """Calculate the diffusion constant from the velocity-velocity auto-correlation function.

    Args:
        v_acf (ndarray): Array of shape (n_frames,) sampled at time steps from t = [0, n_frames*dt].
        dt (float): Time step of the simluation, i.e. time difference between two consecutive frames

    Returns:
        float: the diffusion constant calculated from the vacf.
    """
    return np.trapz(v_acf, dx=dt) / 3


def andersen_thermostat(v, Ti, Tf, m, n_frames, idx_current_frame, p_collision=0.01):
    """Rescales the velocities using Andersen Thermostat.

    Args:
        v (np.array): Matrix of shape (n_atoms, dim=3) containing the velocities of the atoms
        Ti (float): The initial temperature of the system
        Tf (float): The final temperature of the system (usually T_f = T0)
        m (float): Mass of the particle
        n_frames (int): Total number of frames
        idx_current_frame (int): Frame of the current index
        p_collision (float): Probability of collision

    Returns:
        np.array: Matrix of shape (n_atoms, dim=3) containing the rescaled velocities

    """
    T = np.linspace(Ti, Tf, n_frames) # If T0==Tf then this is equal to np.ones(n_frames) * T0
    new_v = v.copy()
    np.random.seed()
    collision_vector = np.random.random(v.shape[0]) # When the number generated is between [0, 0.01] a collision happens
    idx_collision = np.flatnonzero(collision_vector <= p_collision) # Getting the particles indices where a collision happens
    new_v[idx_collision] = np.random.normal(size=(idx_collision.shape[0], 3), loc=0, scale=np.sqrt(T[idx_current_frame]/m))
    return new_v


def get_temperature(total_kinetic_energy, n_atoms):
    """Returns the instantaneous temperature on the MD simulation using equipartition theorem 〈K〉 = 3/2 * k_b * T where
       〈K〉 is the average kinetic energy of a particle in the system (K_total / n_particles).

    Args:
        total_kinetic_energy: Total kinetic energy of the system.
        n_atoms: Number of atoms in the system.

    Returns:
        (float): Instantaneous temperature of the simulation.

    """
    return (total_kinetic_energy / n_atoms) * 2 / 3


def min_dist(coords, box_length, verbose=False):
    """Minimum distance between any two atoms assuming periodic boundary conditions"""
    d_ijk = to_mic(coords[:,np.newaxis,:] - coords[np.newaxis,:,:], box_length)
    dists = np.linalg.norm(d_ijk, axis=2)
    min_dists = np.min(dists[dists != 0].reshape(d_ijk.shape[0], d_ijk.shape[0]-1), axis=-1)
    if verbose:
        print(f'The minimum distance for each atom is {min_dists}')
        print(f'The minimum distance between any two atoms assuming PBC is {min(min_dists):.6f}')

    return min(min_dists), min_dists

if __name__ == '__main__':
    n_atoms = 125
    m = 48 # particle mass
    density = 1.0
    L = (n_atoms/density)**(1/3) # box length
    # L = 4.2323167
    rc = L / 2
    T0, T_f = 0.9, 0.9 # Initial and final temperature (usually is the same)
    dt = 0.03
    n_frames = 1000
    total_time = n_frames * dt
    r = initialize_positions_cube(n_atoms, L)
    v0 = initialize_velocities(n_atoms, T0, m, seed=103)
    v = v0.copy()
    F = np.zeros_like(r) # Force on the current iteration
    F_next = np.zeros_like(r) # Force on the next iteration
    K = np.zeros(n_frames) # Total Kinetic energy
    V = np.zeros(n_frames) # Total Potential energy
    E = np.zeros(n_frames) # Total energy
    T_arr = np.zeros(n_frames) # Temperature
    v_autocorr = np.zeros(n_frames)
    g_r = []
    n_bins = 100
    s_k = []
    k_vecs = generate_kvecs(7, L)
    triu1_idx = np.triu_indices(n_atoms, k=1) # There are 2016 indices as expected
    pre_eq_cutoff = 80 # Frame where the warm-up or pre-equilibration ends
    use_thermostat = False
    sigma = 1.0
    epsilon = 1.0
    dist_table_next = np.zeros((n_atoms, n_atoms))
    disp_table_next = np.zeros_like(r)

    for i in tqdm(range(n_frames)):
        dist_table = construct_distance_table(r, L)
        if i >= pre_eq_cutoff: # We only need to calculate them after the warm-up time
            g_r.append(calculate_pair_distribution(dist_table[triu1_idx], n_atoms, n_bins, L))
            # s_k.append(calculate_structure_factor(k_vecs, r))
        K[i] = kinetic_energy(v, m)
        V[i] = rc_potential_energy(dist_table, rc, sigma=sigma, epsilon=epsilon)
        E[i] = K[i] + V[i]
        F = rc_force_vectorized(r, L, rc, sigma=sigma, epsilon=epsilon) if i == 0 else F_next.copy()
        r = verlet_pos(r, v, F, m, dt) # Updating positions
        r = wrap_coords(r, L) # Wrapping the new positions
        F_next = rc_force_vectorized(r, L, rc, sigma=sigma, epsilon=epsilon)
        v = verlet_vel(v, F, F_next, m, dt)
        if use_thermostat:
            v = andersen_thermostat(v, T0, T_f, m, n_frames, i, 0.01)
        v_autocorr[i] = calculate_velocity_autocorrelation(v, i)
        T_arr[i] = get_temperature(K[i], n_atoms)


    # Faster because we calc the disp and dist table only once per iteration, but we need to modify rc_force_vectorized
    # for i in tqdm(range(n_frames)):
    #     disp_table = disp_table_next.copy() if i != 0 else construct_displacement_table(r, L)
    #     dist_table = dist_table_next.copy() if i != 0 else construct_distance_table(disp_table, L)
    #     if i >= pre_eq_cutoff: # We only need to calculate them after the warm-up time
    #         g_r.append(calculate_pair_distribution(dist_table[triu1_idx], n_atoms, n_bins, L))
    #         s_k.append(calculate_structure_factor(k_vecs, r))
    #     K[i] = kinetic_energy(all_v[i], m)
    #     V[i] = rc_potential_energy(dist_table, rc, sigma=sigma, epsilon=epsilon)
    #     E[i] = K[i] + V[i]
    #     F = F_next.copy() if i != 0 else rc_force_vectorized(r, L, rc, disp_table, dist_table, sigma=sigma, epsilon=epsilon)
    #     r = velvet_pos(r, all_v[i], F, m, dt) # Updating positions
    #     r = wrap_coords(r, L) # Wrapping the new positions
    #     disp_table_next = construct_displacement_table(r, L)
    #     dist_table_next = construct_distance_table(disp_table_next)
    #     F_next = rc_force_vectorized(r, L, rc, disp_table_next, dist_table_next, sigma=sigma, epsilon=epsilon)
    #     all_v[i+1] = velet_vel(all_v[i], F, F_next, m, dt)
    #     if use_thermostat:
    #         all_v[i+1] = andersen_thermostat(all_v[i+1], T, T, m, n_frames, i, 0.01)
    #     v_autocorr[i] = calculate_velocity_autocorrelation(all_v, i)
    #     T_arr[i] = get_temperature(K[i], n_atoms)

    g_r = np.mean(np.array(g_r), axis=0) # Both have n_rows = n_frames - pre_eq_cutoff
    # s_k = np.mean(np.array(s_k), axis=0)

#--------------------------------------------------PLOTS---------------------------------------------------------------#
    ## Total, kinetic, and potential energies
    # acf_time = autocorrelation_time(K)
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0, total_time, n_frames), K, color='orangered', label='Kinetic Energy')
    # ax.plot(np.linspace(0, total_time, n_frames), E, color='darkviolet', label='Total Energy')
    # ax.plot(np.linspace(0, total_time, n_frames), V, color='navy', label='Potential Energy')
    # ax.plot([pre_eq_cutoff * dt]*2, (V.min(), K.max()), color='black', ls='-.', alpha=0.8)
    # ax.set_xlabel('Time (arb)')
    # ax.set_ylabel('E (arb)')
    # ax.set_title(rf'$\Delta t = {dt}$')
    # ax.legend()
    # # fig.savefig(f'C:/Users/Vitor/Desktop/bresa/UIUC/Courses/22.2 - Fall/CSE 485/Homework/3/p3_energies_T{T}.pdf', dpi=500)
    # plt.show()


    ## Pair Distribution Function
    fig, ax = plt.subplots()
    print(g_r)
    ax.plot(np.linspace(0, L/2, g_r.shape[0]), g_r, color='navy')
    ax.set_xlabel('r (arb)')
    ax.set_ylabel('g(r)')
    ax.set_title('Pair Distribution Function')
    # fig.savefig(f'C:/Users/Vitor/Desktop/bresa/UIUC/Courses/22.2 - Fall/CSE 485/Homework/3/p3_gr_T{T}.pdf', dpi=500)
    plt.show()


    ## Structure Factor
    # k_mag = np.linalg.norm(k_vecs, axis=1).round(8)
    # unique_k_mag = np.unique(k_mag)
    # unique_sk = np.zeros_like(unique_k_mag)
    # # unique_sk = np.array([s_k[np.nonzero(unique_k_mag[i] == k_mag)].mean() for i in range(unique_k_mag.shape[0]])
    # for i in range(unique_k_mag.shape[0]):
    #     idx = np.nonzero(unique_k_mag[i] == k_mag)
    #     unique_sk[i] = s_k[idx].mean()
    #
    # fig, ax = plt.subplots()
    # ax.plot(unique_k_mag[1:], unique_sk[1:], color='orangered')
    # ax.set_xlabel('k (arb)')
    # ax.set_ylabel('S(k)')
    # ax.set_title('Structure Factor')
    # # fig.savefig(f'C:/Users/Vitor/Desktop/bresa/UIUC/Courses/22.2 - Fall/CSE 485/Homework/3/p3_sk_T{T}.pdf', dpi=500)
    # plt.show()




