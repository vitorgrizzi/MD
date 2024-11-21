import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from physchem_funcs import *
from mathstats_funcs import *
from tqdm import tqdm
from scipy.integrate import trapezoid

def extract_cluster(lammps_data, shuffle=True):
    """Takes a LAMMPS input data file and extracts a the cluster information as a dict {cluster_center_id: [neighs_id]}

       OBS: We subtract the `id` in the data file by 1 for them to match python indexes.
    """
    cluster_dict = {}
    bond_flag = False
    with open(lammps_data, 'r') as f:
        for line in f:
            if 'Bonds' in line:
                bond_flag = True

            if bond_flag:
                try:
                    # The first id in the bond is always the center atom id by construction
                    atom_id, neigh_id = list(map(int, line.split()[-2:]))
                    # LAMMPS atom_id is from [1,N] so we must subtract 1 to correspond to a python index
                    atom_id -= 1
                    neigh_id -= 1
                    if cluster_dict.get(atom_id):
                        cluster_dict[atom_id].append(neigh_id)
                    else:
                        cluster_dict[atom_id] = [neigh_id]
                except:
                    continue

    if shuffle:
        cluster_dict = {k: cluster_dict[k] for k in np.random.permutation(list(cluster_dict.keys()))}

    return cluster_dict


def extract_atstress(stress_dump, n_frames, n_stresses=3):
    """Extracts the atomic stresses from a LAMMPS dump file with shape (n_atoms, n_frames, n_stresses)

       OBS1: n_frames should be equal to run/dump_freq, here often run/4.
    """
    with open(stress_dump, 'r') as f:
        for i, line in enumerate(f):
            if i == 3:
                n_atoms = int(line)
                break

    header_size = 9
    init_frame_lines = np.arange(0, n_frames+1, 1) * (n_atoms+header_size) 
    idxs_header_lines = np.ravel(np.arange(0, header_size, 1).reshape(1, -1) + init_frame_lines.reshape(-1, 1))

    # Skip the first few header lines and directly read relevant data
    df = pd.read_table(stress_dump, skiprows=idxs_header_lines, sep='\s+', header=None)

    # Extract the last n_stresses columns
    at_stresses = df.iloc[:, -n_stresses:].to_numpy(dtype=float)
    at_stresses = np.swapaxes(at_stresses.reshape(n_frames, n_atoms, n_stresses), 0, 1)

    return at_stresses


def extract_coords(traj_dump, n_frames):
    """Extracts the coordinates from a LAMMPS dump file with shape (n_atoms, n_frames, n_dim)

       OBS1: n_frames should be equal to run/dump_freq, here often run/100
    """
    with open(traj_dump, 'r') as f:
        for i, line in enumerate(f):
            if i == 3:
                n_atoms = int(line)
                break

    header_size = 9
    init_frame_lines = np.arange(0, n_frames+1, 1) * (n_atoms+header_size)
    idxs_header_lines = np.ravel(np.arange(0, header_size, 1).reshape(1, -1) + init_frame_lines.reshape(-1, 1))

    # Skip the first few header lines and directly read relevant data
    df = pd.read_table(traj_dump, skiprows=idxs_header_lines, sep='\s+', header=None)

    # Extract the last n_stresses columns
    coords = df.iloc[:, -3:].to_numpy(dtype=float)
    coords = np.swapaxes(coords.reshape(n_frames, n_atoms, 3), 0, 1)

    return coords


def to_mic(r_ij, box_length):
    """Impose minimum image convention on the vector r_ij = r_i - r containing the distances between atom 'i' and all
       other atoms in the system."""
    return r_ij - box_length * np.round(r_ij/box_length) # abs(r_ij) - box_length * np.round(abs(r_ij)/box_length)


def construct_distance_table(r_bulk, r_all, box_length):
    """Constructs the distance table of the system given the matrix 'r' containing the coordinates of each atom
    """
    return np.linalg.norm(to_mic(r_bulk[:,np.newaxis,:] - r_all[np.newaxis,:,:], box_length), axis=2)


def get_box_length(traj_dump):
    """Get the box length
    """
    with open(traj_dump, 'r') as f:
        for i, line in enumerate(f):
            if i == 5:
                box_length = float(line.split()[-1])
                break

    return box_length


def find_bulk_atom(cluster_dict, n_atoms):
    """Given a cluster dict, find the bulk atoms i.e. atoms that are not a part of any cluster
    """
    possible_ids = np.arange(0, n_atoms)

    center_atoms = np.fromiter(cluster_dict.keys(), dtype=int)
    neigh_atoms = np.array(list(cluster_dict.values()), dtype=int).ravel()
    cluster_atoms = np.hstack((center_atoms, neigh_atoms))

    bulk_atoms = np.setdiff1d(possible_ids, cluster_atoms, assume_unique=True)

    return bulk_atoms


def correlation(x1, x2, n_percentile=5):
    """Calculates the cross correlation along the last axis of a 2-D or 1-D array 'x'.
    """
    N = x1.shape[-1]
    acf_size = int(N * n_percentile / 100)  # this defines the maximum time delay tau_max.

    x1_centered = np.transpose(x1.T - x1.mean(axis=-1))
    x2_centered = np.transpose(x2.T - x2.mean(axis=-1))
    x1_padded = np.hstack((x1_centered, np.zeros_like(x1)))
    x2_padded = np.hstack((x2_centered, np.zeros_like(x2)))
    x1_fft = fft(x1_padded)
    x2_fft = fft(x2_padded)
    # We use .real because the acf is the inverse transform of the norm of this complex vector 'x_fft', so the
    # imaginary part is just a residue
    cf = ifft(x1_fft * x2_fft.conjugate()).real[..., :acf_size]

    # The returned array has size (x1.shape[0], acf_size]
    return cf / np.arange(N, N - acf_size, -1)


def bootstrap_correlation(A, B, divisor=1.3, n_samples=50, n_perc=5):
    """Returns the average correlation between array `A` and `B` bootstrapped `n_samples` times
    """
    n_frames = A.shape[-1]  # A and B have the same shape
    sample_size = int(n_frames // divisor)
    acf_size = int(sample_size * n_perc / 100)  # Just return the correlation of the first `n_perc` of the array
    acf_matrix = np.zeros((n_samples, acf_size), dtype=float)
    for i in tqdm(range(n_samples)):
        starting_frame = np.random.randint(0, n_frames - sample_size)
        A_sample = A[:, starting_frame:(starting_frame + sample_size)]
        B_sample = B[:, starting_frame:(starting_frame + sample_size)]
        acf_matrix[i] = np.mean(correlation(A_sample, B_sample, n_percentile=n_perc), axis=0)  # Mean over Pxy Pxz Pyz

    return np.mean(acf_matrix, axis=0)  # Mean over the `n_samples` correlations

#----------------------------------------------------------------------------------------------------------------------#
#                                             CLUSTER FUNCTIONS                                                        #
#----------------------------------------------------------------------------------------------------------------------#

def cluster_acf(cluster_dict, atomic_stresses, n_samples=50, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the autocorrelation of the cluster centers
    """
    cluster_centers = np.fromiter(cluster_dict.keys(), dtype=int)[:n_samples]

    pxy = atomic_stresses[cluster_centers, :, 0]
    pyz = atomic_stresses[cluster_centers, :, 1]
    pzx = atomic_stresses[cluster_centers, :, 2]
    p_stacked = np.vstack((pxy, pyz, pzx))

    return bootstrap_correlation(p_stacked, p_stacked, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def clusterneigh_ccf(cluster_dict, atomic_stresses, n_samples=25, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the crosscorrelation of the cluster centers with its bonding neighbors
    """
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)[:n_samples]
    n_neighs = len(cluster_dict[cluster_center[0]])

    # I have to "replicate" the stress for the cluster center `n_neighs` times
    cluster_center_rep = np.repeat(cluster_center, n_neighs)
    cluster_center_neigh = np.array(list(cluster_dict.values())[:n_samples]).reshape(-1)

    pxy_center = atomic_stresses[cluster_center_rep, :, 0]
    pyz_center = atomic_stresses[cluster_center_rep, :, 1]
    pzx_center = atomic_stresses[cluster_center_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_neigh = atomic_stresses[cluster_center_neigh, :, 0]
    pyz_neigh = atomic_stresses[cluster_center_neigh, :, 1]
    pzx_neigh = atomic_stresses[cluster_center_neigh, :, 2]
    p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

    return bootstrap_correlation(p_center, p_neigh, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def clusterbulk_ccf(cluster_dict, atomic_stresses, n_samples=25, n_bulk=50, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the crosscorrelation of the cluster centers with bulk atoms
    """
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)[:n_samples]
    cluster_center_neigh = np.array(list(cluster_dict.values())[:n_samples]).reshape(-1)

    # I have to "replicate" the stress for each cluster center `n_bulk` times
    cluster_center_rep = np.repeat(cluster_center, n_bulk)

    # Choosing `n_bulk` bulk atoms randomly
    bulk_ids = np.setdiff1d(np.arange(0, n_atoms, 1), np.hstack((cluster_center, cluster_center_neigh)))[:n_bulk]
    bulk_ids_rep = np.tile(bulk_ids, n_samples)

    pxy_center = atomic_stresses[cluster_center_rep, :, 0]
    pyz_center = atomic_stresses[cluster_center_rep, :, 1]
    pzx_center = atomic_stresses[cluster_center_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_neigh = atomic_stresses[bulk_ids_rep, :, 0]
    pyz_neigh = atomic_stresses[bulk_ids_rep, :, 1]
    pzx_neigh = atomic_stresses[bulk_ids_rep, :, 2]
    p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

    return bootstrap_correlation(p_center, p_neigh, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def clusternonneigh_ccf(cluster_dict, atomic_stresses, n_samples=25, n_bulk=50, n_bootstraps=50, n_perc=1, divisor=1.3):
    """Computes the crosscorrelation of the cluster centers with non-neighbors atoms
    """
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)[:n_samples]
    cluster_center_neigh = np.array(list(cluster_dict.values())[:n_samples])  # (n_samples, n_neighs)

    # I have to "replicate" the stress for each cluster center `n_bulk` times
    cluster_center_rep = np.repeat(cluster_center, n_bulk)

    # Choosing `n_bulk` non-neighbor atoms for each cluster randomly
    possible_ids = np.arange(0, n_atoms, 1)
    nonneigh_ids = np.zeros((n_samples, n_bulk), dtype=int)
    for i in range(n_samples):
        candidate_ids = np.setdiff1d(possible_ids, np.hstack((cluster_center[i], cluster_center_neigh[i])))
        nonneigh_ids[i] = candidate_ids[np.random.randint(0, candidate_ids.size, size=n_bulk)]
    nonneigh_ids = nonneigh_ids.ravel()

    pxy_center = atomic_stresses[cluster_center_rep, :, 0]
    pyz_center = atomic_stresses[cluster_center_rep, :, 1]
    pzx_center = atomic_stresses[cluster_center_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_nonneigh = atomic_stresses[nonneigh_ids, :, 0]
    pyz_nonneigh = atomic_stresses[nonneigh_ids, :, 1]
    pzx_nonneigh = atomic_stresses[nonneigh_ids, :, 2]
    p_nonneigh = np.vstack((pxy_nonneigh, pyz_nonneigh, pzx_nonneigh))

    return bootstrap_correlation(p_center, p_nonneigh, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def cluster_total_cf(cluster_dict, atomic_stresses, n_samples=1, n_bootstraps=20, n_perc=5, divisor=1.3):
    """Correlation between cluster center and all other atoms
    """
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)[:n_samples]

    cluster_center_rep = np.repeat(cluster_center, n_atoms)
    pxy_center = atomic_stresses[cluster_center_rep, :, 0]
    pyz_center = atomic_stresses[cluster_center_rep, :, 1]
    pzx_center = atomic_stresses[cluster_center_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_all = np.tile(atomic_stresses[:, :, 0], (n_samples, 1))
    pyz_all = np.tile(atomic_stresses[:, :, 1], (n_samples, 1))
    pzx_all = np.tile(atomic_stresses[:, :, 2], (n_samples, 1))
    p_all = np.vstack((pxy_all, pyz_all, pzx_all))

    return bootstrap_correlation(p_center, p_all, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def cluster_visc_contribution(cluster_dict, atomic_stresses, n_bulk=50, n_samples=10, n_bootstraps=20, n_perc=1,
                              divisor=1.3, dt=0.005, dump_freq=4):
    c_acf = cluster_acf(cluster_dict, atomic_stresses, n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc,
                        divisor=divisor)
    cn_ccf = clusterneigh_ccf(cluster_dict, atomic_stresses, n_samples=n_samples, n_bootstraps=n_bootstraps,
                              n_perc=n_perc, divisor=divisor)
    cb_ccf = clusternonneigh_ccf(cluster_dict, atomic_stresses, n_bulk=n_bulk, n_samples=n_samples // 2,
                                 n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)

    dtau = dt * dump_freq
    c_acf_integral = trapezoid(c_acf, dx=dtau)
    cn_ccf_integral = trapezoid(cn_ccf, dx=dtau)
    cb_ccf_integral = trapezoid(cb_ccf, dx=dtau)

    # Contributions: 1 ACF, n_neighs neighbor CCF, N_atoms - (clustersize) bulk CCF
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)
    n_neighs = len(cluster_dict[cluster_center[0]])
    contributions = np.array([1 * c_acf_integral,
                              n_neighs * cn_ccf_integral,
                              (n_atoms - 1 - n_neighs) * cb_ccf_integral])

    return [np.round(np.abs(contributions) / np.abs(contributions).sum() * 100, decimals=2), contributions]

#----------------------------------------------------------------------------------------------------------------------#
#                                             NEIGHBOR FUNCTIONS                                                       #
#----------------------------------------------------------------------------------------------------------------------#

def ncc_acf(cluster_dict, atomic_stresses, n_samples=50, n_bootstraps=100, n_perc=3, divisor=1.2):
    """Computes the autocorrelation of non-center cluster atoms
    """
    neighs = np.array(list(cluster_dict.values()))[:n_samples,0]

    pxy = atomic_stresses[neighs, :, 0]
    pyz = atomic_stresses[neighs, :, 1]
    pzx = atomic_stresses[neighs, :, 2]
    p_stacked = np.vstack((pxy, pyz, pzx))

    return bootstrap_correlation(p_stacked, p_stacked, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)

def nccneigh_ccf(cluster_dict, atomic_stresses, traj, box_length, n_samples=25, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the cross correlation between non-center cluster atoms and their nearest neighbors
    """
    n_stresses = atomic_stresses.shape[1]
    n_trajs = traj.shape[1]
    stress_traj_ratio = (n_stresses-1)//(n_trajs-1)
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)

    noncenter_cluster_atoms = np.array(list(cluster_dict.values()))[:n_samples, 0]
    n_neighs = len(cluster_dict[cluster_center[0]])
    noncenter_cluster_atoms_rep = np.repeat(noncenter_cluster_atoms, n_neighs)

    pxy_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 0]
    pyz_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 1]
    pzx_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 2]
    p_noncenter_cluster_atoms = np.vstack((pxy_bulk, pyz_bulk, pzx_bulk))

    # Bootstrap correlation using the nearest neighbors of the chosen bulk atoms at `n_samples` frames
    sample_size = int(n_stresses // divisor)
    acf_size = int(sample_size * n_perc / 100)
    acf_matrix = np.zeros((n_bootstraps, acf_size), dtype=float)
    max_traj_frame = int(n_trajs * (n_stresses-sample_size)/n_stresses)
    for i in tqdm(range(n_bootstraps)):
        traj_frame = np.random.randint(0, max_traj_frame)

        # Finding the index of the nearest neighbors of the chosen bulk atoms
        dist_bulk = construct_distance_table(traj[noncenter_cluster_atoms, traj_frame, :],
                                             traj[:, traj_frame, :],
                                             box_length)
        dist_bulk[np.isclose(dist_bulk, 0)] = np.inf
        idxs_neighs = np.argsort(dist_bulk, axis=1)[:, :n_neighs].reshape(-1)

        pxy_neigh = atomic_stresses[idxs_neighs, :, 0]
        pyz_neigh = atomic_stresses[idxs_neighs, :, 1]
        pzx_neigh = atomic_stresses[idxs_neighs, :, 2]
        p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

        starting_frame = traj_frame * stress_traj_ratio
        acf_matrix[i] = np.mean(correlation(p_noncenter_cluster_atoms[:, starting_frame:(starting_frame + sample_size)],
                                            p_neigh[:, starting_frame:(starting_frame + sample_size)],
                                            n_percentile=n_perc), axis=0)

    return np.mean(acf_matrix, axis=0)  # Mean over the `n_samples` correlations


def nccnonneigh_ccf(cluster_dict, atomic_stresses, traj, box_length, n_samples=25, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the cross correlation between non-center cluster atoms atoms and their non immediate neighbors
    """
    n_stresses = atomic_stresses.shape[1]
    n_trajs = traj.shape[1]
    stress_traj_ratio = (n_stresses-1)//(n_trajs-1)
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)

    noncenter_cluster_atoms = np.array(list(cluster_dict.values()))[:n_samples, 0]
    n_neighs = len(cluster_dict[cluster_center[0]])
    noncenter_cluster_atoms_rep = np.repeat(noncenter_cluster_atoms, n_neighs)

    pxy_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 0]
    pyz_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 1]
    pzx_bulk = atomic_stresses[noncenter_cluster_atoms_rep, :, 2]
    p_noncenter_cluster_atoms = np.vstack((pxy_bulk, pyz_bulk, pzx_bulk))

    # Bootstrap correlation using the nearest neighbors of the chosen bulk atoms at `n_samples` frames
    sample_size = int(n_stresses // divisor)
    acf_size = int(sample_size * n_perc / 100)
    acf_matrix = np.zeros((n_bootstraps, acf_size), dtype=float)
    max_traj_frame = int(n_trajs * (n_stresses-sample_size)/n_stresses)
    for i in tqdm(range(n_bootstraps)):
        traj_frame = np.random.randint(0, max_traj_frame)

        # Finding the index of the nearest neighbors of the chosen bulk atoms
        dist_bulk = construct_distance_table(traj[noncenter_cluster_atoms, traj_frame, :],
                                             traj[:, traj_frame, :],
                                             box_length)
        dist_bulk[np.isclose(dist_bulk, 0)] = np.inf
        idxs_neighs = np.argsort(dist_bulk, axis=1)[:, n_neighs:]
        idx_rows = np.repeat(np.arange(0, idxs_neighs.shape[0]), n_neighs)
        idx_cols = np.array([np.random.permutation(idxs_neighs.shape[1])[:n_neighs] for _ in range(idxs_neighs.shape[0])]).ravel()
        idxs_neighs = idxs_neighs[idx_rows, idx_cols]

        pxy_neigh = atomic_stresses[idxs_neighs, :, 0]
        pyz_neigh = atomic_stresses[idxs_neighs, :, 1]
        pzx_neigh = atomic_stresses[idxs_neighs, :, 2]
        p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

        starting_frame = traj_frame * stress_traj_ratio
        acf_matrix[i] = np.mean(correlation(p_noncenter_cluster_atoms[:, starting_frame:(starting_frame + sample_size)],
                                            p_neigh[:, starting_frame:(starting_frame + sample_size)],
                                            n_percentile=n_perc), axis=0)

    return np.mean(acf_matrix, axis=0)  # Mean over the `n_samples` correlations


def ncc_visc_contribution(cluster_dict, atomic_stresses, traj, box_length, n_bulk=50, n_samples=10, n_bootstraps=20,
                           n_perc=1, divisor=1.3, dt=0.005, dump_freq=4):
    ncc_acf_ = ncc_acf(cluster_dict, atomic_stresses,
                       n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)
    nccneigh_ccf_ = nccneigh_ccf(cluster_dict, atomic_stresses, traj, box_length,
                                 n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)
    nccnonneigh_ccf_ = nccnonneigh_ccf(cluster_dict, atomic_stresses, traj, box_length,
                                       n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)

    dtau = dt * dump_freq
    ncc_acf_integral = trapezoid(ncc_acf_, dx=dtau)
    nccneigh_ccf_integral = trapezoid(nccneigh_ccf_, dx=dtau)
    nccnonneigh_ccf_integral = trapezoid(nccnonneigh_ccf_, dx=dtau)

    # Contributions: 1 ACF, n_neighs neighbor CCF, N_atoms - (clustersize) bulk CCF
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)
    n_neighs = len(cluster_dict[cluster_center[0]])
    contributions = np.array([1 * ncc_acf_integral,
                              n_neighs * nccneigh_ccf_integral,
                              (n_atoms - 1 - n_neighs) * nccnonneigh_ccf_integral])

    return [np.round(np.abs(contributions) / np.abs(contributions).sum() * 100, decimals=2), contributions]

#----------------------------------------------------------------------------------------------------------------------#
#                                                BULK FUNCTIONS                                                        #
#----------------------------------------------------------------------------------------------------------------------#

def bulk_acf(cluster_dict, atomic_stresses, n_samples=50, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the autocorrelation of the bulk atoms, i.e. atoms that are not part of any cluster
    """
    n_atoms = atomic_stresses.shape[0]
    bulk_atoms = find_bulk_atom(cluster_dict, n_atoms)[:n_samples]

    pxy = atomic_stresses[bulk_atoms, :, 0]
    pyz = atomic_stresses[bulk_atoms, :, 1]
    pzx = atomic_stresses[bulk_atoms, :, 2]
    p_stacked = np.vstack((pxy, pyz, pzx))

    return bootstrap_correlation(p_stacked, p_stacked, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)

def bulkneigh_ccf(cluster_dict, atomic_stresses, traj, box_length, n_samples=25, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the cross correlation between bulk atoms and its `n_neighs` nearest neighbors at the specific time
       where the correlation is being computed
    """
    n_atoms = atomic_stresses.shape[0]
    n_stresses = atomic_stresses.shape[1]
    n_trajs = traj.shape[1]
    stress_traj_ratio = (n_stresses-1)//(n_trajs-1)


    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)
    n_neighs = len(cluster_dict[cluster_center[0]])
    bulk_atoms = find_bulk_atom(cluster_dict, n_atoms)[:n_samples]
    bulk_atoms_rep = np.repeat(bulk_atoms, n_neighs)

    pxy_bulk = atomic_stresses[bulk_atoms_rep, :, 0]
    pyz_bulk = atomic_stresses[bulk_atoms_rep, :, 1]
    pzx_bulk = atomic_stresses[bulk_atoms_rep, :, 2]
    p_bulk = np.vstack((pxy_bulk, pyz_bulk, pzx_bulk))

    # Bootstrap correlation using the nearest neighbors of the chosen bulk atoms at `n_samples` frames
    sample_size = int(n_stresses // divisor)
    acf_size = int(sample_size * n_perc / 100)
    acf_matrix = np.zeros((n_bootstraps, acf_size), dtype=float)
    max_traj_frame = int(n_trajs * (n_stresses-sample_size)/n_stresses)
    for i in tqdm(range(n_bootstraps)):
        traj_frame = np.random.randint(0, max_traj_frame)

        # Finding the index of the nearest neighbors of the chosen bulk atoms
        dist_bulk = construct_distance_table(traj[bulk_atoms, traj_frame, :],
                                             traj[:, traj_frame, :],
                                             box_length)
        dist_bulk[np.isclose(dist_bulk, 0)] = np.inf
        idxs_neighs = np.argsort(dist_bulk, axis=1)[:, :n_neighs].reshape(-1)

        pxy_neigh = atomic_stresses[idxs_neighs, :, 0]
        pyz_neigh = atomic_stresses[idxs_neighs, :, 1]
        pzx_neigh = atomic_stresses[idxs_neighs, :, 2]
        p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

        starting_frame = traj_frame * stress_traj_ratio
        acf_matrix[i] = np.mean(correlation(p_bulk[:, starting_frame:(starting_frame + sample_size)],
                                            p_neigh[:, starting_frame:(starting_frame + sample_size)],
                                            n_percentile=n_perc), axis=0)

    return np.mean(acf_matrix, axis=0)  # Mean over the `n_samples` correlations


def bulkbulk_ccf(cluster_dict, atomic_stresses, n_samples=25, n_bulk=50, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the crosscorrelation of the cluster centers with non-neighbor atoms
    """
    n_atoms = atomic_stresses.shape[0]

    bulk_ids = np.random.permutation(find_bulk_atom(cluster_dict, n_atoms))
    bulk1_ids = bulk_ids[:n_samples]
    bulk1_rep = np.repeat(bulk1_ids, n_bulk)

    bulk2_ids = np.setdiff1d(bulk_ids, bulk1_ids)[:n_bulk]
    bulk2_rep = np.tile(bulk2_ids, n_samples)

    pxy_center = atomic_stresses[bulk1_rep, :, 0]
    pyz_center = atomic_stresses[bulk1_rep, :, 1]
    pzx_center = atomic_stresses[bulk1_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_neigh = atomic_stresses[bulk2_rep, :, 0]
    pyz_neigh = atomic_stresses[bulk2_rep, :, 1]
    pzx_neigh = atomic_stresses[bulk2_rep, :, 2]
    p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

    return bootstrap_correlation(p_center, p_neigh, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def bulkany_ccf(cluster_dict, atomic_stresses, n_samples=25, n_bulk=50, n_bootstraps=50, n_perc=5, divisor=1.3):
    """Computes the cross-correlation of bulk atoms with other atoms in the system
    """
    n_atoms = atomic_stresses.shape[0]
    possible_ids = np.arange(0, n_atoms, 1)

    bulk_ids = np.random.permutation(find_bulk_atom(cluster_dict, n_atoms))
    bulk1_ids = bulk_ids[:n_samples]
    bulk1_rep = np.repeat(bulk1_ids, n_bulk)

    any_ids = np.random.permutation(np.setdiff1d(possible_ids, bulk1_ids))[:n_bulk]
    any_rep = np.tile(any_ids, n_samples)

    pxy_center = atomic_stresses[bulk1_rep, :, 0]
    pyz_center = atomic_stresses[bulk1_rep, :, 1]
    pzx_center = atomic_stresses[bulk1_rep, :, 2]
    p_center = np.vstack((pxy_center, pyz_center, pzx_center))

    pxy_neigh = atomic_stresses[any_rep, :, 0]
    pyz_neigh = atomic_stresses[any_rep, :, 1]
    pzx_neigh = atomic_stresses[any_rep, :, 2]
    p_neigh = np.vstack((pxy_neigh, pyz_neigh, pzx_neigh))

    return bootstrap_correlation(p_center, p_neigh, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def bulk_total_cf(cluster_dict, atomic_stresses, n_samples=1, n_bootstraps=20, n_perc=5, divisor=1.3):
    """Correlation between bulk atom and all other atoms
    """
    n_atoms = atomic_stresses.shape[0]
    bulk_atoms = find_bulk_atom(cluster_dict, n_atoms)[:n_samples]

    bulk_atoms_rep = np.repeat(bulk_atoms, n_atoms)
    pxy_bulk = atomic_stresses[bulk_atoms_rep, :, 0]
    pyz_bulk = atomic_stresses[bulk_atoms_rep, :, 1]
    pzx_bulk = atomic_stresses[bulk_atoms_rep, :, 2]
    p_bulk = np.vstack((pxy_bulk, pyz_bulk, pzx_bulk))

    pxy_all = np.tile(atomic_stresses[:, :, 0], (n_samples, 1))
    pyz_all = np.tile(atomic_stresses[:, :, 1], (n_samples, 1))
    pzx_all = np.tile(atomic_stresses[:, :, 2], (n_samples, 1))
    p_all = np.vstack((pxy_all, pyz_all, pzx_all))

    return bootstrap_correlation(p_bulk, p_all, n_perc=n_perc, n_samples=n_bootstraps, divisor=divisor)


def bulk_visc_contribution(cluster_dict, atomic_stresses, traj, box_length, n_bulk=50, n_samples=10, n_bootstraps=20,
                           n_perc=1, divisor=1.3, dt=0.005, dump_freq=4):
    b_acf = bulk_acf(cluster_dict, atomic_stresses,
                     n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)
    bn_ccf = bulkneigh_ccf(cluster_dict, atomic_stresses, traj, box_length,
                           n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)
    bb_ccf = bulkany_ccf(cluster_dict, atomic_stresses,
                              n_bulk=n_bulk, n_samples=n_samples, n_bootstraps=n_bootstraps, n_perc=n_perc, divisor=divisor)

    dtau = dt * dump_freq
    b_acf_integral = trapezoid(b_acf, dx=dtau)
    bn_ccf_integral = trapezoid(bn_ccf, dx=dtau)
    bb_ccf_integral = trapezoid(bb_ccf, dx=dtau)

    # Contributions: 1 ACF, n_neighs neighbor CCF, N_atoms - (clustersize) bulk CCF
    n_atoms = atomic_stresses.shape[0]
    cluster_center = np.fromiter(cluster_dict.keys(), dtype=int)
    n_neighs = len(cluster_dict[cluster_center[0]])
    contributions = np.array([1 * b_acf_integral,
                              n_neighs * bn_ccf_integral,
                              (n_atoms - 1 - n_neighs) * bb_ccf_integral])

    return [np.round(np.abs(contributions) / np.abs(contributions).sum() * 100, decimals=2), contributions]
#----------------------------------------------------------------------------------------------------------------------#


lmp_data = 'C:/Users/Vitor/Desktop/Simulations/phd/lj/ljcluster.txt'
st_dump = 'C:/Users/Vitor/Desktop/Simulations/phd/lj/stress_lj.dump'
traj_dump = 'C:/Users/Vitor/Desktop/Simulations/phd/lj/0_100_traj_lj.dump'

cluster_dict = extract_cluster(lammps_data=lmp_data)
atstresses = extract_atstress(stress_dump=st_dump, n_frames=int(50000/4)+1) # (n_atoms, n_frames, 3)
# coords = extract_coords(traj_dump, n_frames=int(50000/4)) #


# 1) Autocorrelation for the cluster center with itself (DONE)
# 2) Cross correlation between the cluster center and its n_cluster neighbors
# 3) Cross correlation between all members in the cluster
# 4) Average total contribution of a cluster atom, i.e. correlation with all other atoms (DONE)

# What I think happens is that due to diffusion the nearest neighbors of a bulk atoms is constantly changing. Hence,
# they don't have this "consistent correlation" contributing to the viscosity. Thus, diffusion is the means by which
# stress correlation is destroyed, but since in a cluster its atoms cannot diffuse to far away due to a force binding
# them together, the stress correlation is larger.

#----------------------------------------------------------------------------------------------------------------------#
# Comparing the ACF of the cluster centers with bulk atoms yields the same result. Thus, the enhanced viscosity
# doesn't come from the enhanced autocorrelation of cluster members. Thus, it must come from the crosscorrelation
# of cluster members.

cc_acf = cluster_acf(cluster_dict, atstresses)
b_acf = cluster_acf(cluster_dict, atstresses)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cc_acf, color='navy', label='cluster center atoms', alpha=0.7, ls='--', lw=2.0)
ax.plot(b_acf, color='orangered', label='bulk atoms', alpha=0.7, ls='-.', lw=2.0)
ax.axhline(y=0, xmin=0, xmax=cc_acf.shape[0], ls='-', color='black', alpha=0.8)
ax.set_xlabel('Frame')
ax.set_ylabel('Autocorrelation')
ax.legend()
# fig.savefig(f'C:/Users/Vitor/Desktop/bresa/Python/scripts/zlab/plots_saved/phd/lj/bulk_vs_cluster_ACF.png', dpi=500)
# fig.savefig(f'C:/Users/Vitor/Desktop/bresa/Python/scripts/zlab/plots_saved/phd/lj/bulk_vs_cluster_ACF.pdf', dpi=500)
plt.show()

quit()
#----------------------------------------------------------------------------------------------------------------------#
# Looking at cross correlations with `N_cluster` nearest neighbors for bulk and cluster center atoms.


#----------------------------------------------------------------------------------------------------------------------#
# Looking at the total contribution of each atom, i.e. autocorrelation and cross correlation with all other atoms

#----------------------------------------------------------------------------------------------------------------------#
# Compute cross-correlation as a function of bond strength (i.e. different `k` values).
