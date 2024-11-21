import numpy as np
import os
import timeit
from itertools import islice
from MD import construct_distance_table, min_dist, to_mic
import matplotlib.pyplot as plt

def get_box_volume(xdatcar_path):
    """
        Get the box volume of an NVT simulation

        OBS: As of right now, it only works if the box is cubic
    """
    a = np.sum(np.loadtxt(xdatcar_path, skiprows=2, max_rows=3), axis=1)

    return np.prod(a)


def get_box_length(xdatcar_path):
    """
        Get the length of the cubic box
    """

    return get_box_volume(xdatcar_path)**(1/3)


def get_num_atoms(xdatcar_path):
    """ Get the total number of atoms in the simulation cell
    """
    return int(np.sum(np.loadtxt(xdatcar_path, skiprows=6, max_rows=1)))


def extract_coordinates(xdatcar_path, frame_interval=1):
    """Extract coodinates of VASP's XDATCAR file.

    Args:
        xdatcar_path (str): Path of the XDATCAR file.
        frame_interval (int): Only store each 'frame_interval' frames.

    Returns:
         (ndarray): A 3-D array of shape (N, n_atoms, 3)  where 'N' is the number of frames.

    OBS1: I've run this script to calculate the minimum distance between any two atoms during the MD. For FLiBe, at
          temperatures [800 K,1200 K] I found ~1.16 at 1200 K. Naturally, as temperature increases the minimum
          distance decreases.
    """
    n_atoms = get_num_atoms(xdatcar_path)
    with open(xdatcar_path, 'r') as xdatcar:
        data = np.array(xdatcar.readlines()[7:], dtype='object')

    n_frames = int(data.shape[0] / (n_atoms+1))
    idxs_to_rmv = np.arange(0, n_frames*(n_atoms+1), n_atoms+1) # indexes that contain "Direct ..."
    data = np.delete(data, idxs_to_rmv, axis=0)
    data = np.array( [row.rstrip('\n').split() for row in data], dtype=float ).reshape(n_frames, n_atoms, 3)

    return data[::frame_interval]


def find_min_dist(xdatcar_path, plot_hist=False):
    """Finds the minimum distance for each atom over the couse of the simulation and the minimum overall distance
       reached in the simulation"""
    L = get_box_length(xdatcar_path)
    coordinates = extract_coordinates(xdatcar_path) * L # Index [:,:50,:] to get distances of a specific species

    min_dists = np.array([min_dist(coord, L)[1] for coord in coordinates])
    min_dist_per_atom = np.min(min_dists, axis=0)

    if plot_hist:
        hist_dist = np.histogram(min_dists.reshape(-1,1), np.arange(1.2, 2.5, 0.05))
        fig, ax = plt.subplots()
        ax.hist(min_dists.reshape(-1,1), bins=hist_dist[1], rwidth=0.95, color='navy', edgecolor='black')

        for i, rect in enumerate(ax.patches):
            height = rect.get_height()
            ax.annotate(f'{int(hist_dist[0][i])}', xy=(rect.get_x()+rect.get_width()/2, height),
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=15, color='orangered')
        plt.show()

    return min_dist_per_atom, min(min_dist_per_atom)

# flibe = "C:/Users/Vitor/Desktop/Simulations/phd/flibe/1200k/eq/XDATCAR" # 1.18
# flinath = "C:/Users/Vitor/Desktop/Simulations/ms/flinath/1473k/XDATCAR_total" # 1.32
# flinau = "C:/Users/Vitor/Desktop/Simulations/ms/flinau/1473k/XDATCAR_total" # 1.35
# flina = "C:/Users/Vitor/Desktop/Simulations/ms/flina/1273k/XDATCAR" # 1.36
# dists, abs_min_dist = find_min_dist(flina, plot_hist=False)
# F_dist = dists[:50]
# Li_dist = dists[50:80]
# Na_dist = dists[80:]
# colors = ['navy', 'purple', 'orangered']
# atoms = ['F', 'Li', 'Na']
# atoms_dists = [F_dist, Li_dist, Na_dist]
# print(F_dist)
# print(Li_dist)
# print(Na_dist)
# fig, ax = plt.subplots()
# for i in range(3):
#     # hist_dist = np.histogram(at_dists[i], np.arange(min(dists)*0.95, max(dists)*1.05, 0.01))
#     ax.hist(atoms_dists[i], bins=np.arange(min(dists)*0.95, max(dists)*1.05, 0.01), rwidth=0.95, color=colors[i],
#             alpha=0.5, label=atoms[i], edgecolor='black')
# plt.legend()
# plt.show()
# quit()

def concatenate_xdatcar(xdatcar_list, common_path):
    """ Concatenates XDATCAR files. We usually want to do that to get a better statistics in liquidlib.

        OBS: do with os to get current directory
    """
    common_path = common_path.rstrip('/')
    xdatcar_list = [common_path + '/' + el for el in xdatcar_list]
    write_header = True
    frame = 1

    with open(f'{common_path}/XDATCAR_total', 'w') as xdatcar_total:
        for file in xdatcar_list:
            with open(file, 'r') as xdatcar:
                for i, line in enumerate(xdatcar):
                    if i < 7 and write_header is False: # This is needed to avoid writing the header of each XDATCAR
                        continue
                    if 'Direct configuration=' in line:
                        blank_spaces = ' ' * (6 - len(str(frame)))
                        xdatcar_total.write(f'Direct configuration={blank_spaces + str(frame)}' + '\n')
                        frame += 1
                    else:
                        xdatcar_total.write(line)
            write_header = False

    return None

# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/973k')
# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/1073k')
# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/1173k')
# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/1273k')
# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/1373k')
# concatenate_xdatcar(['XDATCAR1', 'XDATCAR2', 'XDATCAR3'], f'C:/Users/Vitor/Desktop/Simulations/ms/flinau/1473k')
# quit()

def shift_coordinates(xdatcar_path, x_shift=0, y_shift=0, z_shift=0, fractional_shift=False):
    """Creates a new xdatcar file with the coordinates translated by a given (x,y,z).
    """
    common_path = xdatcar_path.rstrip('/XDATCAR')

    n_atoms = get_num_atoms(xdatcar_path)
    box_length = get_box_length(xdatcar_path)
    shift = np.array([x_shift, y_shift, z_shift]) if fractional_shift else np.array([x_shift, y_shift, z_shift]) / box_length
    coord_matrix = []

    with open(f'{common_path}/XDATCAR_shifted', 'w') as xdatcar_shifted:

        with open(xdatcar_path, 'r') as xdatcar:

            ## Method 1 (5 runs, 18.8917138 seconds)
            for i, line in enumerate(xdatcar):
                if i < 7: # writing header
                    xdatcar_shifted.write(line)
                elif 'Direct configuration=' in line:
                    xdatcar_shifted.write(line)
                else:
                    coord_matrix.append(line.split())
                    if len(coord_matrix) == n_atoms:
                        # Translating, wrapping, and rounding coordinates
                        shifted_matrix = np.array(coord_matrix, dtype=float) + shift
                        shifted_matrix = np.where((shifted_matrix > 1) | (shifted_matrix < 0), shifted_matrix % 1, shifted_matrix)
                        shifted_matrix = np.around(shifted_matrix, 6)
                        for row in shifted_matrix: # this is just to format the rows
                            formatted_row = '  '.join(map(str, row))
                            xdatcar_shifted.write('   ' + formatted_row + '\n')
                        coord_matrix = []

            ## Method 2 (veeeery slow)
            # for i, line in enumerate(xdatcar):
            #     if i < 7: # writing header
            #         xdatcar_shifted.write(line)
            #     else:
            #         break
            # for i in range(5):
            #     coord_matrix = np.loadtxt(xdatcar_path, skiprows=(n_atoms + 1)*i + 8, max_rows=n_atoms)
            #     shifted_matrix = coord_matrix + shift
            #     print(coord_matrix[0], shifted_matrix[0])
            #     shifted_matrix = np.where((shifted_matrix > 1) | (shifted_matrix < 0), shifted_matrix % 1, shifted_matrix)
            #     print(shifted_matrix[0])
            #     blank_spaces = ' ' * (6 - len(str(i+1)))
            #     xdatcar_shifted.write(f'Direct configuration={blank_spaces + str(i+1)}' + '\n')
            #     np.savetxt(xdatcar_shifted, shifted_matrix)


            ## Method 3: using islice (5 runs, 18.7293341 seconds):
            # for i, line in enumerate(xdatcar):
            #     if i < 7: # writing header
            #         xdatcar_shifted.write(line)
            #     elif 'Direct configuration=' in line:
            #         xdatcar_shifted.write(line)
            #         coord_matrix = islice(xdatcar, 0, n_atoms) # 0 because it takes into account the pos of the cursor
            #         coord_matrix = [row.rstrip('\n').split() for row in coord_matrix]
            #
            #         # Translating and wrapping coordinates, and then writing into the new file
            #         shifted_matrix = np.array(coord_matrix, dtype=float) + shift
            #         shifted_matrix = np.where((shifted_matrix > 1) | (shifted_matrix < 0), shifted_matrix % 1, shifted_matrix)
            #         shifted_matrix = np.around(shifted_matrix, 6)
            #
            #         for row in shifted_matrix:
            #             # To format the number of zeros another for loop is needed
            #             formatted_row = '  '.join(map(str, row))
            #             xdatcar_shifted.write('   ' + formatted_row + '\n')
            #         coord_matrix = []

    return None


# shift_coordinates('C:/Users/Vitor/Desktop/Simulations/531/4Cr4C/eq/XDATCAR', x_shift=-4, y_shift=2, z_shift=6, n_frames=9904)
# print(timeit.timeit("shift_coordinates('C:/Users/Vitor/Desktop/Simulations/531/4Cr4C/eq/XDATCAR)', x_shift=-4, y_shift=2, z_shift=6, n_frames=9904)",
#                     globals=globals(), number=5))


def unwrap_coordinates(xdatcar_path):
    """Returns a new XDATCAR file with the unwrapped coordinates

        To do: 1) Create a displacement vector matrix r_ij thorugh frame[i+1] - frame[i]
               2) Check np.any(abs(r_ij) > box_length/2)
               3) For the displacements greater than box_length/2 that are negative we add box_length; for those that
                  are positive we subtract box_length i.e. np.where(r_ij < box_length/2, r_ij,
               OBS1: We will need to open the XDATCAR in read file and write another XDATCAR_unwrapped while we read
                     the XDATCAR. There will be to variables current_frame and previous_frame of the XDATCAR file that
                     we gonna keep track of, as well as unwrapped_frame which will be written to XDATCAR_unwrapped.
    """
    # Unwrapping coordinates
    common_path = xdatcar_path.rstrip('/XDATCAR')
    prev_coords_matrix = []
    coords_matrix = []
    box_length = get_box_length(xdatcar_path)
    n_atoms = get_num_atoms(xdatcar_path)
    with open(f'{common_path}/XDATCAR_unwrapped', 'w') as xdatcar_unwrapped:

        with open(xdatcar_path, 'r') as xdatcar:

            for i, line in enumerate(xdatcar):
                if i < 7: # writing header
                    xdatcar_unwrapped.write(line)
                elif 'Direct configuration=' in line:
                    xdatcar_unwrapped.write(line)
                else:
                    coord = list(map(float, line.rstrip('\n').split()))

    pass

# unwrap_coordinates('C:/Users/Vitor/Desktop/Simulations/ms/flina/1273k/eq/XDATCAR')


def count_neighbors(xdatcar_path, atomic_system, r_cutoff, atom_type1, atom_type2=None):
    """
        Counts the number of 'B' atoms whose distance from the reference atoms 'A' is less than the defined cutoff. In
        other words, counts the number of 'B' atoms inside a sphere of radius r=cutoff using the atoms 'A' as origin.

        Args:
            xdatcar_path (str): Path for the XDATCAR file
            atomic_system (dict): Dictionary {atom_type: count} containing how many atoms of each type there are
            r_cutoff (float): Maximum distance that we want to consider when counting the neighbors
            atom_type1 (str): Reference atom type
            atom_type2 (str): Atom type surrounding the reference atom that we wish to count

        Return:
            (float): Coordination number
    """
    # very inefficient, work on this code to:
    # 1) Manage different atom counts, this code only works for the coord number of atom_type1 around itself. This is
    #    easy, just extract coords of atom 1 and atom 2 into two different matrices and find the pairwise distances up
    #    to the cutoff
    # 2) Add the start/end index to the coord matrix based on the atom type because now it is simply gets the first (n_count values)
    #    so if the atom that we want to count is in the middle of the coordinate it wont work
    # 3) avoid append
    coord_num = []
    coord_matrix = []
    n_count = atomic_system[atom_type1]
    box_length = get_box_length(xdatcar_path)

    with open(xdatcar_path, 'r') as xdatcar:
        for i, line in enumerate(xdatcar):
            if i > 7:
                if 'Direct configuration=' in line:
                    coord_matrix = np.array(coord_matrix)[:n_count, :] * box_length
                    # coord_matrix = np.array(coord_matrix) * box_length
                    dist_table = construct_distance_table(coord_matrix, box_length)
                    coord_num.append(np.mean(np.sum((dist_table > 0) & (dist_table < r_cutoff), axis=1)))
                    # coord_matrix = []
                    # print(np.min(dist_table[dist_table !=0], axis=None))
                    coord_matrix = []
                else:
                    coord = list(map(float, line.rstrip('\n').split()))
                    coord_matrix.append(coord)

    return np.mean(coord_num)

# print(count_neighbors('C:/Users/Vitor/Desktop/Simulations/ms/flina/1273k/eq/XDATCAR', {'F': 50 , 'Li':30 , 'Na': 20}, 4.44, 'F'))

## Creates an histogram of particles distances of a single MD frame.
# coords = []
# with open("C:/Users/Vitor/Downloads/snap3.txt", 'r') as f:
#     for row in f:
#         line = row.split()
#         line = [float(el) for el in line]
#         coords.append(line)
# # L = 1320.23**(1/3) # flibe 1100k
# L = 1783.44**(1/3) # flinak 1100k
# coords = np.array(coords) * L
# d_ijk = to_mic(coords[:,np.newaxis,:] - coords[np.newaxis,:,:], L)
# unique_vecdist = np.triu(np.swapaxes(np.swapaxes(np.swapaxes(d_ijk,1,2),0,2)[:,:,::-1],0,1)[:,::-1,:], k=1).reshape(-1)
# dists = np.linalg.norm(d_ijk, axis=2).reshape(-1)
# dists = dists[dists!=0]
# #
# hist_dist = np.histogram(dists, np.arange(0, L, 0.2))
# fig, ax = plt.subplots()
# ax.hist(dists, bins=hist_dist[1], rwidth=0.95, color='navy')
# plt.show()
# quit()
#
# def generate_kvecs(max_n, box_length):
#   k = 2 * np.pi / box_length * np.linspace(0, max_n, max_n+1)
#   k_vecs = np.array(np.meshgrid(k,k,k)).T.reshape(-1,3) # [(x,y,z) for x in k for y in k for z in k]
#   k_vecs[:, [1,2]] = k_vecs[:, [2,1]]
#   k_vecs[:, [0,1]] = k_vecs[:, [1,0]]
#   return k_vecs
#
# def calculate_rho_k(k_vecs, pos):
#     kr = k_vecs @ pos.T
#     return np.sum(np.cos(kr) - 1.0j * np.sin(kr), axis=1)
#
# def calculate_structure_factor(kvecs, pos):
#     return (calculate_rho_k(kvecs, pos) * calculate_rho_k(-kvecs, pos)).real / pos.shape[0]
#
# def calc_sk(k_vecs, coords):
#     dist_vecs = to_mic(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], L)
#     for k in k_vecs:
#         for dists in dist_vecs:
#             np.dot(dists, k)
#
#         sk.append(np.cos(kr) - 1.0j * np.sin(kr))
#     return sk
#
# k_vecs = generate_kvecs(6, L)
# k_mag = np.linalg.norm(k_vecs, axis=1).round(8)
# unique_k_mag = np.unique(k_mag)
# unique_sk = np.zeros_like(unique_k_mag)
#
# # s_k1 = calc_sk(k_vecs, d_ijk)
# s_k2 = calculate_structure_factor(k_vecs, coords)
# # print(s_k1)
# # print(s_k2)
# # quit()
#
# # unique_sk = np.array([s_k[np.nonzero(unique_k_mag[i] == k_mag)].mean() for i in range(unique_k_mag.shape[0]])
# for i in range(unique_k_mag.shape[0]):
#     idx = np.nonzero(unique_k_mag[i] == k_mag)
#     unique_sk[i] = s_k2[idx].mean()
#
# fig, ax = plt.subplots()
# ax.plot(unique_k_mag[1:], unique_sk[1:], color='orangered')
# ax.set_xlabel('k (arb)')
# ax.set_ylabel('S(k)')
# ax.set_title('Structure Factor')
# # fig.savefig(f'C:/Users/Vitor/Desktop/bresa/UIUC/Courses/22.2 - Fall/CSE 485/Homework/3/p3_sk_T{T}.pdf', dpi=500)
# plt.show()