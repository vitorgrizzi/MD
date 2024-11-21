import numpy as np
import matplotlib.pyplot as plt

dict_key = {'1': 'F', '2': 'Li', '3': 'Na', '4': 'K'}
xyz_f = []
with open("C:/Users/Vitor/Downloads/snap.txt", 'r') as f:
    for row in f:
        line = row.split()[1:]
        line[0] = dict_key[line[0]]
        xyz_f.append(' '.join(line) + ' ')


# with open("xyz_f.txt", 'w') as f:
#     f.writelines("\n".join(xyz_f))

with open("xyz_dft_new.txt", 'w') as f_n:
    with open("C:/Users/Vitor/Downloads/xyz_dft.txt", 'r') as f:
        for row in f:
            row_list = row.split()
            if len(row_list) == 4:
                coords = [f'{float(el) * 12.12:.5f}' for el in row_list[1:]]
                row_list[1:] = coords
                print(row_list)
                f_n.write(' '.join(row_list))
            else:
                f_n.write(row)
            f_n.write('\n')
quit()

def get_full_trajectory(traj_path, n_atoms=795, n_frames=402):
    """ Calculates the average distance between nearest neighbors
    """
    coords = []
    with open(traj_path, 'r') as f:
        for line in f:
            if len(line.split()) == 5: # this only works if the traj contains only [id, at_type, x, y, z]
                coords.append(list(map(float, line.split()[2:])))

    return np.array(coords).reshape(n_frames, n_atoms, 3)


def get_thorium(traj_path, n_th=4000, n_total_atoms=100000, n_frames=1000):
    """ Calculates the average distance between nearest neighbors
    """
    coords = []
    aux_coords = []
    with open(traj_path, 'r') as f:
        for line in f:
            if len(line.split()) == 5: # this only works if the traj contains only [id, at_type, x, y, z]
                aux_coords.append(list(map(float, line.split()[2:])))
                if len(aux_coords) == n_total_atoms:
                    coords.append(aux_coords[-4000:])
                    aux_coords = []

    return np.array(coords).reshape(n_frames, n_th, 3)


temperatures = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
# temperatures = np.array([200, 300, 400])
avg_dist_temperature = np.zeros(temperatures.shape[0], dtype=float)
std_dist_temperature = np.zeros(temperatures.shape[0], dtype=float)

for k, T in enumerate(temperatures):
    traj = get_full_trajectory(f'C:/Users/Vitor/Desktop/Simulations/graphene/reaxff/{T}k/graphene_single_layer_ReaxFF_795_C_temp_{T}K.lammpstrj')
    avg_frames = np.zeros(traj[300:].shape[0]) # Using only fram 300 onwards.
    for j, frame in enumerate(traj[300:]):
        neigh_matrix = np.zeros(traj.shape[1])
        for i, atom in enumerate(frame):
            dist = np.linalg.norm(atom - frame, axis=1)
            neigh_matrix[i] = np.mean(np.sort(dist)[1:4])
        avg_frames[j] = np.mean(neigh_matrix)

    avg_dist_temperature[k] = np.mean(avg_frames)
    std_dist_temperature[k] = np.std(avg_frames)
    print(k)

# fig, ax = plt.subplots()
# ax.errorbar(temperatures, avg_dist_temperature, yerr=std_dist_temperature, ls='none', color='tab:red', marker='o', markersize=7)
# ax.set_ylim([1.4, 1.57])
# ax.set_xlabel('Temperature (K)')
# ax.set_ylabel('Average C-C Bond Length')
# fig.savefig(f'C:/Users/Vitor/Desktop/bresa/UIUC/Courses/22.2 - Fall/CSE 485/bond_length.png', dpi=500)
# plt.show()

