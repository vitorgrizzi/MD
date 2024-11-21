import numpy as np
import re
from itertools import islice

def extract_property(outcar_path, property_name, create_file=True, new_file_name=None):
    if new_file_name is None:
        new_file_name = property_name
    property_value = []

    if property_name == 'E0':
        property_name = 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM'
        pos_value = -1

    elif property_name == 'E':
        property_name = 'total energy   ETOTAL'
        pos_value = -2

    elif property_name == 'F':
        property_name = '% ion-electron'
        pos_value = -3

    elif property_name == 'EK': # property_name in ('kinetic energy', 'EK')
        property_name = 'kinetic energy EKIN'
        pos_value = -1

    elif property_name == 'external pressure':
        property_name = 'external pressure'
        pos_value = 3

    elif property_name == 'total pressure':
        property_name = 'total pressure'
        pos_value = -2

    elif property_name == 'volume of cell' or property_name == 'volume':
        property_name = 'volume of cell'
        pos_value = -1

    elif property_name == 'T': # property_name in ('temperature', 'EK')
        property_name = 'EKIN_LAT'
        pos_value = -2

    elif property_name == 'Pxy':
        property_name = 'Total+kin.'
        pos_value = 4

    elif property_name == 'Pyz':
        property_name = 'Total+kin.'
        pos_value = 5

    elif property_name == 'Pzx':
        property_name = 'Total+kin.'
        pos_value = 6

    else:
        raise Exception('Invalid property name')

    with open(outcar_path, 'r') as outcar:
        if property_name == 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM':
            for line in outcar:
                if property_name in line:
                    property_value.append(' '.join(islice(outcar, 3, 4)).split()[-1])

        else:
            for line in outcar:
                if property_name in line:
                    property_value.append(line.split()[pos_value])
        # pattern = r'FREE ENERGIE OF THE ION-ELECTRON SYSTEM[\(\)\.\d\w\n-= ]* energy\(sigma->0\)[ =]*([-\.\d]*)'
        # property_value = re.findall(pattern, outcar.read())

    if property_name == 'volume of cell':
        del property_value[0]

    if create_file:
        new_file_path = outcar_path.rstrip('OUTCAR') + new_file_name
        with open(f'{new_file_path}.txt', 'w') as f:
            f.writelines('\n'.join(property_value))

    return np.array(list(map(float, property_value)))


def extract_all(outcar_path, only_core_properties=True, count_iter=False, temperature=None):
    """ Extract all relevant thermo and physical properties of the OUTCAR file and dump it into different text files.
    """

    key_labels = ['E0', 'E', 'F', 'EK', 'external_pressure', 'total_pressure', 'volume', 'T', 'Pxy', 'Pyz', 'Pzx']
    properties = {key: [] for key in key_labels}

    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line: # This is the "potential energy"
                properties['E0'].append(' '.join(islice(outcar, 3, 4)).split()[-1])

            elif 'kinetic energy EKIN' in line:
                properties['EK'].append(line.split()[-1])

            elif 'total pressure' in line:
                properties['total_pressure'].append(line.split()[-2])

                if count_iter and len(properties['total_pressure']) % 5000 == 0:
                    print(f"{len(properties['total_pressure'])}, {temperature} K")

            elif 'volume of cell' in line:
                properties['volume'].append(line.split()[-1])

            elif 'Total+kin.' in line:
                pressures = line.split()
                properties['Pxy'].append(pressures[4])
                properties['Pyz'].append(pressures[5])
                properties['Pzx'].append(pressures[6])

            if only_core_properties is False:
                if 'total energy   ETOTAL' in line:
                    properties['E'].append(line.split()[-2])

                elif '% ion-electron' in line:
                    properties['F'].append(line.split()[-3])

                elif 'external pressure' in line:
                    properties['external_pressure'].append(line.split()[3])

                elif 'EKIN_LAT' in line:
                    properties['T'].append(line.split()[-2])

    # Asserting that the volume array will have the same number of elements as the others (any other property could be
    # used, I chose total_pressure). This is necessary because depending on how we start the job volume can have two or
    # one extra initial values.
    properties['volume'] = properties['volume'][-len(properties['total_pressure']):]

    for k, v in properties.items():
        if v: # Creating txt files only if the value associated with the key is not an empty list
            new_file_path = outcar_path.rstrip('OUTCAR') + k
            with open(f'{new_file_path}.txt', 'w') as f:
                f.writelines('\n'.join(v))

    return None


def get_values(property_file_name):
    """Read the values from the text file generated from the POSCAR file that contains the desired properties.
    """
    property_value = []
    with open(property_file_name, 'r') as f:
        for line in f:
            property_value.append(float(line))

    return np.array(property_value)


def get_thermostat_mean(outcar_path, ensemble='npt'):
    """Find the mean thermostate temperature over the simulation"""
    with open(outcar_path, 'r') as outcar:
        T = []
        for line in outcar:
            if ensemble.lower() == 'npt':
                if 'EKIN_LAT' in line:
                    T.append(float(line.split()[-2]))
            else:
                if 'mean value of Nose-termostat' in line:
                    return line[-1]

    return np.mean(T)


def get_volume(outcar_path):
    """Get the initial volume of the simulation box"""
    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if 'volume of cell' in line:
                return float(line.split()[-1])

def get_natoms(outcar_path):
    """Get the total number of atoms/ions in the simulation box"""
    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if 'NIONS' in line:
                return int(line.split()[-1])

def get_cell(outcar_path):
    """Get the lattice vectors of the simulation cell"""
    cell = []
    cell_flag = False
    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if cell_flag:
                cell.append(list(map(float, line.split()[:3])))

            if 'direct lattice vectors' in line:
                cell_flag = True

            if len(cell) == 3:
                break

    return np.array(cell)


def check_energy_drift(E0, n_last_frames, window=1000):
    """Checks the drift of the total energy in the simulation. It is desired that ΔE < 2 eV for each 1000 steps or
       ΔE < 0.2 eV for each step. If the drift is too large, the system is not equilibrated or we should decrease the
       timestep (POTIM)
    """
    E = E0[:,-n_last_frames:]
    drift = np.abs(E[:,window:] - E[:,:-window])
    max_drift = np.round(drift.max(axis=1), 2)
    avg_drift = np.round(drift.mean(axis=1), 2)
    print(f'The maximum energy drift was {max_drift} and the average {avg_drift} on the last {n_last_frames} frames with '
          f'a window {window}')

    return None


def extract_forces(outcar_path):
    """Extract the x, y, z forces [eV/A] acting on each ion on each frame.

        Args:
            outcar_path (str): Path of the outcar file
        Returns:
            (n_frames, n_atoms, 3) array containing the x- y- z-components of the force on each atom at each frame
    """
    frame_forces = []
    n_atoms = get_natoms(outcar_path)

    counter = 0
    pos_flag = False
    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if 'POSITION' in line:
                counter = 0
                pos_flag = True
                forces = []
                continue

            if counter > 0 and pos_flag:
                forces.append(line.split()[3:])
                if counter == n_atoms:
                    frame_forces.append(forces)
                    pos_flag = False
            counter += 1

    return np.array(frame_forces, dtype=float)


def extract_coordinates(outcar_path):
    """Extract the x, y, z coordinates of each ion in all frames

        Args:
            outcar_path (str): Path of the outcar file
        Returns:
            (n_frames, n_atoms, 3) array containing the x- y- z-coordinates on each atom at each frame
    """
    n_atoms = get_natoms(outcar_path)
    cell = get_cell(outcar_path)
    box_length = cell.max()

    frame = 1
    counter = 0
    pos_flag = False
    coords_matrix = []
    with open(outcar_path, 'r') as outcar:
        for line in outcar:
            if 'POSITION' in line:
                counter = 0
                frame += 1
                pos_flag = True
                continue

            if counter > 0 and pos_flag:
                # coords_matrix.append(np.array(line.split()[:3], dtype=float))
                coords_matrix.append(line.split()[:3])
                if counter == n_atoms:
                    pos_flag = False
                    # print(coords_matrix)
                    # quit()
            counter += 1

    return np.array(coords_matrix, dtype=float).reshape(-1, n_atoms, 3)

# cords = extract_coordinates("C:/Users/Vitor/Desktop/Simulations/phd/flinazr/1200k/eq/OUTCAR")
# print(cords[0], cords[-1], cords.shape)

def create_xdatcar(outcar_path, atomic_system):
    """Create XDATCAR file from OUTCAR file for a cubic simulation box

        TODO: extend for a general simulation box, just have to figure out how to multiple the fractional coordinates
              using the cell matrix
    """
    atoms = '   '.join(atomic_system.keys())
    num_atoms = '   '.join( map(str, atomic_system.values()) )
    system_name = ''.join([k + str(v) for k,v in atomic_system.items()])
    cell = get_cell(outcar_path)
    box_length = cell.max()
    n_atoms = get_natoms(outcar_path)

    with open('XDATCAR', 'w') as xdatcar:
        n_decimals = len(str(cell.max()).split('.')[-1])

        # Writing XDATCAR header
        xdatcar.write(system_name + '\n')
        xdatcar.write('           1' + '\n')
        for row in cell:
            xdatcar.write('   ')
            for el in row:
                nl = str(el).split('.')[0]
                xdatcar.write(' '*(3-len(nl)) + f'{el:.{n_decimals}f}  ')
            xdatcar.write('\n')
        xdatcar.write('   ' + atoms + '\n')
        xdatcar.write('  ' + num_atoms + '\n')

        frame = 1
        counter = 0
        pos_flag = False
        with open(outcar_path, 'r') as outcar:
            for line in outcar:
                if 'POSITION' in line:
                    counter = 0
                    xdatcar.write('Direct configuration=' + ' '*(6-len(str(frame))) + str(frame) + '\n')
                    frame += 1
                    pos_flag = True
                    continue

                if counter > 0 and pos_flag:
                    frac_coords = np.array(line.split()[:3], dtype=float) / box_length
                    frac_coords = [format(el, '.6f') for el in frac_coords.tolist()]
                    xdatcar.write('   ' + '  '.join(frac_coords) + '\n')
                    if counter == n_atoms:
                        pos_flag = False
                counter += 1

    return None


# create_xdatcar("C:/Users/Vitor/Desktop/Simulations/phd/flinazr/1200k/eq/OUTCAR", atomic_system={'F': 56, 'Li':24 , 'Na': 16, 'Zr':4})
# create_xdatcar("C:/Users/Vitor/Desktop/Simulations/phd/mgcl2/1300k/eq/OUTCAR", atomic_system={'Mg': 33, 'Cl': 66})

## K-mesh study:
# 1x1x1 (1 k-point), 2x2x2 (8 k-points), 3x3x3 (14 k-points) in a L=11.33 Angstrom box.
# x1 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/1x1x1_small/OUTCAR').reshape(-1,3)
# x2 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/2x2x2_small/OUTCAR').reshape(-1,3)
# x3 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/3x3x3_small/OUTCAR').reshape(-1,3)
# abs_diff = np.abs(x1-x3)
# np.set_printoptions(suppress=True)
# print(abs_diff)
# print(np.argmax(abs_diff))
# print(f'Max absolute difference {np.max(abs_diff):.2f} eV/Angs and the average difference per atom {np.sum(abs_diff, axis=1).mean():.3f} eV/Angs')

# 1x1x1 (1 k-point), 2x2x2 (8 k-points), 3x3x3 (14 k-points) in a L=14.28 Angstrom box.
# x1 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/1x1x1_large/OUTCAR').reshape(-1,3)
# x2 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/2x2x2_large/OUTCAR').reshape(-1,3)
# x3 = extract_forces('C:/Users/Vitor/Desktop/Simulations/kgrid_test/3x3x3_large/OUTCAR').reshape(-1,3)
# abs_diff = np.abs(x1-x2)
# np.set_printoptions(suppress=True)
# print(abs_diff)
# print(f'Max absolute difference {np.max(abs_diff):.2f} eV/Angs and the average difference per atom {np.sum(abs_diff, axis=1).mean():.3f} eV/Angs')