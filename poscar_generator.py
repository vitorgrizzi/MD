import numpy as np
from MD import to_mic, box_dimension_estimator, generate_coordinates

def generate_coordinates_old(n_atoms, box_length, min_distance=2.0, decimal_digits=6, scale_coords=True, save_txt=False):
    """
    Generates relative random coordinates for VAPS's POSCAR file. These coordinates are relative to the box/cell size,
    i.e., the real coordinates are box_param * coords.

    n_atoms => Number of atom in the system and thus the number of coordinates that will be generated.
    box_length => The parameter of the cubic box where the simulation takes place, where the volume of the box is equal
                  to (box_length)**3
    min_distance => The minimum distance between two atoms in Angstrom. If two atoms are too close to each other VASP
                    will crash because the repulsion force between the atoms will be very large (-∇V will be too high).
                    Thus, when we randomly generate coordinates, we have to check this.
    decimal_digits => Controls the number of decimal digits of the coordinates
    scale_coords => Controls whether to scale the coordinates by '1/box_length' or not. If its 'True', all coordinates
                    will be within [0,1]
    save_txt => If set to 'True', a text file will be created with the coordinates

    Returns a coordinate matrix (n_atoms, 3)

    OBS1: We want to create a smaller box inside the original box that is offsetted in all directions 'x', 'y', 'z' by
         'min_distance/2' to account for the boundary conditions and make sure that the atoms close to the boundaries
          won't be closer than this minimum distance when the minimum image convention (MIC) is invoked in the MD code.
          For instance, if the box length is 10 and the min_distance is 2 then the coordinates of each atom on all
          directions have to always be within [1,9] when we initialize the positions to avoid dealing with MIC which
          will make the code more complicated. After the MD simulation starts, the code will worry about MIC for us.

    """
    coords = np.zeros((n_atoms, 3))
    for n in range(n_atoms):
        overlap_flag = True
        while overlap_flag is True:
            ## Method 1: Placing the atoms on a smaller box to deal with PBC. To understand the lines below, first
            #            picture the simulation box. The first line scales the coordinates range from [0, box_length]
            #            to [0, box_length - min_distance] on all directions. Then, we translate or shift this scaled
            #            coordinates on all directions by adding 'min_abs_distance/2'. We can think of this addition
            #            as an displacement vector r = min_distance/2 i + min_distance/2 j + min_distance/2 k that is
            #            added to the coordinate vector to offset it from the origin. After these steps it is as if we
            #            placed the atoms inside a smaller box that is offsetted from the origin by min_distance/2.
            # candidate_coord = (np.random.random_sample(3) * (box_length - min_distance)) + (min_distance / 2)
            # distances = np.linalg.norm(coords[:n, :] - candidate_coord, axis=1)
            # if np.all(distances > min_distance):
            #     coords[n] = candidate_coord
            #     overlap_flag = False

            # Method 2: Using minimum image convention to check PBC distances (faster, especially for higher densities)
            candidate_coord = np.random.random_sample(3) * box_length
            distances = np.linalg.norm(to_mic(coords[:n, :] - candidate_coord, box_length), axis=1)
            if np.all(distances > min_distance):
                coords[n] = candidate_coord
                overlap_flag = False

    coords = np.around(coords, decimal_digits)
    rng = np.random.default_rng()
    rng.shuffle(coords, axis=0)

    if scale_coords:
        coords /= box_length

    if save_txt:
        np.savetxt(f'{n_atoms}_positions.txt', coords, fmt=f'%.{decimal_digits}f')

    return coords

## Testing if the generate coordinates are all at least min_distance=2.0 from each other.
# coords = generate_coordinates(100, 12, scale_coords=False)
# d_ijk = to_mic(coords[:,np.newaxis,:] - coords[np.newaxis,:,:])
# dists = np.linalg.norm(d_ijk, axis=2)
# print(np.min(dists[dists != 0]))
# quit()

def create_poscar(header, atomic_system, coordinate_matrix=None, box_length=None, density=None, min_abs_distance=2.0,
                  decimal_digits=6, print_min_dist=True):
    """Generates a POSCAR file for VASP

    Args:
        header(str): Header of the poscar
        atomic_system (dict): A dictionary containing the atom type as key and the number of this type of atom as value
        coordinate_matrix (ndarray): (n_atoms,3) array containing the coordinates of the atoms.
        box_length (float): The size of the box that the simulation will take place
        density (float): The density of the system. This parameter is specified when we do not know the box_size
        min_abs_distance (float): Minimum absolute distance of any two atoms under PBC.
        decimal_digits (int): Number of decimal digits to represent the coordinates.
        print_min_dist (bool) => Prints the minimum distance between any two atoms assuming PBC.

    """
    if box_length is None and density is None:
        raise Exception('Please specify a box size or a density so that the box size can be estimated')
    elif box_length is None:
        density = density/1000 if density > 100 else density # convert to g/cm^3 if it is in kg/m^3
        box_length = box_dimension_estimator(atomic_system, density)

    header = header + ' (' + ''.join([k + str(v) for k,v in atomic_system.items()]) + ')'

    # Matrix containing the lattice vectors that defines the cubic simulation box
    cell = np.diag([box_length] * 3)

    # Extracting atom types and numbers
    atoms_types = atomic_system.keys()
    atoms_num = atomic_system.values()

    # Generating the initial random positions if we don't specify a coordinate matrix
    if coordinate_matrix is None:
        coordinate_matrix = generate_coordinates(n_atoms=sum(atoms_num), box_length=box_length, decimal_digits=decimal_digits,
                                                   min_distance=min_abs_distance)
        if print_min_dist:
            dists = np.linalg.norm(to_mic(box_length * (coordinate_matrix[:,np.newaxis,:] - coordinate_matrix[np.newaxis,:,:]),
                                          box_length),
                                   axis=2)
            print(np.min(dists[dists != 0], axis=None))
    else:
        coordinate_matrix = np.array(coordinate_matrix) / box_length

    with open(f'POSCAR.txt', 'w') as f:
        f.write(f'{header}\n')
        f.write('1.0\n')

        num_of_leading_digits = len(str(int(box_length)))
        for lattice_vector in cell:
            for el in lattice_vector:
                if el == 0:
                    f.write(' '*(num_of_leading_digits-1) + f'{el:.{decimal_digits}f}' + ' ')
                else:
                    f.write(f'{el:.{decimal_digits}f}' + ' ')
            f.write('\n')

        f.write(' ' +  ' '.join(atoms_types) + '\n')
        f.write(' ' + ' '.join(map(str,atoms_num)) + '\n')
        f.write('Direct\n')

        for coordinate_vector in coordinate_matrix:
            for coordinate in coordinate_vector:
                f.write(f'{coordinate:.{decimal_digits}f}' + ' ')
            f.write('\n')


def create_poscar_crystal(cell_type, lattice_parameter, atom_type, decimal_digits=6, header=None):
    if not header:
        header = f'{cell_type} {atom_type}'

    if cell_type == 'Cubic':
        pass

    elif cell_type == 'BCC':
        pass

    elif cell_type == 'FCC':
        scaling_factor = 1.0
        cell = np.eye(3) * lattice_parameter
        coords_matrix = np.array([[0.0, 0.0, 0.0],
                                  [0.5, 0.5, 0.0],
                                  [0.5, 0.0, 0.5],
                                  [0.0, 0.5, 0.5]])

    elif cell_type == 'HPC':
        a, ca_ratio = lattice_parameter
        c = a * ca_ratio
        scaling_factor = 1.0
        cell = np.array([[  a,      0,         0],
                         [-a/2, a*(3**0.5)/2,  0],
                         [  0,      0,         c]])

        coords_matrix = np.array([[0.0, 0.0, 0.0],
                                  [2/3, 1/3, 0.5]])


    with open(f'POSCAR.txt', 'w') as f:
        f.write(f'{header}\n')
        f.write(f'{scaling_factor}\n')

        num_of_leading_digits = 1
        for lattice_vector in cell:
            for el in lattice_vector:
                if el == 0:
                    f.write(' '*(num_of_leading_digits-1) + f'{el:.{decimal_digits}f}' + ' ')
                else:
                    f.write(f'{el:.{decimal_digits}f}' + ' ')
            f.write('\n')

        f.write(' ' +  atom_type + '\n')
        f.write(' ' + str(coords_matrix.shape[0]) + '\n')
        f.write('Direct\n')

        for positon_vector in coords_matrix:
            for coordinate in positon_vector:
                f.write(f'{coordinate:.{decimal_digits}f}' + ' ')
            f.write('\n')

# create_poscar_crystal('HPC', [2.95, 1.58745762], 'Ti')
# create_poscar_crystal('HPC', [3.05, 1.58745762], 'Ti')
# create_poscar_crystal('HPC', [2.85, ca], 'Ti', header=f'c/a = {ca}')


def create_poscar_crystal_supercell(size, cell_type, lattice_parameter, atom_type, decimal_digits=6, header=None):
    pass

def packmol_to_outcar(packmol_output, header, atomic_system, box_length=None, density=None):
    """Reads the packmol output (.pdb file) and creates a POSCAR based on the given atomic_system

    Args:
        packmol_output (str): Path to packmol output file
        header (str): Header of outcar file
        atomic_system (dict): dictionary containing the atom type as key and the number of this type of atom as value
        box_length (float): Box length
        density (float): Density of the system
    """
    coordinates_dict = {key: [] for key in atomic_system.keys()}
    coordinates_matrix = []
    n_atoms = sum(atomic_system.values())

    with open(packmol_output, 'r') as f:
        for i, line in enumerate(f):
            if 4 < i < (n_atoms + 5):
                row = line.split()
                row = [row[i] for i in [2, -6, -5 , -4]]
                row[0] = row[0].title()
                coordinates_dict[row[0]].append(list(map(float, row[1:])))

    for atom_type in coordinates_dict.keys():
        coordinates_matrix.extend(coordinates_dict[atom_type])

    create_poscar(header, atomic_system, coordinate_matrix=coordinates_matrix, box_length=box_length, density=density)

# packmol_to_outcar('C:/Users/Vitor/Downloads/Packmol/poscar_nonlinear_bicah2.txt', 'Non-Linear BiCaH2', {'Bi': 40, 'Ca': 10, 'H': 20}, density=7.9)
# packmol_to_outcar('C:/Users/Vitor/Downloads/Packmol/poscar_linear_cah2.txt', 'Linear CaH2', {'Ca': 25, 'H': 50}, box_length=10.0922)


#----------------------------------------------------------------------------------------------------------------------#
## Uranium impurity
# estimated density for 4:1:2 is (2.64*4 + 2.56*1 + 6.7*2) / 7 = 3.78
# create_poscar(header='FLiNaU', atoms={'F': 52, 'Li': 16, 'Na': 4 , 'U': 8}, density=3.78) # LiF-NaF-UF4 (4-1-2)
# create_poscar(header='FLiNaU', atoms={'F': 54, 'Li': 24, 'Na': 6 , 'U': 6}, density=3.3) # LiF-NaF-UF4 (4-1-1)
# F-  => 9
# Li+ => 4
# Na+ => 1
# U+4 => 1
# Sum: 15 atoms
# 15 * 6 => 90
# 15 * 5 => 75 ok

# create_poscar(header='NaClU', atoms={'Na': 30, 'Cl':50, 'U': 5}, density=2.6)
# create_poscar(header='NaClU', atoms={'Na': 36, 'Cl':48, 'U': 4}, box_size=14.5)
# create_poscar(header='LiKClU', atoms={'Li': 38, 'K': 26, 'Cl': 70, 'U': 2}, density=1.75)

#----------------------------------------------------------------------------------------------------------------------#
## FLiNa 58.2-14.7-27.1 mol% LiF-NaF-UF4

# 4-1-2 LiF-NaF-UF4
# F -> (5 + 8) * x   (5 from the salt and 8 from UF4)
# Li -> 4 * x
# Na -> 1 * x
# U -> 2 * x
# x = 5

## Doing 4-1-2 (57.1-14.3-28.6)
# create_poscar(header='FLiNa', atomic_system={'F': 50 , 'Li': 40 , 'Na': 10}, density=2, min_abs_distance=1.8)
# create_poscar(header='FLiNaTh', atomic_system={'F': 65 , 'Li': 20 , 'Na': 5, 'Th': 10}, density=4.211, min_abs_distance=2)
# create_poscar(header='FLiNaU', atomic_system={'F': 65 , 'Li': 20 , 'Na': 5, 'U': 10}, density=4.211, min_abs_distance=2)

# create_poscar(header='FLiNa+C', atomic_system={'F': 50 , 'Li': 40 , 'Na': 10, 'C': 4}, density=2, min_abs_distance=1.8)
# create_poscar(header='FLiNa+Cs', atomic_system={'F': 50 , 'Li': 40 , 'Na': 10, 'Cs': 4}, density=2, min_abs_distance=1.8)
# create_poscar(header='FLiNa+Cr', atomic_system={'F': 50 , 'Li': 40 , 'Na': 10, 'Cr': 4}, density=2, min_abs_distance=1.8)

#----------------------------------------------------------------------------------------------------------------------#
# Ms (https://en.wikipedia.org/wiki/Molten_salt_reactor)
# LiF-NaF (eutectic) in a 3-2 proportion (60-40 mol%) has a 656 Celsius (929 K) melting point.

# F -> (10 + 4) * x   (10 from the salt and 4 from UF4)
# Li -> 6 * x
# Na -> 4 * x
# U -> 1 * x
# Using x = 4 for a total of 104 atoms

flina_calc = lambda T: 2.4689 - 0.000425 * T
flina_exp = lambda T: 2.5702325 - 0.00055 * T

flinath_imix = lambda T: 3.00153223 - 0.00057462 * T # Same as 0.909 * flina(T) + 0.091 * thf4(T)
flinath_calc = lambda T: 3.6437 - 5.686e-4 * T
# n_Th = 4
# print(box_dimension_estimator({'F': 40+4*n_Th , 'Li':24 , 'Na': 16, 'Th':n_Th}, flinath_calc(1500)))
# quit()
flinau_calc = lambda T: 3.7054 - 6.286e-4 * T

## Doing 6-4-1 (54.5-36.4-9.1)
# create_poscar(header='FLiNa', atomic_system={'F': 50 , 'Li': 30 , 'Na': 20}, density=flina_calc(1473), min_abs_distance=1.8)
# create_poscar(header='FLiNaTh', atomic_system={'F': 56 , 'Li':24 , 'Na': 16, 'Th':4}, density=flinath_calc(1473), min_abs_distance=1.8)
create_poscar(header='FLiNaU', atomic_system={'F': 56 , 'Li':24 , 'Na': 16, 'U':4}, density=flinau_calc(1473), min_abs_distance=1.8)


# create_poscar(header='Al', atomic_system={'Al': 100}, density=2.3, min_abs_distance=1.8)
#-----------------------
# LANL
# create_poscar(header='lanl', atomic_system={'F': 25, 'Li': 12, 'Na': 8, 'K': 9, 'Cl': 24, 'Mg': 10, 'Be': 12},
#               box_length=13, min_abs_distance=1.5)

# FLiNaK (46.5-11.5-42 mol % of LiF-NaF-KF), ~720K melting point
# 46.5-11.5-42 -> 46-12-42 -> 23-6-21
# F  -> 50 * x
# Li -> 23 * x
# Na ->  6 * x
# K  -> 21 * x


# rho_eutectic_flinak_exp = lambda T: 2.5793 - 0.624e-3*T # [K]
# rho_eutectic_flinak_sim = lambda T: 2.41 - 0.44e-3*T # [K]
#
# create_poscar(header='flinak', atomic_system={'F': 50, 'Li': 23, 'Na': 6, 'K': 21}, density=rho_eutectic_flinak_sim(1300),
#               min_abs_distance=1.7)


#-----------------------
# Zn-Cl2

# ZnCl2
# sim_zncl2 = lambda T: 2.45167 - 0.0001 * T  # 650-850k
# create_poscar(header='ZnCl2', atomic_system={'Zn': 33, 'Cl': 66}, density=sim_zncl2(850), min_abs_distance=1.9)

# MgCl2
# sim_mgcl2 = lambda T: 1.916 - 0.0002 * T # 1100-1300k
# create_poscar(header='MgCl2', atomic_system={'Mg': 33, 'Cl': 66}, density=sim_mgcl2(1300), min_abs_distance=1.7)

# LiF-BeF2 66-33 mol%
# sim_flibe = lambda T: 2.29 - 0.0003 * T # 800-1000k
# create_poscar(header='FLiBe', atomic_system={'F': 64, 'Li': 32, 'Be':16}, density=2.8, min_abs_distance=1.7)

# LiF-NaF-ZrF4
# sim_flinazr = lambda T: 2.903 - 0.0005 * T # 1000-1200k
# create_poscar(header='FLiNaZr', atomic_system={'F': 56 , 'Li':24 , 'Na': 16, 'Zr':4}, density=sim_flinazr(1100), min_abs_distance=1.8)