import numpy as np
from physchem_funcs import *
from MD import box_dimension_estimator, generate_coordinates, construct_distance_table
from ase.data import atomic_masses, atomic_numbers


def create_lammps_data_file(atomic_system, style='full', charges=None, min_distance=2.0, box_length=None, density=None,
                             decimals=6, double_box=False, axis_to_double='z', filename='lammps_data'):
    """Generates a data file for LAMMPS

    Args:
        atomic_system (dict): A dictionary containing the atom type as key and the number of this type of atom as value.
        style (str): Style of lammps data file, can be either "full", "atomic", or "charge".
        charges (dict): A dictionary containing atom type as keys and its charge as values.
        min_distance (float): Minimum distance between any two atoms under pbc.
        box_length (float): The length of the box that the simulation will take place.
        density (float): The density of the system. This parameter is specified when we do not know the box_size.
        decimals (int): Decimal places to write the atomic coordinates.

    """

    if box_length is None and density is None:
        raise Exception('Please specify a box size or a density so the box size can be estimated')
    elif box_length is None:
        density = density/1000 if density > 100 else density # convert to g/cm^3 if it is in kg/m^3
        box_length = box_dimension_estimator(atomic_system, density)

    atom_types, atom_nums = zip(*atomic_system.items())
    n_atoms = sum(atom_nums)
    lammps_type = range(1, len(atom_types)+1)
    type_array = np.hstack([[t] * n for t, n in zip(lammps_type, atom_nums)])
    charge_array = np.zeros(n_atoms) if charges is None else np.hstack([[charges[at]] * atomic_system[at] for at in atom_types])
    box_vec = np.ones(3) * box_length

    if double_box: # Used if we want to create a rectangular box
        target_idx = ['x', 'y', 'z'].index(axis_to_double)
        coord1 = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals, min_distance=min_distance)
        coord2 = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals, min_distance=min_distance)
        coord2[:,target_idx] += box_length
        positions = np.vstack((coord1, coord2))
        n_atoms *= 2
        type_array = np.hstack((type_array, type_array))
        charge_array = np.hstack((charge_array, charge_array))
        box_vec[target_idx] *= 2

    else:
        positions = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals,
                                                min_distance=min_distance)

    coordinate_matrix = np.hstack((np.arange(1, n_atoms+1).reshape(-1,1),
                                   np.zeros((n_atoms,1), dtype=int),
                                   type_array.reshape(-1,1),
                                   charge_array.reshape(-1,1),
                                   positions
                                   ))

    with open(f'{filename}.txt', 'w') as f:
        f.write('Start File for LAMMPS' + 2*'\n')

        f.write(f'{n_atoms} atoms' + '\n')
        f.write(f'{len(atom_types)} atom types' + 2*'\n')

        f.write(f'0.00000000 {box_vec[0]} xlo xhi' + '\n')
        f.write(f'0.00000000 {box_vec[1]} ylo yhi' + '\n')
        f.write(f'0.00000000 {box_vec[2]} zlo zhi' + 2*'\n')

        f.write(f'Masses' + 2*'\n')

        for i, atom in enumerate(atom_types):
            f.write(f'{i+1} {atomic_masses[atomic_numbers[atom]]}\n')
        f.write('\n' + 'Atoms'  + 2*'\n')

        for c in coordinate_matrix:
            if style == 'atomic':
                f.write(f'{int(c[0])} {int(c[2])} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')
            elif style == 'charge':
                f.write(f'{int(c[0])} {int(c[2])} {c[3]:.1f} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')
            elif style == 'full':
                f.write(f'{int(c[0])} {int(c[1])} {int(c[2])} {c[3]:.1f} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')


# create_lammps_data_file(atomic_system={'F': 52, 'Li': 16, 'Na': 4 , 'U': 8}, charges={'F': -1, 'Li': 1, 'Na': 1 , 'U': 4},
#                         density=3.78, double_box=True, axis_to_double='z', )
# quit()
# FLiNaK (46.5-11.5-42 mol % of LiF-NaF-KF), ~720K melting point
rho_eutectic_flinak = lambda T: 2.5793 - 0.624e-3*T # [K]
# Total atoms = 10.000
# LiF => 4650
# NaF => 1150
# KF  => 4200
# F: 5000, Li: 2325, Na: 575, K: 2100

# create_lammps_data_file(atomic_system={'F': 5000, 'Li': 2325, 'Na': 575 , 'K': 2100}, density=rho_eutectic_flinak(1000), style='atomic')
flinath_calc = lambda T: 3.6437 - 5.686e-4 * T

# n_Th = 40
# create_lammps_data_file(atomic_system={'F': 400 + 4*n_Th , 'Li': 240 , 'Na': 160, 'Th': n_Th},
#                         density=flinath_calc(1300), style='atomic', filename='data_1k')

# n_Th = 4000
# create_lammps_data_file(atomic_system={'F': 40000 + 4*n_Th , 'Li': 24000 , 'Na': 16000, 'Th': n_Th},
#                         density=flinath_calc(1273), style='atomic', filename='data_100k')

#
# n_Th = 200
# create_lammps_data_file(atomic_system={'F': 2000 + 4*n_Th , 'Li': 1200 , 'Na': 800, 'Th': n_Th}, density=flinath_calc(1200),
#                         double_box=True, axis_to_double='z', style='atomic', filename='data_2z')

#
flina_calc = lambda T: 2.4689 - 0.000425 * T
# create_lammps_data_file(atomic_system={'F': 2500, 'Li': 1500 , 'Na': 1000}, density=flina_calc(1200),
#                         double_box=True, axis_to_double='z', style='atomic', filename='data_pure10k')
