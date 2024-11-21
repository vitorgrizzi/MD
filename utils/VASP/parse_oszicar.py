import numpy as np

# FALTA TESTAR!!!!!!!!!!!!!!! mas acho q ta certo hehe
def extract_property(oszicar_path, create_file=False, new_file_path=None):
    """
        Extract the line of the OSZICAR file that contains the temperature and the energies. The order of each quantity
        in the returned 'data' matrix is [T, E, F, E0, EK, SP, SK]
    """
    data = []

    with open(oszicar_path, 'r') as oszicar:
        for line in oszicar:
            if 'T=' in line:
                data.append(' '.join(line.split()[2::2]))
                # dei um split pra pegar os elementos q eu quero com o slicing e dps juntei dnv p poder usar o
                # writelines q recebe uma lista de strings

    if create_file:
        with open(f'{new_file_path}.txt', 'w') as f:
            f.writelines('\n'.join(data))
        return None
    else:
        return [list(map(float, row.split())) for row in data]


def get_values(property_file_name):
    """
        Read the values from the text file generated from the POSCAR file that contains the desired properties.
    """
    property_value = []
    with open(property_file_name, 'r') as f:
        for line in f:
            property_value.append(float(line))

    return np.array(property_value)
