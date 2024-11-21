import numpy as np


# Posso usar o skiprows e maxrows p fazer isso.
# Leio uma vez o arquivo e vejo as linhas em que o T_array comeca e termina;
# faco os mesmo com os dados de gs_r;
# reabro o arquivo usando skiprows e maxrows pra so ler as linhas que tem T_array e depois que tem gs_r
def extract_van_hove(gs_path, solid_angle_correction=True):
    """
        Extracts the relevant information from LiquidLib calculated self/collective Van Hove correlations

        Returns:
        time_array => A (n,) array containing each time point where G(r,t) was calculated, where 'n' depends on both
                      'number_of_time_points' and 'time_interval' flags
        gs_rt => A (number_of_bins, n+1) matrix containing G(r,t). The first column corresponds to the distance 'r'
                 from the origin, and the other 'n' columns are the G(r,t) at sucessive time points of time_array at
                 that distance. For instance, G[:,3] corresponds to G(r,t) at t=time_array[2] at all distances.

        OBS: We have to multiply the returned gs_rt by 4πr² because our box is 3-D, liquids are isotropic, and we want
             to know the probability of finding a particle inside a spherical shell of volume V = 4π(r+dr)² - 4πr²
             given that at t=0 it was at r=0. This correction is set by solid_angle_correction parameter.
    """
    with open(gs_path, 'r') as gs_rt:
        time_array = []
        for line in gs_rt:
            try:
               time_array.append(float(line))
            except ValueError:
                continue

    time_array = np.array(time_array)
    print(time_array, len(time_array))
    gs_rt = np.loadtxt(gs_path, skiprows=len(time_array)+3)

    if solid_angle_correction:
        gs_rt[:,1:] = (4 * np.pi * gs_rt[:,0:1]**2) * gs_rt[:,1:] # gs_rt[:,0:1] instead of gs_rt[:,0] to broadcast

    return time_array, gs_rt

# common_path = 'C:/Users/Vitor/Desktop/Simulations/FLiNa/FLiNa_ThF4'
# t, gs = extract_van_hove(f'{common_path}/873k/eq/self_van_hove_correlation')
# print(t.shape, gs.shape)


def extract_intermediate_scattering(f_path):
    """
        Extracts the relevant information from LiquidLib calculated self/collective intermediate scattering function.

        Returns:
        k_array => A (n,) array containing the magnitude of all k-vectors used in the calculation.
        f_kt => A (number_of_bins, n+1) matrix representing F(k,t). The first column f_kt[:,0] corresponds to the time
                points 't', while the other 'n' columns are the F(k,t) for the different |k|'s in 'k_array'. For
                instance, f_kt(:,1) contains F(k_array[0],t), or the F(k,t) of the lowest k-vector.
    """
    k_array = []
    with open(f_path, 'r') as f_kt:
        for line in f_kt:
            if 'F_s(k,t)' in line:
                break
            else:
                try:
                    k_array.append(float(line))
                except ValueError:
                    continue

    k_array = np.array(k_array)
    f_kt = np.loadtxt(f_path, skiprows=len(k_array)+3)

    return k_array, f_kt

