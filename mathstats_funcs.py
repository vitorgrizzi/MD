import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline

def pad_zero(num, n_integers):
    """ Represents the number using 'n_integers'.
    """
    return '0' * (n_integers - len(str(num))) + str(num)

def to_sci10(n, decimal_digits=2):
    """
        Converts the number to a scientific representation using power of 10. For instance, if 0.00543 is given, it
        returns (5.43, -3) where the first is the number to the left of the power of 10 rounded to the desired decimal
        digits and the second is the power of 10 (5.43 x 10^-3).

        OBS: This will be mostly used to print the number or use it as label in plots
    """
    num, power = f'{n:.{decimal_digits}e}'.split('e')

    return float(num), int(power)


def moving_average(x, window):
    """Calculates the moving average of an array over its last axis/dimension.

    Args:
        x (ndarray): Array that we want to calculate the moving average
        window (int): Size of the moving average, or how many elements will be "lumped" together into a single
                        representative number.

    Returns:
        (ndarray) containing the moving average. Note that the size of this array will depend on the window that
        we chose. For instance, for a window 'k' and a 1-D array of size 'N', the final size will be 'N-k'

    """
    x_csum = np.insert(np.cumsum(x, axis=-1), 0, 0, axis=-1)

    return (x_csum[..., window:] - x_csum[..., :-window]) / window


def reescale_array(x, new_limits=(0,1)):
    """
        Changes the range of an array 'x' preserving the relative distance between each element. For instance, if
        x=[-10,50] we can reescale it to x=[10,15]

        x (array) => The array containing the data that we wish to transform
        new_limits (tuple) => A tuple containing the new range that we wish for 'x' to be, where new_limits[0] is the
                              lower and new_limits[1] is the upper.

    """
    # 1) The first term (x - x.min()) changes the lower limit to 0 x=[0, x.max()-x.min()];
    # 2) The second term (new_limits[1] - new_limits[0]) / (x.max() - x.min()) adjusts the range of the data to match
    #    the new desired range;
    # 3) The addition term new_limits[0] shifts the data (which now has the lower limit of 0 and the correct range) to
    #    the desired lower and upper limits.
    return (x - x.min()) * ((new_limits[1] - new_limits[0]) / (x.max() - x.min())) + new_limits[0]


def get_statistics(property_array, n_last_frames=0, decimals=2):
    """
        Returns the average and standard deviation of 'property_array' taking into account only the last 'n_last_frames'
        elements over axis=1.
    """
    if n_last_frames == 0:
        n_last_frames = property_array.shape[1]
    property_avg = np.round(np.mean(property_array[:, -n_last_frames:], axis=1), decimals=decimals)
    property_std = np.round(np.std(property_array[:, -n_last_frames:], axis=1), decimals=decimals)

    return property_avg, property_std


def closest_to_n(x, n):
    """Returns the element inside the array 'x' that has the closest value to 'n' and its corresponding index.
    """
    idx_closest = np.abs((x - n)).argmin()
    return idx_closest, x[idx_closest]


def find_critical_points(x, mode, noise_cancel=True, noise_cancel_order=2):
    """
        Find all the local maxima or minima of the array 'x'. The parameter noise_cancel is used to filter data noise.

        x [array] => array that we wish to find the local maxima/minima points
        noise_cancel [bool] => Controls wheather we want to turn on the noise filter or not.
        noise_cancel_order [int] => Given a candidate local maxima point this parameter controls the minimum number of
                                    consecutive increasing points needed to consider the point is a real critical point
                                    instead of noise.

        Returns:
            max_idx [list] => Returns the indexes of the local maxima/minima of array 'x'
            x[max_idx] [array] => Returns the local maxima/minima of array 'x'.
    """
    x = np.array(x)
    critical_idx = []
    slope_sign = np.sign(np.diff(x))
    previous_slope = slope_sign[0]
    critical_slope = 1 if mode == 'min' else -1

    # We need to do that for slope_sign[i+1] or [i+2] not to crash the function
    N = np.newaxis
    if noise_cancel_order == 1:
        N = -1
    elif noise_cancel_order == 2:
        N = -2

    for i, slope in enumerate(slope_sign[:N]):
        if slope == critical_slope and previous_slope != slope:
            if noise_cancel: # rewrite this code to consider a k-order noise (easy)
                if noise_cancel_order == 1:
                    if slope_sign[i] == slope_sign[i+1]:
                        critical_idx.append(i)
                elif noise_cancel_order == 2:
                    if slope_sign[i] == slope_sign[i+1] == slope_sign[i+2]:
                        critical_idx.append(i)
            else:
                critical_idx.append(i)
        previous_slope = slope

    return np.array(critical_idx), x[critical_idx]


def find_local_mins(x, noise_cancel=True, noise_cancel_order=2):
    """
        Find all the local minima of the array 'x'. The parameter noise_cancel is used to filter data noise.

        x [array] => array that we wish to find the local minima points
        noise_cancel [bool] => Controls wheather we want to turn on the noise filter or not.
        noise_cancel_order [int] => Given a candidate local minima point this parameter controls the minimum number of
                                    consecutive increasing points needed to consider the point is a real local minima
                                    instead of noise.

        Returns:
            min_idx [list] => Returns the indexes of the local minimas of array 'x'
            x[min_idx] [array] => Returns the local minimas of array 'x'.

    """

    return find_critical_points(x, mode='min', noise_cancel=noise_cancel, noise_cancel_order=noise_cancel_order)


def find_local_maxs(x, noise_cancel=True, noise_cancel_order=2):
    """
        Find all the local maxima of the array 'x'. The parameter noise_cancel is used to filter data noise.

        x [array] => array that we wish to find the local maxima points
        noise_cancel [bool] => Controls wheather we want to turn on the noise filter or not.
        noise_cancel_order [int] => Given a candidate local maxima point this parameter controls the minimum number of
                                    consecutive decreasing points needed to consider the point is a real local maxima
                                    instead of noise.

        Returns:
            max_idx [list] => Returns the indexes of the local maxima of array 'x'
            x[max_idx] [array] => Returns the local maxima of array 'x'.
    """

    return find_critical_points(x, mode='max', noise_cancel=noise_cancel, noise_cancel_order=noise_cancel_order)


def cum_mean(x, axis=None):
    """
        Returns the cumulative mean of the array 'x'
    """
    return np.cumsum(x, axis=axis) / np.arange(1, x.size+1)


def sequential_sample(x, sample_size, rand_n=None):
    """Returns a random sequential sample of the array 'x' with 'sample_size' elements.

    Args:
        x (np.array) => Array that we want to bootstrap
        sample_size (int) => Size of the samples
        rand_n (int) => Initial position where the boostraped sample will start.

    Return:
        np.array of size (sample_size,) containing the randomly chosen sequence.
    """

    if rand_n is None:
        max_idx = x.size - sample_size
        rand_n = np.random.randint(0, max_idx)

    return x[rand_n: rand_n+sample_size]

def correlation(x1, x2, n_percentile=5):
    """Calculates the correlation along the last axis of a 2-D or 1-D array 'x'.

       OBS1: As a rule of thumb, we should use the smallest possible `n_perc` that can still capture the correlation
             function decaying and oscillating around its zero (more technically, around its mean). This ensures better
             statistics and thus more reliable results.
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


def autocorrelation(x, n_percentile=50, center_array=True, do_fft=True):
    """Calculates the autocorrelation along the last axis of a 2-D or 1-D array 'x'.

        Note that this is technically the autocovariance because we have units of 'x' squared. Autocorrelation is the
        autocovariance divided by the variance, which eliminates the units and limit the autocorrelation value to
        [-1,1]. However, in most fields autocovariance and autocorrelation are used interchangeably.

    Args:
        x (ndarray): The array that we wish to calculate the autocorrelation
        n_percentile (int): How much of the array 'x' we wish to take into account in the autocorrelation or at
                            which point we stop "sliding" the array 'x' over itself. For n_percentile=75, it
                            will stop when the first element of the sliding array x[:N-shift] reach the element
                            corresponding to the 75th percentile of the "static" array x[shift:].
        center_array (Bool): Controls if we want to center the array around its mean or not. Remember that the
                             autocorrelation always decays to the mean, so if we center the array around its
                             mean the autocorrelation will decay and then oscillate around 0.
        do_fft (Bool): Controls whether we take the autocorrelation in real or Fourier domain. It will be always
                       faster to do it in Fourier domain for our use cases, so just leave it as True. The real
                       space approach is only faster for very small arrays (N < 15)

    Returns:
        (ndarray): A 1-D or 2-D np.array whose last axis is of size int(x.shape[-1] * n_percentile/100) and stores
                   the autocorrelation. For a (N,M) 2-D array, the autocorrelation of each row is calculated and
                   stored along the axis=1, so a (N, int(M*n_percentile/100)) shaped array is returned.

    OBS1: If we use n_percentile=100, the whole array will be covered but the last autocorrelation will be simply
          x[-1] * x[0] which won't be representative. The acf will thus have poor statistics on its last elements.
          As we cover more of the array 'x', the acf will be more and more noisy due to the lack of data to average.
    OBS2: If we consider the full correlation as specified by np.correlate(mode='full'), we are getting the
          [50, (100+n_percentile)/2] range of np.correlate. For n_percentile=75, we get 50-87.5% of the np.correlate
          array because when n_percentile=100 we get 50-100% of the np.correlate.
    OBS3: This function is very similar to np.correlate, the only difference is that here we divide the dot product
          by the number of elements.
    OBS4: Note that when we do the fft we take the multiplication of x_fft with its conjugate because we are doing
          autocorrelation. Another option would be to multiply x_fft by fft(x[::-1]), i.e. the fft of its reverse.
          If we were doing convolution, then we would simply multiply x_fft by itself (no conjugate).
    OBS5: We have to add the zero padding otherwise it won't work. Apparently FFT is a circular cross-correlation
          so we need to add the zeros to simulate the signal that goes to zero in infinity. I think we must add
          the zeros to show that de signal decay to zero because for the Fourier transform to exist the function

    OBS6: For some reason if the pad of zeros has the same size as the array the FFT operation will be much faster.
          For instance, with an array of size N=10k I get 378 it/s with a 'N-1' zero padding and 2900 it/s with a
          'N' zero padding. For a N=100k array its 33 it/s and 133 it/s respectively.

    OBS7: As a rule of thumb, we should use the smallest possible `n_perc` that can still capture the correlation
          function decaying and oscillating around its zero (more technically, around its mean). This ensures better
          statistics and thus more reliable results.

    """
    assert x.ndim in [1,2], f"The dimension of the array has to be either 1 or 2, but this array has dimension {x.ndim}."

    N = x.shape[-1]
    acf_size = int(N * n_percentile/100) # this defines the maximum time delay tau_max.

    if do_fft:
        if center_array:
            # I did a transpose here to make advantage of numpy broadcasting rules. This works because transposing a
            # 1-D array don't do anything and for 1-D arrays taking the mean along axis={0,-1,None} is the same. The
            # other alternative was x_centered = x - x.mean() if x.ndim == 1 else x - x.mean(axis=1).reshape(-1,1).
            x_centered = np.transpose(x.T - x.mean(axis=-1))
            x_padded = np.hstack((x_centered, np.zeros_like(x)))
        else:
            x_padded = np.hstack((x, np.zeros_like(x)))
        x_fft = fft(x_padded)
        # We use .real because the acf is the inverse transform of the norm of this complex vector 'x_fft', so the
        # imaginary part is just a residue
        acf = ifft(x_fft * x_fft.conjugate()).real[..., :acf_size]

    # The code below for acf in real space only works for 1-D arrays. This is just for testing purposes.
    else: # Better to divide the whole array by np.arange() at the end instead of 'N-shift' at each iteration
        acf = np.zeros(acf_size)
        mean = x.mean()
        for shift in range(acf_size): # shift is the time delay 'tau'
            # x[:N-shift] is sliding over x[shift:]
            if center_array:
                acf[shift] = np.dot(x[:N-shift] - mean, x[shift:] - mean) # / (N - shift)
            else:
                acf[shift] = np.dot(x[:N-shift], x[shift:]) # / (N - shift)

    return acf / np.arange(N, N-acf_size, -1)


# np.random.seed(103)
# a = np.random.normal(size=10)
# a_fft = fft(a)
# print(ifft(a_fft * a_fft.conjugate()).real)
# print(np.correlate(a, a, mode='full')[len(a)//2 + 4:])
# print(acf(a, n_percentile=100, do_fft=True, center_array=False))
# from tqdm import tqdm
# for _ in tqdm(range(100000)):
#     acf(a, n_percentile=75, do_fft=False)
# quit()

# np.random.seed(101)
# a = np.random.normal(size=10)
# b = np.random.normal(size=10)
# c = np.random.normal(size=10)
# x = np.vstack((a,b,c))
# x_acf = np.vstack((acf(a, do_fft=False), acf(b, do_fft=False), acf(c, do_fft=False)))
# # x_acf = np.vstack((acf(a), acf(b), acf(c)))
# print(np.allclose(acf(x, do_fft=True), x_acf)) # True, so doing the acf for the matrix is the same as for the vector
# quit()



def bin_values(x, n_bins, limit=()):
    """Bins the elements of the given array.

    Args:
        x (np.array): array that we want to bin
        n_bins (int): Number of bins
        limit (tuple): Tuple containing the inferior and superior limit [inf_lim, sup_lim]

    Return:
        (np.array): shape (n_bins,) array containing the count on each bin
        (np.array): shape (n_bins+1,) array representing the bins where the difference between two subsequent elements
                    is the bin size

    OBS1: I think we can do that using matrix multiplication. One array contains the values to bin and the RHS array
          has only 0s and 1s and each column/row represents a different bin.

          *** We could do that by A - B where 'B' contains a range. If in abs(A-B) an element is within a range [x1,x2]
              then the number in 'A' falls in the bin being represented by that row or column of B.
    """
    if not limit:
        limit = (x.min(), x.max()) # [inf_lim, sup_lim]

    inf_lim, sup_lim = limit
    bin_range = np.linspace(inf_lim, sup_lim, n_bins + 1)
    bin_count = np.zeros(n_bins)

    # Method 1: 60 it/s for a (1000000,) array; 12404 it/s for (1000,) array
    for i in range(n_bins):
        bin_count[i] = np.sum((x > bin_range[i]) & (x < bin_range[i+1]))

    # Method 2: 56 it/s for a (1000000,) array; 14930 it/s for (1000,) array
    # for i in range(n_bins):
    #     bin_count[i] = np.flatnonzero((x > bin_range[i]) & (x < bin_range[i+1]))[0]


    return bin_count, bin_range

# a = np.random.random(1000)
# a = np.random.random(1000)
#
# from tqdm import tqdm
# for _ in tqdm(range(100000)):
#     bin_values(a, 10, (0,1))
    # np.histogram(a, 10)

# Testing ACF a delay 0 and very high delay
# rng = np.random.RandomState()
# rng.seed(101)
# n = 25000
# r = rng.rand(n) + 2
# y = acf(r, 60)
# x = np.arange(1, y.shape[0]+1)
#
# fig, ax = plt.subplots()
# ax.plot([x.min(), x.max()], [r.var() + r.mean()**2, r.var() + r.mean()**2], color='green', ls='-.', label='var')
# ax.plot([x.min(), x.max()], [r.mean(), r.mean()], color='blue', ls='-.', label='mean')
# ax.plot([x.min(), x.max()], [r.mean()**2, r.mean()**2], color='red', ls='-.', label='mean squared')
# ax.plot(y, color='black', label='acf')
# ax.legend()
# plt.show()


# p = lambda x : 1 / (abs(x)**2.2 + 1)
# x = np.linspace(-20, 20, 100)
#
# fig, ax = plt.subplots()
# ax.plot(x, p(x))
# plt.show()

