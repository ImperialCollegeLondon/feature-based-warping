import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, convolve
from scipy.spatial import Voronoi
from tqdm import tqdm
import multiprocessing as mp
from numba import cuda, jit, float32
import gc
import config

@jit(nopython=True)
def upsample_signal(signal, factor):
    """Upsample the signal by the given factor using interpolation."""
    n = len(signal)
    upsampled_signal = np.interp(np.linspace(0, n, n * factor), np.arange(n), signal)
    return upsampled_signal

import numpy as np
from numba import jit

@jit(nopython=True)
def argrelextrema_numba(data, comparator, order=7):
    """
    Numba-compatible version of `argrelextrema` to detect relative extrema.

    Parameters:
    - data (np.ndarray): Input 1D array to analyze.
    - comparator (callable): A function that compares two values, e.g., np.greater or np.less.
    - order (int): How many points on each side to use for comparison.

    Returns:
    - extrema_indices (list): Indices of the detected relative extrema as a list.
    """
    extrema_indices = []  # Collect indices as a Python list
    n = len(data)

    for i in range(order, n - order):
        is_extreme = True
        for j in range(1, order + 1):
            if not comparator(data[i], data[i - j]) or not comparator(data[i], data[i + j]):
                is_extreme = False
                break
        if is_extreme:
            extrema_indices.append(i)

    return extrema_indices


@jit(nopython=True)
def find_peaks_and_troughs(data, order=7):
    """
    Detect peaks and troughs in a 1D signal using a Numba-compatible implementation.

    Parameters:
    - data (np.ndarray): Input 1D array to analyze.
    - order (int): How many points on each side to use for comparison.

    Returns:
    - peaks (list): Indices of detected peaks as a Python list.
    - troughs (list): Indices of detected troughs as a Python list.
    """
    def greater(a, b):
        return a > b

    def less(a, b):
        return a < b

    peaks = argrelextrema_numba(data, greater, order)
    troughs = argrelextrema_numba(data, less, order)

    return np.array(peaks), np.array(troughs)

@jit(nopython=True)
def match_features(base_features, mon_features):
    """Match pairs of features based on proximity."""
    matched_pairs = []
    used_mon_features = set()
    for base_feature in base_features:
        if len(mon_features) > 0:  # Check if there are features to match
            closest_feature = mon_features[0]
            min_distance = abs(closest_feature - base_feature)
            for mf in mon_features:
                distance = abs(mf - base_feature)
                if distance < min_distance:
                    closest_feature = mf
                    min_distance = distance
            matched_pairs.append((base_feature, closest_feature))
            used_mon_features.add(closest_feature)
            mon_features = np.array([mf for mf in mon_features if mf != closest_feature])
    return matched_pairs

@jit(nopython=True)
def sort_features(paired_peaks, paired_troughs):
    """Sort paired features by their positions in the base signal."""
    paired_peaks.sort()
    paired_troughs.sort()
    return paired_peaks, paired_troughs

@jit(nopython=True)
def bresenham_line(x0, y0, x1, y1):
    """Generate points on a line from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))  # Use a list to accumulate points
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points  # Return the list of points

@jit(nopython=True)
def construct_feature_based_cost_matrix(base1d, mon1d, waypoints):
    """Construct a cost matrix based on feature alignment with straight valleys between waypoints."""
    n, m = len(base1d), len(mon1d)
    high_cost = 1000.0  # Initial high cost for the matrix
    cost_matrix = np.full((n + 1, m + 1), high_cost)

    waypoints = np.array(waypoints)
    waypoints = waypoints[waypoints[:, 0].argsort()]  # Ensure waypoints are in order

    max_width = 20
    for k in range(len(waypoints) - 1):
        start_wp = waypoints[k]
        end_wp = waypoints[k + 1]
        points = bresenham_line(start_wp[0], start_wp[1], end_wp[0], end_wp[1])
        
        # Add a linear cost gradient around the zero-cost line
        for (x, y) in points:
            for i in range(-max_width, max_width + 1):
                for j in range(-max_width, max_width + 1):
                    if 0 <= x + i < n and 0 <= y + j < m:
                        distance = abs(i) + abs(j)  # Manhattan distance
                        cost = (distance / max_width) * high_cost
                        cost_matrix[x + i, y + j] = min(cost_matrix[x + i, y + j], cost)

    return cost_matrix

@jit(nopython=True)
def fbw_with_feature_based_cost(base1d, mon1d, waypoints):
    """Perform DTW with waypoints using a feature-based cost matrix."""
    cost_matrix = construct_feature_based_cost_matrix(base1d, mon1d, waypoints)
    n, m = cost_matrix.shape[0] - 1, cost_matrix.shape[1] - 1

    # Initialize DTW matrix
    fbw_matrix = np.full((n + 1, m + 1), np.inf)
    fbw_matrix[0, 0] = 0

    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i, j]
            fbw_matrix[i, j] = cost + min(fbw_matrix[i - 1, j],    # Insertion
                                          fbw_matrix[i, j - 1],    # Deletion
                                          fbw_matrix[i - 1, j - 1])  # Match

    # Backtrack to find the path and shifts
    shifts = np.zeros(m)
    path = []
    i, j = waypoints[-1]  # Start backtracking from the last waypoint
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        min_val = min(fbw_matrix[i - 1, j] if i > 0 else np.inf,
                      fbw_matrix[i, j - 1] if j > 0 else np.inf,
                      fbw_matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf)
        if min_val == fbw_matrix[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif min_val == fbw_matrix[i, j - 1]:
            j -= 1
        else:
            i -= 1

    path.reverse()
    for k, (i, j) in enumerate(path):
        shifts[j] = j - i

    return shifts, cost_matrix, path

@jit(nopython=True)
def smooth_shifts(shifts, window_size=5):
    """Smooth shifts using a moving average filter with edge padding."""
    window = np.ones(window_size) / window_size
    padded_shifts = custom_pad(shifts, window_size // 2, shifts[0])
    padded_shifts = custom_pad(padded_shifts, window_size // 2, shifts[-1])
    smoothed_shifts = np.convolve(padded_shifts, window, mode='valid')
    return smoothed_shifts

@jit(nopython=True)
def custom_diff(arr):
    """Compute the first-order difference of an array with prepending."""
    result = np.empty(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] - arr[i - 1]
    return result

@jit(nopython=True)
def custom_pad(arr, pad_size, pad_value):
    """Pad an array with a specific value."""
    padded_arr = np.empty(len(arr) + pad_size)
    padded_arr[:len(arr)] = arr
    padded_arr[len(arr):] = pad_value
    return padded_arr

@jit(nopython=True)
def downsample_shifts(shifts, factor):
    """Downsample the shifts by the given factor using averaging."""
    n = len(shifts) // factor
    downsampled_shifts = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(factor):
            sum_val += shifts[i * factor + j]
        downsampled_shifts[i] = sum_val / factor
    return downsampled_shifts

@jit(nopython=True)
def check_strain_limit(smoothed_shifts, strain_limit):
    """Check if the strain limit is satisfied."""
    time_strain = custom_diff(smoothed_shifts)
    return np.all(time_strain <= strain_limit)

def fbw_with_strain_limit(base_signal, mon_signal, strain_limit, upsample_factor, window_size, plot_results=False):
    """Perform FBW to satisfy the strain limit."""
    upsampled_base = upsample_signal(base_signal, upsample_factor)
    upsampled_mon = upsample_signal(mon_signal, upsample_factor)

    base_peaks, base_troughs = find_peaks_and_troughs(upsampled_base)
    mon_peaks, mon_troughs = find_peaks_and_troughs(upsampled_mon)

    paired_peaks = match_features(base_peaks, mon_peaks)
    paired_troughs = match_features(base_troughs, mon_troughs)
    paired_peaks, paired_troughs = sort_features(paired_peaks, paired_troughs)
    waypoints = paired_peaks + paired_troughs

    shifts, cost_matrix, path = fbw_with_feature_based_cost(upsampled_base, upsampled_mon, waypoints)
    shifts = np.array(shifts) / upsample_factor
    shifts = downsample_shifts(shifts, upsample_factor)
    smoothed_shifts = smooth_shifts(shifts, window_size)
    time_strain = np.diff(smoothed_shifts)
        
    if check_strain_limit(smoothed_shifts, strain_limit):
        return smoothed_shifts, time_strain

    return np.full_like(base_signal, np.nan), np.full_like(base_signal, np.nan)

    
def process_signal_pair(args):
    i, j, base_signal, mon_signal, strain_limit, upsample_factor = args
    smoothed_shifts, time_strain = fbw_with_strain_limit(base_signal, mon_signal, strain_limit, upsample_factor, window_size)
    return i, j, smoothed_shifts, time_strain

def process_3d_signals(base_signals, mon_signals, strain_limit, upsample_factor):
    n_slices, n_rows, n_cols = base_signals.shape
    shifts_3d = np.full((n_slices, n_rows, n_cols), np.nan)
    time_strain_3d = np.full((n_slices, n_rows, n_cols), np.nan)

    args = [(i, j, base_signals[i, j, :], mon_signals[i, j, :], strain_limit, upsample_factor) 
            for i in range(n_slices) for j in range(n_rows)]

    with mp.Pool(mp.cpu_count()) as pool:
        for i, j, shifts, time_strain in tqdm(pool.imap_unordered(process_signal_pair, args), total=n_slices * n_rows, desc="Processing", leave=True, ncols=100):
            if shifts.shape[0] != n_cols:  # Ensure the lengths match
                if shifts.shape[0] < n_cols:
                    # Pad shifts to match the length
                    shifts = np.pad(shifts, (0, n_cols - shifts.shape[0]), 'edge')
                else:
                    # Truncate shifts to match the length
                    shifts = shifts[:n_cols]

            if time_strain.shape[0] != n_cols:  # Ensure the lengths match
                if time_strain.shape[0] < n_cols:
                    # Pad time_strain to match the length
                    time_strain = np.pad(time_strain, (0, n_cols - time_strain.shape[0]), 'edge')
                else:
                    # Truncate time_strain to match the length
                    time_strain = time_strain[:n_cols]

            shifts_3d[i, j, :] = shifts
            time_strain_3d[i, j, :] = time_strain

    return shifts_3d, time_strain_3d

strain_limit = config.STRAIN_LIMIT
upsample_factor = config.UPSAMPLE_FACTOR
window_size = config.WINDOW_SIZE