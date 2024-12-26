import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import config  # Import centralized parameter management
import segyio
from mayavi import mlab

def calculate_statistics(shifts_3d):
    """
    Calculate basic statistics for the 3D shifts.

    Parameters:
        shifts_3d (numpy.ndarray): 3D array of shift values.

    Returns:
        dict: Dictionary containing mean, median, and standard deviation of shifts.
    """
    stats = {
        "mean": np.nanmean(shifts_3d),
        "median": np.nanmedian(shifts_3d),
        "std": np.nanstd(shifts_3d),
    }
    return stats

def plot_shifts_histogram(shifts_3d, bins=20):
    """
    Plot a histogram of the shift values.

    Parameters:
        shifts_3d (numpy.ndarray): 3D array of shift values.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(shifts_3d[~np.isnan(shifts_3d)].flatten(), bins=bins, color='blue', alpha=0.7)
    plt.title("Histogram of Shift Values")
    plt.xlabel("Shift Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_shifts_slice(shifts_3d, slice_idx):
    """
    Plot a specific slice of the 3D shifts.

    Parameters:
        shifts_3d (numpy.ndarray): 3D array of shift values.
        slice_idx (int): Index of the slice to plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(shifts_3d[slice_idx, :, :], cmap="viridis", aspect="auto")
    plt.colorbar(label="Shift Value")
    plt.title(f"Shift Values - Slice {slice_idx}")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

def remove_nans(data_3d):
    """
    Replace NaN values in a 3D array with zeros.

    Parameters:
        data_3d (numpy.ndarray): Input 3D array with NaN values.

    Returns:
        numpy.ndarray: Array with NaN values replaced by zeros.
    """
    return np.nan_to_num(data_3d, nan=0.0)

def smooth_data(data_3d, filter_size=10):
    """
    Smooth 3D data using a uniform filter.

    Parameters:
        data_3d (numpy.ndarray): Input 3D array to smooth.
        filter_size (int): Size of the uniform filter.

    Returns:
        numpy.ndarray: Smoothed 3D array.
    """
    return uniform_filter(data_3d, size=filter_size)

def plot_time_strain(time_strain_3d, slice_idx):
    """
    Plot the time strain for a specific slice.

    Parameters:
        time_strain_3d (numpy.ndarray): 3D array of time strain values.
        slice_idx (int): Index of the slice to plot.
    """
    plt.imshow(time_strain_3d[:, :, slice_idx], vmin=-0.3, vmax=0.3, aspect='auto', cmap='seismic')
    plt.colorbar(label='Time Strain, unitless')
    plt.xlabel('Inline')
    plt.ylabel('Crossline')
    plt.show()

def plot_crossline_section(time_strain_3d, crossline_idx):
    """
    Plot a crossline section of the time strain.

    Parameters:
        time_strain_3d (numpy.ndarray): 3D array of time strain values.
        crossline_idx (int): Index of the crossline to plot.
    """
    plt.figure(figsize=(6, 8))
    plt.imshow(time_strain_3d[:, crossline_idx, :].T, aspect='auto', origin='upper', vmin=-0.5, vmax=0.5, cmap='turbo')
    plt.colorbar(label='Time Strain, unitless')
    plt.xlabel('Crossline')
    plt.ylabel('TWT')
    plt.show()

def post_process_and_visualize(shifts_3d, time_strain_3d):
    """
    Perform post-processing and visualization of the 3D shifts and time strain.

    Parameters:
        shifts_3d (numpy.ndarray): 3D array of shift values.
        time_strain_3d (numpy.ndarray): 3D array of time strain values.
    """
    print("Removing NaNs...")
    time_strain_3d = remove_nans(time_strain_3d)
    shifts_3d = remove_nans(shifts_3d)

    print("Smoothing data...")
    time_strain_3d = smooth_data(time_strain_3d)
    shifts_3d = smooth_data(shifts_3d)

    print("Plotting time strain slice...")
    slice_idx = config.SLICE_IDX if hasattr(config, 'SLICE_IDX') else 38
    plot_time_strain(time_strain_3d, slice_idx)

    print("Plotting crossline section...")
    crossline_idx = config.CROSSLINE_IDX if hasattr(config, 'CROSSLINE_IDX') else 66
    plot_crossline_section(time_strain_3d, crossline_idx)

    print("Calculating statistics...")
    stats = calculate_statistics(shifts_3d)
    print("Statistics:", stats)

    print("Plotting histogram...")
    plot_shifts_histogram(shifts_3d)
    
def load_segy_cube(file_path):
    """
    Load a SEG-Y file and return its data cube, inlines, xlines, and depth samples.

    Parameters:
        file_path (str): Path to the SEG-Y file.

    Returns:
        tuple: data_cube, inlines, xlines, depth_samples
    """
    with segyio.open(file_path, ignore_geometry=False) as segy_file:
        segy_file.mmap()
        depth = segy_file.samples
        inlines = segy_file.ilines
        xlines = segy_file.xlines
        data_cube = segyio.tools.cube(segy_file)

    return data_cube, inlines, xlines, depth

def explore3d(data_cube, inlines, xlines, depth, preset=True, I=-1, X=-1, Z=-1):
    """
    Visualize a 3D data cube using Mayavi.

    Parameters:
        data_cube (numpy.ndarray): 3D data cube.
        inlines (numpy.ndarray): Inline coordinates.
        xlines (numpy.ndarray): Crossline coordinates.
        depth (numpy.ndarray): Depth samples.
        preset (bool): If True, automatically set slice indices to the center of the cube.
        I (int): Inline slice index.
        X (int): Crossline slice index.
        Z (int): Depth slice index.
    """
    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]  # Adjust spacing as needed
    vm = np.percentile(data_cube, 95)  # Dynamic range threshold

    if preset:
        nx, ny, nz = data_cube.shape
        I = nx // 2
        X = ny // 2
        Z = nz // 2

    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes',
                                     slice_index=I, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes',
                                     slice_index=X, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes',
                                     slice_index=Z, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.show()

# Main Execution for Visualization
def visualize_segy_output():
    """
    Visualize the output SEG-Y file using Mayavi.
    """
    print("Loading SEG-Y output for visualization...")
    data, inlines, xlines, depth = load_segy_cube(config.OUTPUT_STRAIN_PATH)

    # Default geological features of interest (center of the cube)
    I = np.where(inlines == inlines[len(inlines) // 2])[0][0]
    X = np.where(xlines == xlines[len(xlines) // 2])[0][0]
    Z = np.where(depth == depth[len(depth) // 2])[0][0]

    print(f"Exploring data at Inline={I}, Crossline={X}, Depth={Z}")
    explore3d(data, inlines, xlines, depth, preset=False, I=I, X=X, Z=Z)
