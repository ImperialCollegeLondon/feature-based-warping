import warnings
import pathlib
import numpy as np
import xarray as xr
from segysak.segy import segy_loader, segy_writer
import config  # Import centralized parameter management

def load_segy_data(file_path, iline, xline, cdpx, cdpy, vert_domain="TWT"):
    """
    Loads SEG-Y data from a given file path.

    Parameters:
        file_path (pathlib.Path): Path to the SEG-Y file.
        iline (int): Inline byte location.
        xline (int): Crossline byte location.
        cdpx (int): CDP X byte location.
        cdpy (int): CDP Y byte location.
        vert_domain (str): Vertical domain, e.g., "TWT".

    Returns:
        xarray.Dataset: Loaded SEG-Y data.
    """
    return segy_loader(file_path, iline=iline, xline=xline, cdpx=cdpx, cdpy=cdpy, vert_domain=vert_domain)

def prepare_data(base_path, monitor_paths):
    """
    Prepares base and monitor datasets based on user-provided paths and parameters from config.py.

    Parameters:
        base_path (pathlib.Path): Path to the base SEG-Y file.
        monitor_paths (list of pathlib.Path): Paths to monitor SEG-Y files.

    Returns:
        dict: Dictionary containing base and monitor data arrays.
    """
    iline = config.ILINE
    xline = config.XLINE
    cdpx = config.CDPX
    cdpy = config.CDPY
    twt_range = config.TWT_RANGE
    vert_domain = config.VERT_DOMAIN

    data = {}

    base_dataset = load_segy_data(base_path, iline, xline, cdpx, cdpy, vert_domain)
    data["base"] = np.asarray(base_dataset.data[:, :, twt_range[0]:twt_range[1]])

    for i, monitor_path in enumerate(monitor_paths):
        monitor_dataset = load_segy_data(monitor_path, iline, xline, cdpx, cdpy, vert_domain)
        data[f"mon{i+1}"] = np.asarray(monitor_dataset.data[:, :, twt_range[0]:twt_range[1]])

    return data, base_dataset

def save_to_segy(base_dataset, output_data_3d, twt_range, output_path):
    """
    Save a 3D array (e.g., time strain or shifts) into a SEG-Y file.

    Parameters:
        base_dataset (xarray.Dataset): The original dataset to copy attributes and dimensions from.
        output_data_3d (numpy.ndarray): The 3D data array to save.
        twt_range (tuple): The TWT range indices (start, end).
        output_path (str): Path to save the SEG-Y file.
    """
    # Step 1: Create an output array with the same shape as the base dataset
    output_data = np.zeros_like(base_dataset.data)

    # Step 2: Assign the computed data (time strain or shifts) to the specific range
    output_data[:, :, twt_range[0]:twt_range[1]] = output_data_3d

    # Step 3: Create an xarray.DataArray and copy attributes from the base dataset
    output_data_array = xr.DataArray(
        output_data,
        coords=base_dataset.coords,
        dims=base_dataset.dims
    )
    output_data_array.attrs = base_dataset.attrs

    # Step 4: Create an xarray.Dataset and assign attributes
    output_dataset = xr.Dataset(data_vars={'data': output_data_array})
    output_dataset = output_dataset.assign_attrs(base_dataset.attrs)

    # Step 5: Write the dataset to a SEG-Y file
    segy_writer(
        output_dataset,
        output_path,
        trace_header_map=dict(
            iline=config.ILINE_OUTPUT,
            xline=config.XLINE_OUTPUT,
            cdp_x=config.CDPX,
            cdp_y=config.CDPY
        )
    )
    print(f"Data saved to SEG-Y file: {output_path}")

def save_results_as_segy(base_dataset, time_strain_3d, shifts_3d):
    """
    Save time strain and shifts data to SEG-Y files.

    Parameters:
        base_dataset (xarray.Dataset): The original dataset to copy attributes and dimensions from.
        time_strain_3d (numpy.ndarray): The 3D array of time strain values.
        shifts_3d (numpy.ndarray): The 3D array of shift values.
    """
    twt_range = config.TWT_RANGE

    # Save time strain data
    output_path_strain = config.OUTPUT_STRAIN_PATH
    save_to_segy(base_dataset, time_strain_3d, twt_range, output_path_strain)

    # Save shift data
    output_path_shifts = config.OUTPUT_SHIFT_PATH
    save_to_segy(base_dataset, shifts_3d, twt_range, output_path_shifts)