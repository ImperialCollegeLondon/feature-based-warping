import numpy as np
from data_input_preprocessing import prepare_data, save_to_segy, save_results_as_segy
from signal_processing_warping import process_3d_signals
from post_processing_visualization import visualize_segy_output
import config

strain_limit = config.STRAIN_LIMIT
upsample_factor = config.UPSAMPLE_FACTOR
window_size = config.WINDOW_SIZE

def main():
    """
    Main function to test the entire workflow.

    Loads data, applies preprocessing, runs the warping process,
    and performs post-processing and visualization.
    """
    # Define paths to base and monitor SEG-Y files
    base_path = config.BASE_PATH
    monitor_paths = config.MONITOR_PATHS

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data, base_dataset = prepare_data(base_path, monitor_paths)

    base = data["base"]
    monitors = [data[f"mon{i+1}"] for i in range(len(monitor_paths))]

    # For simplicity, assuming one monitor for now
    if len(monitors) > 1:
        print("Multiple monitors detected. Using the first monitor for testing.")
    mon = monitors[0]

    # Run the warping process
    print("Running the warping process...")
    shifts_3d, time_strain_3d = process_3d_signals(base, mon, strain_limit, upsample_factor)
    print(f"Percentage of dropped traces: {100*np.isnan(time_strain_3d).sum()/time_strain_3d.size} %")
    
    # Save outputs
    print("Saving outputs...")
    save_results_as_segy(base_dataset, time_strain_3d, shifts_3d)
    
    # Visualize
    # After saving outputs
    print("Saving outputs completed.")
    print("Starting visualization...")

    # Visualize the time strain output
    visualize_segy_output()

if __name__ == "__main__":
    main()
