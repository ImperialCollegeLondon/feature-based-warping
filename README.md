# Feature-based Warping Toolkit

The Feature-based Warping Toolkit is a Python-based framework designed for time shift analysis of post-stack time-lapse seismic data in a feature based approach. This toolkit enables preprocessing SEG-Y files, performing feature-based warping, saving results in SEG-Y format, and creating interactive 3D visualizations.

---

## Features

- **Data Preprocessing**:
  - Support for SEG-Y data handling.
  
- **Feature-Based Warping**:
  - A warping algorithm using peaks and troughs for feature alignment.
  - Optimized with multiprocessing and Numba for efficient computation.

- **Post-Processing and Visualization**:
  - Save processed outputs (time strain and shifts) in SEG-Y format.
  - Interactive 3D visualization of seismic attributes using Mayavi.

---

## Installation

### Prerequisites
Ensure you have Python 3.5+ installed on your system.

### Steps

1. Clone the repository:
   git clone https://github.com/ImperialCollegeLondon/feature-based-warping.git

2. Install dependencies
   pip install -r requirements.txt

## Quick Start
1. Running the Workflow
To preprocess, warp, save, and visualize seismic data:

python test.py

## Project Structure
SeismicToolkit/
├── README.md                     # Project documentation
├── LICENSE                       # License file
├── requirements.txt              # Dependencies
├── config.py                     # Parameter configurations
├── data_input_preprocessing.py   # Preprocessing functions
├── signal_processing_warping.py  # Signal processing and warping
├── post_processing_visualization.py # Post-processing and visualization
├── test.py                       # Main script for workflow
└── data/
    ├── input/                    # Input SEG-Y files
    └── output/                   # Output SEG-Y files

## Configuration
All configurable parameters are defined in the config.py file. Below are some key parameters:

Paths:
BASE_PATH: Path to the baseline SEG-Y file.
MONITOR_PATHS: List of paths to the monitor SEG-Y files.
OUTPUT_STRAIN_PATH: Path to save the time strain SEG-Y output.
OUTPUT_SHIFT_PATH: Path to save the shifts SEG-Y output.

Processing Parameters:
STRAIN_LIMIT: Maximum allowable time strain.
UPSAMPLE_FACTOR: Factor for upsampling signals.
WINDOW_SIZE: Size of the moving average filter for smoothing.

## Dependencies
The toolkit requires the following libraries:

numpy
matplotlib
segysak
numba
mayavi
tqdm
scipy

## Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a feature branch: git checkout -b feature-name.
Commit your changes: git commit -m "Description of changes".
Push to the branch: git push origin feature-name.
Open a pull request.

## License
This project is licensed under the Apache License.