# config.py
# Data paths
BASE_PATH = "data/input/B2-Butter-0-5-25-35.sgy"
MONITOR_PATHS = ["data/input/M1-Butter-0-5-25-35.sgy", "data/input/M2-Butter-0-5-25-35.sgy", "data/input/M3-Butter-0-5-25-35.sgy", "data/input/M4-Butter-0-5-25-35.sgy"]
OUTPUT_STRAIN_PATH = "data/output/Time_Strain_3D.sgy"
OUTPUT_SHIFT_PATH = "data/output/Time_Shift_3D.sgy"

# Signal Processing Parameters
UPSAMPLE_FACTOR = 10 # Increases accuracy
WINDOW_SIZE = 10 # Smoothing window size
ORDER = 7 # Changes the sensitivity of the feature detection

# Warping Parameters
STRAIN_LIMIT = 2 # To prevent cycle skipping
METHOD = "bresenham" # A strict option which draws straight lines between features which the path must follow. "voronoi" allows for flexibility and is when dynamic programming really applies.

# SEG-Y Parameters (byte locations)
ILINE = 5
XLINE = 21
ILINE_OUTPUT = 189
XLINE_OUTPUT = 193
CDPX = 73
CDPY = 77
VERT_DOMAIN = "TWT"
TWT_RANGE = (850, 1110)