# config.py
# Data paths
BASE_PATH = "data/input/B2-Butter-25-35.sgy"
MONITOR_PATHS = ["data/input/M1-Butter-25-35.sgy", "data/input/M2-Butter-25-35.sgy", "data/input/M3-Butter-25-35.sgy", "data/input/M4-Butter-25-35.sgy"]
OUTPUT_STRAIN_PATH = "data/output/Time_Strain_3D.sgy"
OUTPUT_SHIFT_PATH = "data/output/Time_Shift_3D.sgy"

# Signal Processing Parameters
UPSAMPLE_FACTOR = 10
WINDOW_SIZE = 10
ORDER = 7

# Warping Parameters
STRAIN_LIMIT = 2
METHOD = "bresenham"

# SEG-Y Parameters
ILINE = 5
XLINE = 21
ILINE_OUTPUT = 189
XLINE_OUTPUT = 193
CDPX = 73
CDPY = 77
VERT_DOMAIN = "TWT"
TWT_RANGE = (850, 1110)

# Post Processing Parameters
SLICE_IDX = 38
CROSSLINE_IDX = 48