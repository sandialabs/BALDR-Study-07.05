# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Contains global constants for experiment configuration.
"""
import os

# Directories
DATA_DIR = "./data/"
os.makedirs(DATA_DIR, exist_ok=True)
IMAGES_DIR = "./imgs/"
os.makedirs(IMAGES_DIR, exist_ok=True)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
RUN_DIR = "./runs"
os.makedirs(RUN_DIR, exist_ok=True)
TUNING_DIR = "./tuning"
os.makedirs(TUNING_DIR, exist_ok=True)
BETA_TUNING_DIR = "./beta_models/"
os.makedirs(BETA_TUNING_DIR, exist_ok=True)

# Seed files
SEED_FILE = os.path.join(DATA_DIR, "seeds.pcf")
FG_SEED_FILE = os.path.join(DATA_DIR, "fg_seeds.h5")
BG_SEED_FILE = os.path.join(DATA_DIR, "bg_seeds.h5")
IND_FG_SEED_FILE = os.path.join(DATA_DIR, "ind_fg_seeds.h5")
OOD_FG_SEED_FILE = os.path.join(DATA_DIR, "ood_fg_seeds.h5")

# Other files
TRAIN_FILE = os.path.join(DATA_DIR, "ind_synth_train.h5")
TEST_FILE = os.path.join(DATA_DIR, "ind_synth_test.h5")
MEASUREMENT_FILE = os.path.join(DATA_DIR, "measurements.h5")
TRAIN_BG_MIXTURES_FILE = os.path.join(DATA_DIR, "ind_synth_train_bg_mixed_seeds.h5")
TRAIN_FG_MIXTURES_FILE = os.path.join(DATA_DIR, "ind_synth_train_fg_mixed_seeds.h5")
TEST_FG_MIXTURES_FILE = os.path.join(DATA_DIR, "ind_synth_test_fg_mixed_seeds.h5")
MEASUREMENT_TEST_MEASURED_FILE = os.path.join(DATA_DIR, "ind_measured_test.h5")
# The following file represents an in-distribution, synthetic test dataset
# only containing samples for those sources we could actually measure in the lab.
MEASUREMENT_MATCH_SYNTHETIC_FILE = os.path.join(DATA_DIR, "matched_ind_synth_test.h5")

# Mixing
BG_MIX_SIZE = 4
BG_MIX_SAMPLES = 15
BG_ALPHA = 1
FG_MIX_SIZE = 3
FG_MIX_SAMPLES = 300
FG_ALPHA = 1

# Synthesizing
BG_CPS = 300
SPS = 200
SNR_RANGE = (5, 100)
SNR_SAMPLING = "log10"
LIVE_TIME_RANGE = (60, 60)
TEST_REDUC = 4

# Pre-processing
NORMALIZE = True  # Whether to convert spectra to an L1 norm
TARGET_BINS = 256

# Measurement test parameters
MEASURED_LLD = 4
MEASURED_MIX_SIZE = 200
MEASURED_BG_MIX_SAMPLES = 5
MEASURED_SPS = 100
OOD_SPS = 200
