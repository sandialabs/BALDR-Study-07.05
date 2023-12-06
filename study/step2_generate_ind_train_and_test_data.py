# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Generates in-distribution, synthetic, training and testing data.
"""
from config import (BG_CPS, BG_MIX_SAMPLES, LIVE_TIME_RANGE, SNR_RANGE,
                    SNR_SAMPLING, SPS, TEST_FG_MIXTURES_FILE, TEST_FILE,
                    TEST_REDUC, TRAIN_BG_MIXTURES_FILE, TRAIN_FG_MIXTURES_FILE,
                    TRAIN_FILE)
from riid.data.sampleset import read_hdf
from riid.data.synthetic.static import StaticSynthesizer

train_mixed_fg_seeds_ss = read_hdf(TRAIN_FG_MIXTURES_FILE)
train_mixed_bg_seeds_ss = read_hdf(TRAIN_BG_MIXTURES_FILE)

test_mixed_fg_seeds_ss = read_hdf(TEST_FG_MIXTURES_FILE)
test_mixed_bg_seeds_ss = train_mixed_bg_seeds_ss.sample(
    int(BG_MIX_SAMPLES / TEST_REDUC)
)

# Generate train/test data
static_syn = StaticSynthesizer(
    samples_per_seed=SPS,
    bg_cps=BG_CPS,
    live_time_function="uniform",
    live_time_function_args=LIVE_TIME_RANGE,
    snr_function=SNR_SAMPLING,
    snr_function_args=SNR_RANGE,
    apply_poisson_noise=True,
)

train_fg_ss, _ = static_syn.generate(
    fg_seeds_ss=train_mixed_fg_seeds_ss,
    bg_seeds_ss=train_mixed_bg_seeds_ss
)

static_syn.samples_per_seed = int(SPS / TEST_REDUC)
test_fg_ss, _ = static_syn.generate(
    fg_seeds_ss=test_mixed_fg_seeds_ss,
    bg_seeds_ss=test_mixed_bg_seeds_ss
)

train_fg_ss.clip_negatives()
test_fg_ss.clip_negatives()
train_fg_ss.normalize_sources()
test_fg_ss.normalize_sources()
train_fg_ss.drop_spectra_with_no_contributors()
test_fg_ss.drop_spectra_with_no_contributors()

# Save SampleSets
train_fg_ss.to_hdf(TRAIN_FILE, verbose=True)
test_fg_ss.to_hdf(TEST_FILE, verbose=True)
