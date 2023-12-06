# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Loads seeds from initial PCF and prepares them for synthesis.
"""
from config import (BG_ALPHA, BG_MIX_SAMPLES, BG_MIX_SIZE, BG_SEED_FILE,
                    FG_ALPHA, FG_MIX_SAMPLES, FG_MIX_SIZE, FG_SEED_FILE,
                    IND_FG_SEED_FILE, OOD_FG_SEED_FILE, SEED_FILE, TARGET_BINS,
                    TEST_FG_MIXTURES_FILE, TEST_REDUC, TRAIN_BG_MIXTURES_FILE,
                    TRAIN_FG_MIXTURES_FILE)
from riid.data.sampleset import SpectraType, read_pcf
from riid.data.synthetic.seed import SeedMixer

# Load in data
seeds_ss = read_pcf(SEED_FILE)
fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()
fg_seeds_ss.normalize()
bg_seeds_ss.normalize()
fg_seeds_ss.to_hdf(FG_SEED_FILE)
bg_seeds_ss.to_hdf(BG_SEED_FILE)

fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
bg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)

# Split in-distribution and out-of-distribution seeds
ind_sources = [
    "Am241,100uCi",
    "Ba133,100uCi",
    "Co60,100uCi",
    "U232,100uCi",
    "Cf252,100uCi",
    "Eu152,100uCi",
]
print(f"IND seeds: {ind_sources}")
ood_fg_seeds_ss, ind_fg_seeds_ss = fg_seeds_ss.split_fg_and_bg(ind_sources)
ind_fg_seeds_ss.to_hdf(IND_FG_SEED_FILE)

ood_sources = [
    "Cs137,100uCi",
    "Bi207,100uCi",
]
print(f"OOD seeds: {ood_sources}")
_, ood_fg_seeds_ss = ood_fg_seeds_ss.split_fg_and_bg(ood_sources)
ood_fg_seeds_ss.spectra_type = SpectraType.Foreground
ood_fg_seeds_ss.to_hdf(OOD_FG_SEED_FILE)

# Mix backgrounds
bg_seed_mixer = SeedMixer(
    bg_seeds_ss,
    mixture_size=BG_MIX_SIZE,
    dirichlet_alpha=BG_ALPHA
)

train_mixed_bg_seeds_ss = bg_seed_mixer.generate(BG_MIX_SAMPLES)
train_mixed_bg_seeds_ss.to_hdf(TRAIN_BG_MIXTURES_FILE)
test_mixed_bg_seeds_ss = train_mixed_bg_seeds_ss.sample(
    int(BG_MIX_SAMPLES / TEST_REDUC)
)

# Mix in-distribution foregrounds
fg_seed_mixer = SeedMixer(
    ind_fg_seeds_ss,
    mixture_size=FG_MIX_SIZE,
    dirichlet_alpha=FG_ALPHA
)

train_mixed_fg_seeds_ss = fg_seed_mixer.generate(FG_MIX_SAMPLES)
test_mixed_fg_seeds_ss = fg_seed_mixer.generate(
    int(FG_MIX_SAMPLES / TEST_REDUC)
)

train_mixed_fg_seeds_ss.to_hdf(TRAIN_FG_MIXTURES_FILE)
test_mixed_fg_seeds_ss.to_hdf(TEST_FG_MIXTURES_FILE)
