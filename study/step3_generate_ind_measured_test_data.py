# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Generates in-distribution, pseudo-measured testing data.
"""
from config import (BG_CPS, FG_ALPHA, FG_MIX_SIZE, FG_SEED_FILE,
                    IND_FG_SEED_FILE, LIVE_TIME_RANGE, MEASURED_BG_MIX_SAMPLES,
                    MEASURED_LLD, MEASURED_MIX_SIZE, MEASURED_SPS,
                    MEASUREMENT_FILE, MEASUREMENT_MATCH_SYNTHETIC_FILE,
                    MEASUREMENT_TEST_MEASURED_FILE, SNR_RANGE, SNR_SAMPLING,
                    TARGET_BINS, TRAIN_BG_MIXTURES_FILE)
from riid.data.sampleset import SpectraState, SpectraType, read_hdf
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

measurements_ss = read_hdf(MEASUREMENT_FILE)
measurements_ss.spectra_state = SpectraState.Counts
gross_measurements_ss, bg_measurements_ss = measurements_ss \
    .split_fg_and_bg(["Background"])
gross_measurements_ss.spectra_type = SpectraType.Gross
bg_measurements_ss.spectra_type = SpectraType.Background
fg_measurements_ss = gross_measurements_ss - bg_measurements_ss

fg_seeds_ss = read_hdf(FG_SEED_FILE)
fg_measurements_ss = fg_measurements_ss.as_ecal(*fg_seeds_ss.ecal[0])

fg_measurements_ss.downsample_spectra(target_bins=TARGET_BINS)
fg_measurements_ss.spectra.iloc[:, :MEASURED_LLD] = 0
fg_measurements_ss.normalize(p=1)

ind_seeds_ss = read_hdf(IND_FG_SEED_FILE)

# Drop sources not found in measurements
ind_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
ind_seeds_ss.drop_sources(["Eu152", "Cf252"], target_level="Isotope")
ind_seeds_ss.drop_spectra_with_no_contributors()

srt_order = [19, 13, 9, 3, 2, 18, 7, 20, 4, 15, 16, 5,
             21, 12, 10, 6, 8, 14, 1, 11, 0, 17]
fg_measurements_ss = fg_measurements_ss[srt_order]

fg_measurement_seed_inds = [3, 7, 10, 14]
fg_measurement_seeds_ss = fg_measurements_ss[fg_measurement_seed_inds]
fg_measurement_seeds_ss.drop_sources_columns_with_all_zeros()

train_bg_mixtures_ss = read_hdf(TRAIN_BG_MIXTURES_FILE)
mixed_bg_seeds_ss = train_bg_mixtures_ss.sample(MEASURED_BG_MIX_SAMPLES)

# Mix measurements
measurement_mixer = SeedMixer(
    fg_measurement_seeds_ss,
    mixture_size=FG_MIX_SIZE,
    dirichlet_alpha=FG_ALPHA
)
mixed_measurement_seeds_ss = measurement_mixer.generate(MEASURED_MIX_SIZE)

# Mix in-distribution foregrounds
fg_seed_mixer = SeedMixer(
    ind_seeds_ss,
    mixture_size=FG_MIX_SIZE,
    dirichlet_alpha=FG_ALPHA
)
mixed_synthetic_seeds_ss = fg_seed_mixer.generate(MEASURED_MIX_SIZE)

static_syn = StaticSynthesizer(
    samples_per_seed=MEASURED_SPS,
    bg_cps=BG_CPS,
    live_time_function="uniform",
    live_time_function_args=LIVE_TIME_RANGE,
    snr_function=SNR_SAMPLING,
    snr_function_args=SNR_RANGE,
    apply_poisson_noise=True,
)

test_fg_measurement_ss, _ = static_syn.generate(
    mixed_measurement_seeds_ss,
    mixed_bg_seeds_ss
)

test_fg_synthetic_ss, _ = static_syn.generate(
    mixed_synthetic_seeds_ss,
    mixed_bg_seeds_ss
)

test_fg_measurement_ss.clip_negatives()
test_fg_synthetic_ss.clip_negatives()
test_fg_measurement_ss.normalize_sources()
test_fg_synthetic_ss.normalize_sources()
test_fg_measurement_ss.drop_spectra_with_no_contributors()
test_fg_synthetic_ss.drop_spectra_with_no_contributors()

test_fg_measurement_ss.to_hdf(MEASUREMENT_TEST_MEASURED_FILE)
test_fg_synthetic_ss.to_hdf(MEASUREMENT_MATCH_SYNTHETIC_FILE)
