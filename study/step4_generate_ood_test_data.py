# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Generates out-of-distribution, pseudo-measured, testing data.
"""
import os

from config import (BG_CPS, BG_SEED_FILE, DATA_DIR, FG_SEED_FILE,
                    LIVE_TIME_RANGE, MEASURED_LLD, MEASUREMENT_FILE, OOD_SPS,
                    SNR_RANGE, SNR_SAMPLING, TARGET_BINS,
                    TRAIN_BG_MIXTURES_FILE)
from riid.data.sampleset import SampleSet, SpectraState, SpectraType, read_hdf
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

synthetic_bg_seeds_ss = read_hdf(BG_SEED_FILE)
synthetic_bg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)

measurements_ss = read_hdf(MEASUREMENT_FILE)
measurements_ss.spectra_state = SpectraState.Counts
gross_measurements_ss, bg_measurements_ss = measurements_ss.split_fg_and_bg(
    ["Background"]
)
gross_measurements_ss.spectra_type = SpectraType.Gross
bg_measurements_ss.spectra_type = SpectraType.Background
fg_measurements_ss = gross_measurements_ss - bg_measurements_ss

fg_seeds_ss = read_hdf(FG_SEED_FILE)
fg_measurements_ss = fg_measurements_ss.as_ecal(*fg_seeds_ss.ecal[0])

fg_measurements_ss.downsample_spectra(target_bins=TARGET_BINS)
fg_measurements_ss.spectra.iloc[:, :MEASURED_LLD] = 0
fg_measurements_ss.normalize(p=1)

# resort measurements
srt_order = [
    19, 13, 9, 3, 2, 18, 7, 20, 4, 15, 16,
    5, 21, 12, 10, 6, 8, 14, 1, 11, 0, 17
]
fg_measurements_ss = fg_measurements_ss[srt_order]

measured_ood_seeds_ss = fg_measurements_ss[[18, 21]]

ood_seeds_ss = SampleSet()
ood_seeds_ss.concat([synthetic_bg_seeds_ss, measured_ood_seeds_ss])

ood_seeds_ss.drop_sources_columns_with_all_zeros()

fg_measurement_seed_inds = [3, 7, 10, 14]
fg_measurement_seeds_ss = fg_measurements_ss[fg_measurement_seed_inds]
fg_measurement_seeds_ss.drop_sources_columns_with_all_zeros()

train_bg_mixtures_ss = read_hdf(TRAIN_BG_MIXTURES_FILE)

mixed_bg_seeds_ss = train_bg_mixtures_ss.sample(5)

for i in range(ood_seeds_ss.n_samples):
    ood_seed_ss = ood_seeds_ss[i]
    ood_seed_ss.drop_sources_columns_with_all_zeros()
    seeds_ss = SampleSet()
    seeds_ss.concat([ood_seed_ss, fg_measurement_seeds_ss])

    mixer = SeedMixer(
        seeds_ss=seeds_ss,
        mixture_size=5,
        dirichlet_alpha=1
    )

    syn = StaticSynthesizer(
        samples_per_seed=OOD_SPS,
        bg_cps=BG_CPS,
        live_time_function="uniform",
        live_time_function_args=LIVE_TIME_RANGE,
        snr_function=SNR_SAMPLING,
        snr_function_args=SNR_RANGE,
        apply_poisson_noise=True,
    )

    mixed_fg_seeds_ss = mixer.generate(1000)

    fg_ss, _ = syn.generate(
        mixed_fg_seeds_ss,
        mixed_bg_seeds_ss
    )
    ood_isotope = ood_seed_ss.get_labels()[0]
    output_path = os.path.join(DATA_DIR, f"{ood_isotope}_ood_test.h5")
    fg_ss.to_hdf(output_path)
