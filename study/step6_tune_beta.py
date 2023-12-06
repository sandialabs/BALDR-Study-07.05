# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Train many models to identify a suitable beta parameter.
"""
import os
import time

import numpy as np
import tensorflow as tf
from config import (BETA_TUNING_DIR, IND_FG_SEED_FILE, NORMALIZE, TARGET_BINS,
                    TRAIN_FILE)
from riid.data.sampleset import read_hdf
from riid.models.neural_nets import LabelProportionEstimator

time_str = time.strftime("%Y%m%d-%H%M%S")
n_trials = 5
betas = np.linspace(0, 1, 21)
unsup_loss = "jsd"
sup_loss = "sparsemax"

model_dir = os.path.join(
    BETA_TUNING_DIR,
    f"models_{sup_loss}_{unsup_loss}_{time_str}"
)
os.makedirs(model_dir)

train_ss = read_hdf(TRAIN_FILE)

if NORMALIZE:
    train_ss.normalize()

x = train_ss.get_samples().astype(float)
y = train_ss.get_source_contributions().astype(float)

ind_fg_seeds_ss = read_hdf(IND_FG_SEED_FILE)
ind_fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)

for beta in betas:
    for trial in range(n_trials):
        model = LabelProportionEstimator(
            hidden_layers=(173, 63,),
            sup_loss=sup_loss,
            unsup_loss=unsup_loss,
            beta=beta,
            fg_dict=None,
            optimizer="adam",
            optimizer_kwargs={"epsilon": 0.017},
            learning_rate=0.012,
            hidden_layer_activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=8.1e-4,
                l2=8.2e-5
            ),
            activity_regularizer=tf.keras.regularizers.L1L2(
                l1=8.4e-4,
                l2=4.7e-4
            ),
            dropout=0.04,
            target_level="Isotope",
        )

        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=2
        )]

        history = model.fit(
            seeds_ss=ind_fg_seeds_ss,
            ss=train_ss,
            batch_size=86,
            epochs=25,
            validation_split=0.2,
            callbacks=callbacks,
            patience=3,
            verbose=True,
            es_min_delta=1e-4,
            normalize_scaler=5
        )

        model_path = os.path.join(
            model_dir,
            f"beta{beta}_trial{trial}.onnx"
        )
        model.save(model_path)
