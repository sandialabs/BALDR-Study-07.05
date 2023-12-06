# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Train a final label proportion estimator (LPE).
"""
import os
import time

import tensorflow as tf
from config import (IND_FG_SEED_FILE, MODEL_DIR, NORMALIZE, TARGET_BINS,
                    TRAIN_FILE)
from riid.data.sampleset import read_hdf
from riid.models.neural_nets import LabelProportionEstimator

time_str = time.strftime("%Y%m%d-%H%M%S")
lr = 0.012
batch_size = 86
epochs = 30
beta = 0.85
unsup_loss = "jsd"
sup_loss = "sparsemax"

train_ss = read_hdf(TRAIN_FILE)

if NORMALIZE:
    train_ss.normalize()

x = train_ss.get_samples().astype(float)
y = train_ss.get_source_contributions().astype(float)

ind_fg_seeds_ss = read_hdf(IND_FG_SEED_FILE)
ind_fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


model = LabelProportionEstimator(
    hidden_layers=(173, 63,),
    sup_loss=sup_loss,
    unsup_loss=unsup_loss,
    beta=beta,
    fg_dict=None,
    optimizer="adam",
    optimizer_kwargs={"epsilon": 0.017},
    learning_rate=lr,
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
    patience=3
)]

history = model.fit(
    seeds_ss=ind_fg_seeds_ss,
    ss=train_ss,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks,
    patience=5,
    verbose=True,
    es_min_delta=1e-4,
    normalize_scaler=5
)

model_path = os.path.join(
    MODEL_DIR,
    f"lpe_{sup_loss}_{unsup_loss}_beta{beta}_{time_str}.onnx"
)
model.save(model_path)
