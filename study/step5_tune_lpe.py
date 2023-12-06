# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Train many models to identify suitable non-beta hyperparameters.
"""
import os
import time

import joblib
import optuna
import tensorflow as tf
from config import (IND_FG_SEED_FILE, MEASUREMENT_MATCH_SYNTHETIC_FILE,
                    NORMALIZE, TARGET_BINS, TRAIN_FILE, TUNING_DIR)
from riid.data.sampleset import read_hdf
from riid.models.neural_nets import LabelProportionEstimator
from sklearn.metrics import mean_absolute_error

time_str = time.strftime("%Y%m%d-%H%M%S")

# Import and pre-process data
train_ss = read_hdf(TRAIN_FILE)
test_ss = read_hdf(MEASUREMENT_MATCH_SYNTHETIC_FILE)
test_ss.normalize(p=1)

y_test = test_ss[:].sources.groupby(axis=1, level="Isotope").sum()
y_test["Cf252"] = 0
y_test["Eu152"] = 0
y_test = y_test.reindex(sorted(y_test.columns), axis=1).astype(float)

if NORMALIZE:
    train_ss.normalize()
    test_ss.normalize()

ind_fg_seeds_ss = read_hdf(IND_FG_SEED_FILE)
ind_fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)


def objective(trial):
    hidden_layer_1 = trial.suggest_int("hidden_layer_1", 16, 256)
    hidden_layer_2 = trial.suggest_int("hidden_layer_2", 16, 64)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.05)
    batch_size = trial.suggest_int("batch_size", 32, 512)
    kernel_l2_reg = trial.suggest_float("kernel_l2_reg", 0.0, 0.001)
    kernel_l1_reg = trial.suggest_float("kernel_l1_reg", 0.0, 0.001)
    activity_l2_reg = trial.suggest_float("activity_l2_reg", 0.0, 0.001)
    activity_l1_reg = trial.suggest_float("activity_l1_reg", 0.0, 0.001)
    dropout = trial.suggest_float("dropout", 0.0, 0.05)
    init_learning_rate = trial.suggest_float("init_learning_rate", 0.005, 0.015)
    activation = trial.suggest_categorical(
        "activation",
        ["mish", "relu", "tanh", "softplus"]
    )

    model = LabelProportionEstimator(
        hidden_layers=(hidden_layer_1, hidden_layer_2,),
        sup_loss="sparsemax",
        unsup_loss="jsd",
        beta=0.5,
        fg_dict=None,
        optimizer="adam",
        optimizer_kwargs={"epsilon": epsilon},
        learning_rate=init_learning_rate,
        hidden_layer_activation=activation,
        kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=kernel_l1_reg,
            l2=kernel_l2_reg
        ),
        activity_regularizer=tf.keras.regularizers.L1L2(
            l1=activity_l1_reg,
            l2=activity_l2_reg
        ),
        dropout=dropout,
        target_level="Isotope",
    )

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2
    )]

    model.fit(
        seeds_ss=ind_fg_seeds_ss,
        ss=train_ss,
        batch_size=batch_size,
        epochs=25,
        validation_split=0.2,
        callbacks=callbacks,
        patience=3,
        verbose=True,
        normalize_scaler=5
    )

    model.predict(test_ss)

    y_pred = test_ss.prediction_probas.groupby(
        level="Isotope",
        axis=1
    ).sum().values

    val_mae = mean_absolute_error(
        y_test,
        y_pred
    )

    return val_mae


# Run the optuna study
journal_dir = os.path.join(TUNING_DIR, f"journal_lpe_{time_str}.log")
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(journal_dir),
)
study = optuna.create_study(
    direction="minimize",
    storage=storage,
    study_name=f"lpe_{time_str}"
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

output_dir = os.path.join(TUNING_DIR, f"study_lpe_{time_str}.pkl")
joblib.dump(study, output_dir)
print(study.best_trial)
