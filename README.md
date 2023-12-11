This repo contains the code, data, and best model for the BALDR study titled, "A Semi-Supervised Learning Method to Produce Explainable Radioisotope Proportion Estimates for NaI-based Synthetic and Measured Gamma Spectra."

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10223445.svg)](https://doi.org/10.5281/zenodo.10223445)


# Authors

- Alan Van Omen (ajvanom@sandia.gov)
- Tyler Morrow (tmorro@sandia.gov)


# Acknowledgements

This work was funded by the U.S. Department of Energy, National Nuclear Security Administration, Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D).

The authors are with Sandia National Laboratories, a multi-mission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.

# License and Copyright

License and copyright details can be found [here](./LICENSE.txt).


# Manifest

## `study/`

This directory contains the code used to generate data, models, and results.
It also contains some initial data files, as well as the best model found in our study, which are discussed in subsections.
There is also a `requirements.txt` file which contains list of packages you will need to install in your Python environment (tested with Python 3.9.6 on macOS Ventura), such as by using `pip`:

```sh
pip install -r requirements.txt
```

The code files intended for you to run begin with "step#" where "#" indicates the execution order.
Doc-strings in each code file describe their overall purpose.
The final step is running a Jupyter notebook which use the generated data and models to produce results, primarily plots.
The `config.py` file is referenced by all steps and centralizes nearly all critical parameters.


### `study/data/`

This directory contains the Detector Response Function (DRF) (`Detector.dat`), seeds obtained from GADRAS injection (`seeds.pcf`), and the seed specification file (`seeds.yaml`).

`Detector.dat` and `seeds.yaml` were used by a PyRIID synthesizer to obtain `seeds.pcf` from a batch injection in GADRAS [\[1\]][1][\[2\]][2][\[3\]][3].
To reproduce this, you will need to first copy-paste `Detector.dat` into a new subfolder within your GADRAS Detector folder, and update the `gamma_detector -> name` portion of `seeds.yaml` to reflect the relative location of the folder you just created.
Note that only the unshielded variants of the seeds were actually used in the study; shielded variants are just extra.

`measurements.h5` contains all lab measurements used in the study, including a background measurement.
Both `.pcf` and `.h5` can be read in using PyRIID's `read_pcf()` and `read_hdf()` functions, respectively.


### `study/models/`

If you do not want to perform synthesis and train a model, this directory contains the pre-trained "best" model we found which was then used to generate the primary, final results.
The model file is in the ONNX format and is accompanied by a JSON file providing additional, describing metadata [\[4\]][4].
If you are re-running the experiment, note that this folder is where models are output, so you will see many more appear.

The ONNX file will require the ONNX runtime to load and execute; it is primary a binary file of sorts and not meant to be viewed.
Its accompanying JSON file can be viewed in your text editor of choice.


# Methodology

At a high level, the code represents the following process to conduct the study:

- `seeds.pcf` was synthesized using `seeds.yaml` and `Detector.dat` with the PyRIID `SeedSynthesizer` which calls into GADRAS to perform an injection.
- Seeds were split into foreground and background seeds, then each set was mixed using the PyRIID `SeedMixer`.
    - Additional splits were also performed to divide in-distribution and out-of-distribution data.
- Mixed seeds were turned into noisy gamma spectra randomly varying in terms of signal-to-noise ratio using the PyRIID `StaticSynthesizer`.
- Training data was then used to fit many models. Note that training consisted of tuning which looked for optimal hyperparameters, and then a final step which trained the "optimal" model.
- With a model found, the test data was then used by a Jupyter notebook to generate results, mainly plots.


# References

1. [PCF File Format][1]
2. [GADRAS-DRF Version 18][2]
3. [PyRIID][3]
4. [ONNX][4]


[1]: https://doi.org/10.2172/1762353
[2]: https://doi.org/10.2172/1592910
[3]: https://doi.org/10.11578/dc.20221017.2
[4]: https://github.com/onnx/onnx
