# Signal and Image Processing

See articles https://arxiv.org/pdf/2011.12401.pdf and https://arxiv.org/pdf/2101.08694.pdf for further information about the signal and image professing implemented in ground-based infrared sky images.

## Pyranometer Signal Processing

See files in the directory: pyranometer_processing/* 

This directory contains the files used for the calibration of the pyranometer via software, so the clear sky index can be extracted without biases in the signal. The bias appear when using high resolution measurements of Global Solar Irradiance. There are two different bias modelled in the signal: amplitude and shifting biases.

## Infrared Images Processing

The processing applied to the images is subdivided in feature extraction (Moist Adiabatic Lapse Rate), cyclostationary effects modelling (Atmospheric Radiometric Model and Outdoor Germanium Camera Window Model), and the atmospheric detection model that uses the images to classify the type of weather conditions displayed on the images.

### Moist Adiabatic Lapse Rate

See file in notebook: moist_adiabatic_lapse_rate.ipynb.

### Atmospheric Radiometric Model

See files in the directory: atmospheric_radiometric_model/* 

The files contain the notebooks with the analysis and the files .py used to cross-validate the models in High Performance Computer. The files named as atmospheric_model_parameters_dataset_vX-X.py compute the optimal parameters of the atmospheric radiometric model. The files named as validate_atmospheric_model_parameters_vX-X.py are used to cross-validate the regression model of the optimal atmospheric radiometric model parameters.

### Atmospheric Conditions Model

See files in the directory: atmospheric_condition_model/*

The files in this folder include the jupyter notebooks used in the analysis of the fetures and the analysis of the results. The .py files were used in the High Performance Computer to cross-validate the parameters of the model. The features were quantified as statistics or histogram bins (See .py files in repository).

### Outdoor Germanium Camera Window Model

## Dataset

A sample dataset is publicaly available in DRYAD repository: https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m
