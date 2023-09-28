# Datasheet

## Motivation

The dataset was created to train and test CNN models on denoising multidimensional data (fluorescence recordings).

The data that comprises the dataset was acquired by the research group of Prof. Anton Sirota, from the Bernstein Center for Computational Biology, in the Ludwig-Maximillians University in Munich. There was no specific funding for the creation of this dataset.


## Composition

The dataset is composed of time-series of multiple signal components, namely Acetylcholine levels, hemodynamics and noise, used to generate synthetic training instances during the training process.

The content of the dataset has not been publicly released yet.

## Collection process

- How was the data acquired?

The data was acquired from brain fluorescence recordings in freely-moving mice. Fluorescence was measured using optic fibers implanted in the hippocampus upon inducing expression of a fluorescence reporter for Acetylcholine.

This dataset is a sample of a larger set. It was collected from a single recording session in a mouse.
The data was collected in 2019, over a 2 months time frame.

## Preprocessing/cleaning/labelling

The dataset contains components extracted from the raw signal following a processing pipeline, which is illustrated in the project, in the [Test_models](notebooks/Test_models.ipynb) notebook.

 
## Uses

The dataset was generated for a very specific purpose. Thus, usage to address questions beyond this project is very limited.


## Distribution

No copyright restrictions. However, the author would appreciate a citation of the source of this dataset, in case it is used or published elsewhere.

## Maintenance

The dataset is maintained by Ricardo Santos, the owner of this repository.
