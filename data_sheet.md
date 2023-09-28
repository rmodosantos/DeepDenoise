# Datasheet Template

## Motivation

- For what purpose was the dataset created?

The dataset was created to help improving automatic classification of brain states from brain electrophysiological and behavioral recordings in rodents.


- Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?

The dataset was created from data acquired by the research group of Prof. Anton Sirota, from the Bernstein Center for Computational Biology, in the Ludwig-Maximillians University in Munich. There was no specific funding for the creation of this dataset.

 
## Composition

- What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

The instances represent segments of behavioral and electropysiological signals.

- How many instances of each type are there?

Each type has a total of 7725 instances.

- Is there any missing data?

No.

- Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by    doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?

Yes. The content of the dataset has not been publicly released yet.

## Collection process

- How was the data acquired?

The data was acquired from brain electrophysiological and behavioral recordings in freely-moving mice. Brain electrical signals were recorded from micro-wire bundles implanted in the hippocampus. Behavioral data was collected from an accelerometer incorporated in the head-stage pre-amplifier of the electrophysiological setup. Since the head-stage was connected to the implant during recordings, the accelerometer capture the head movements (instantaneous accelerations).

- If the data is a sample of a larger subset, what was the sampling strategy? 

The data is a sample of a larger set. It was collected by randomly picking reference timepoints during the recordings, and then taking a segment corresponding to a window of +- 50 s relative to each reference. The reference timepoints were uniformly distributed with an average rate of 1 point per 10 seconds.

- Over what time frame was the data collected?

The data was collected in 2018, over a 4 months time frame.

## Preprocessing/cleaning/labelling

- Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. 

For each instance, the electrophysiological data consists of local field potential (LFP) power spectrograms over a time  segment. The spectrograms provide a measure of the energy of the signal over a set of frequencies and were obtained by doing a Fast Fourier Transform on the LFP time series.
Movement signal was obtained by integration of the Hilbert power of the signal in the frequency band from 0.5-25 Hz.
Apart from this, no preprocessing was performed on the data.

- Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? 
No.
 
## Uses

- What other tasks could the dataset be used for? 

The dataset was generated for a very specific purpose. Some analysis on the relationship between movement and brain electrical activity can be carried out using this dataset, but to a limited extent.

- Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms? 

The dataset was collected from recordings in two mice. It is possible that it is not fully representative and the models generated with this data will not generality optimally in other mice, or more importantly, other species.

- Are there tasks for which the dataset should not be used? If so, please provide a description.

Not applicable.


## Distribution

- How has the dataset already been distributed? 

This is the first time this dataset is distributed.

- Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?  

No. However, the authors would appreciate a citation of the source of this dataset, in case it is used or published elsewhere.

## Maintenance

- Who maintains the dataset?

The dataset is maintained by Ricardo Santos, the owner of this repository.
