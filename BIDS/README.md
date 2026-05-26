# README

## Details related to access to the data (TODO)

- [ ] Data user agreement

If the dataset requires a data user agreement, link to the relevant information.

- [ ] Contact person

Indicate the name and contact details (email and ORCID) of the person responsible for additional information.

- [ ] Practical information to access the data

If there is any special information related to access rights or
how to download the data make sure to include it.
For example, if the dataset was curated using datalad,
make sure to include the relevant section from the datalad handbook:
http://handbook.datalad.org/en/latest/basics/101-180-FAQ.html#how-can-i-help-others-get-started-with-a-shared-dataset

## Overview

### EEG Emotion Recognition - Muse Headset
#### 2023-2024

The experiment consists in 40 sessions per user. During each session, users are asked to watch a
music video with the aim to understand their emotions.  
Recordings are performed with a Muse EEG headset at a 256 Hz sampling rate.  
Channels are recorded as follows:
- Channel 0: AF7
- Channel 1: TP9
- Channel 2: TP10
- Channel 3: AF8

The chosen songs have various Last.fm tags in order to create different feelings. The title of every track
can be found in the "TaskName" field of sub-ID***_ses-S***_task-Default_run-001_eeg.json, while the author,
the Last.fm tag and additional information in "TaskDescription".

## Methods

### Subjects

The subject pool is made of 30 college students, aged between 18 and 35. 16 of them are males, 14 females.

### Apparatus

The experiment was performed using the same procedures as those to create
[Deap Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), which is a dataset to recognize emotions via a Brain
Computer Interface (BCI).  


### Task organization

Firstly, music videos were selected. Once 40 songs were picked, the protocol was chosen and the self-assessment
questionnaire was created.

### Task details

In order to evaluate the stimulus, Russell's VAD (Valence-Arousal-Dominance) scale was used.  
In this scale, valenza-arousal space can be divided in four quadrants:
- Low Arousal/Low Valence (LALV);
- Low Arousal/High Valence (LAHV);
- High Arousal/Low Valence (HALV);
- High Arousal/High Valence (HAHV).

### Experimental location

The experiment was performed in a laboratory located at DEI Department of
[Politecnico di Bari](https://www.poliba.it/).

### Missing data
Data recorded during session S023 performed by ID021 user was corrupted, therefore is missing.  
Sessions S033 and S038 of ID015 user show a calculated effective sampling rate lower than 256 Hz:
- ID015_ses-S033 has 226.1320 Hz
- ID015_ses-S038 has 216.9549 Hz