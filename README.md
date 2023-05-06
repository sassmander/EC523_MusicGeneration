# EC523_MusicGeneration
Final project for Spring 2023 EC523 involving music generation task using WaveNet model

### Team members: 

Sally Shin, Yuke Li, William Krska

### Link to Google Drive

https://drive.google.com/drive/u/1/folders/0AEwRQcK0vvTPUk9PVA
# Setup
We highly recommend using Google Co-Lab for the high amounts of storage available.
Download and unpack he Maestro dataset to the workspace. We recommend mounting your Google Drive if you have the room to store it there.
To train/run each other following models, simply use the python notebook. 
# Models
## RoBERTa
To train on the roberta model, run the python notebook from top to bottom. You are able to tweak certain hyperparameters to improve accuracy. Output MIDI files are attempted to be saved to the mounted Google Drive, from which they can be evaluated. 
## Wavenet
To train on the wavenet model, run the python notebook from top to bottom. You are able to tweak certain hyperparameters to improve accuracy. Output CSV files are attempted to be saved to the mounted Google Drive, from which they can be evaluated. 
## GPT
To train on the GPT model, run the python notebook from top to bottom. You are able to tweak certain hyperparameters to improve accuracy. Output MIDI files are attempted to be saved to the mounted Google Drive, from which they can be evaluated. 
# Evaluation
The evaluation python notebook contains both the characterization and evaluation steps of calclating the metrics. Update the filepath to match the locations of the Maestro and Bach Chorales' datasets. Running the program top to bottom will characterize one variant of all network architectures. 
 
