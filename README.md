# IntroToML

## Setup

**Note**: The app doesn't work on Apple Silicon processors (M1, M2) due to build problems with `mediapipe` library.

Recommended python version: 3.8

```
# [OPTIONAL] create conda environment
conda create -n faceage python=3.8
conda activate faceage

# install requirements
pip install -r requirements.txt
```

## Run webcam app

(you don't need to train anything to run the app because it uses saved model from `models/best-checkpoint.ckpt`)

```
streamlit run src/app.py
```

## Run training

First download dataset from: <br>
https://www.kaggle.com/datasets/jangedoo/utkface-new

Unpack the data to `data/` folder. Your folder structure should look like this:

```
data
├── archive
│   ├── UTKFace
│   ├── ...
```

Next you should process dataset by running `notebooks/data_generation.ipynb`. This can take about 5-20 minutes depending on your hardware. The output weights around 25GB and will be saved to `data/face_age_dataset/` folder.

Now you can run training:

```
python src/train.py
```

The default architecture is custom CNN with img (input) size 100x100. 10 epochs should train on CPU for about 10-30 minutes depending on your hardware.

You can change the architecture and hyperparameters in `src/train.py` file.

## Project structure

```
.
├── data
│   ├── archive <- raw dataset downloaded from kaggle
│   └── face_age_dataset <- processed dataset generated after running `notebooks/02_data_generation.ipynb`
│
├── logs <- this folder will be generated when running training, contains model checkpoints and logs
│   └── ...
│
├── models
│   └── best-checkpoint.ckpt <- default checkpoint used by the app
│
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_generation.ipynb
│   ├── 03_predictions_analysis.ipynb
│   ├── 04_case_study1.ipynb
│   └── 05_case_study2.ipynb
│
├── README.md
│
├── requirements.txt <- python dependencies
│
└── src
    ├── data
    │   ├── face_age_datamodule.py <- pytorch lightning datamodule encapsulating pytorch dataset
    │   └── face_age_dataset.py <- pytorch dataset
    │
    ├── models
    │   └── face_age_module.py <- pytorch lightning module encapsulating train/val/test loop
    │   └── architectures.py <- model architectures
    │
    ├── utils
    │   ├── functions.py <- app utilites
    │   ├── face_recognition.py <- app face recognition module
    │   └── predict.py <- app prediction module
    │
    ├── app.py <- run streamlit app
    └── train.py <- run training
```
