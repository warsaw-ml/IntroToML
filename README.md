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

```
streamlit run src/app.py
```

## Run training

First download dataset from: <br>
https://www.kaggle.com/datasets/jangedoo/utkface-new

Unpack the data to `data/` folder.

Next you should process dataset by running `notebooks/data_generation.ipynb`

Now you can run training:

```
python src/train.py
```

The default architecture is custom CNN with img (input) size 100x100. 10 epochs should train on CPU for about 10-30 minutes depending on your hardware.
