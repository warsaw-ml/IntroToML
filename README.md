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

(you don't need to train anything to run the app because it uses saved model)

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

You can change the architecture in `src/train.py` file.
