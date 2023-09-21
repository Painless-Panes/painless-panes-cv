#### 0. Organize Training Data

Before getting started, you need to organize your annotated data from the data generation step as follows.
Let `$DG` be the data generation directory, which in this repo is
`../01-data-generation/`, and follow these steps:

1. Move the `classes.txt` file from `$DG` into this directory.
2. Create folders `train/` and `val/` here.
3. Create `val/images/` and `val/labels/` subfolders and move the last six images and
labels from `$DG/images/` and `$DG/labels/` into them, making sure the labels match.
4. Move the `$DG/images/` and `$DG/labels/` subfolders with the remaining images and
labels into the `train/` directory.
5. Create an `annotated_data.yaml` file here and add the following information to it.
(You can find the number of classes (`nc`) and their names in order (`names`) in the `classes.txt` file.)
```
train: /absolute/path/to/train/folder
val: /absolute/path/to/val/folder

nc: 3

names: ["class-name-1", "class-name-2", "class-name-3"]
```
6. Optionally, if you have unused images in `$DG/unused-images`, you can copy those into a `test/` directory here in order to test the model after training.

#### 1. Train the model

Run the training script:
```
python 01_train.py
```

#### 2. Test the model

The model weights are stored in `runs/train/weights/best.pt`. Copy that file into this directory and rename it `custom-model.pt`.

You can then test the model by running the prediction script:
```
python 02_predict.py
```
