# Face Recognition

Face Recognition using a CNN architecture.

## Dataset

Project uses Yale dataset as a demo, but is extensible to any face recognition dataset. Download Yale Face Database from [here](vision.ucsd.edu/content/yale-face-database).

## Running

- Do `pip install -r requirements.txt` to install all deps.
- Get the database as mentioned above. Run `python utils.py <yale_dataset_folder> <data_folder>` to create train and val data from `yale_dataset_folder` into `data_folder`.
- Run `giftopng.sh <folder>` to convert all the images inside the subfolders of `<folder>` into pngs.
- Run using `python main.py --data <data_folder>` where `data_folder` contains folders `train` and `val` containing training and validation data respectively.
- Inside `train` and `val` folders, program expects folder for each of the labels and these folder contains samples for these labels.

## TODO

Add picture for the architecture and tune it.

## License
MIT


