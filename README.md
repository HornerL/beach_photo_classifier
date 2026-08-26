# Beach Photo Classifier

This project uses a **SegFormer semantic segmentation model** to classify beach substrate types and calculate substrate percent cover from handheld photographs of mixed-sediment beaches.

Training data were generated using Dan Buscombe's supervised image segmentation tool, [Doodler](https://github.com/Doodleverse/dash_doodler).

## Using the Pre-Trained Model

`Classify_Photos.py` can be used to classify the sample imagery or your own imagery using the existing pre-trained model weights.

## Training Your Own Model

To train a SegFormer model using your own training images and segmentation masks:

1. Use `TrainModel.py` to train the model.
2. Use `EvaluateModel.py` to evaluate model performance.

The dataset structure and class definitions used by the training and evaluation scripts are contained in `DefineDataset.py`.
