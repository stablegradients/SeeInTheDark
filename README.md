# SeeInTheDark
Course project for EE 610

Authors: R. Shrinivas, S. Sridhar

Dataset used: See in the dark dataset (C. Chen et. al.)
Dataset can be downloaded from https://github.com/cchen156/Learning-to-See-in-the-Dark

Run train.py to either train or validate/test the model.
To train: python train.py -m train
To validate/test: python train.py -m validate

To evaluate PSNR and SSIM of the resulting test images, run evaluate.py
python evaluate.py result.png ground_truth.png
