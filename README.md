# Wine Origin Classifier

A neural network built with **PyTorch** that predicts which of three grape cultivars a wine
comes from, using 13 results of its chemical analysis (alcohol, malic acid, flavanoids,
colour intensity and more).

**Result: 97.2% accuracy on wines the model has never seen.**

## Dataset

The [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), included in
scikit-learn, so nothing has to be downloaded:

- 178 wines grown in the same region of Italy
- 13 numeric features per wine
- 3 classes: the cultivar (grape variety) the wine was made from

## How it works

1. **Load and split**: 80% of the wines for training, 20% for testing. The split is
   stratified, so every cultivar appears in both sets in the same proportion.
2. **Scale**: `StandardScaler` puts all features on the same scale. It is fitted on the
   training data only, so no information leaks from the test set.
3. **Batch**: the data becomes PyTorch tensors and is fed in batches of 32 with `DataLoader`.
4. **Model**: a fully connected network:

   ```
   13 features → Linear(13, 30) → ReLU → Linear(30, 15) → ReLU → Linear(15, 3) → class scores
   ```

5. **Train**: 50 epochs with the Adam optimizer (learning rate 0.01) and cross-entropy loss.
6. **Evaluate**: the class with the highest score is the prediction; the script prints the
   accuracy and a per-class report.

## Results

```
========================================
=== TEST ACCURACY: 97.22% ===
========================================

Classification Report:
              precision    recall  f1-score   support

  Cultivar 0       1.00      1.00      1.00        12
  Cultivar 1       0.93      1.00      0.97        14
  Cultivar 2       1.00      0.90      0.95        10

    accuracy                           0.97        36
```

35 of the 36 test wines are classified correctly. The random seed is fixed, so you
get the same result on every run.

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/Cuman000Q/Wine-Origin-Classifier.git
cd Wine-Origin-Classifier
pip install -r requirements.txt
python wine_classifier.py
```

It trains in a few seconds on a normal CPU, no GPU needed.

## Built with

- [PyTorch](https://pytorch.org/): the neural network and training loop
- [scikit-learn](https://scikit-learn.org/): dataset, train/test split, scaling and metrics

## Ideas for next steps

- [ ] Compare with a simple baseline (logistic regression, random forest)
- [ ] Use k-fold cross-validation, since the test set has only 36 wines
- [ ] Plot the training loss and a confusion matrix

## Author

Made by [Cuman000Q](https://github.com/Cuman000Q). See also the
[California Housing Price Predictor](https://github.com/Cuman000Q/California-House-Price-Predictor),
the regression counterpart of this project.
