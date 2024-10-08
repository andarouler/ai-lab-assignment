# Task Description

The uploaded zip file contains generated images of cities in rainy/sunny weather and realistic/artistic style.
Your task is to train _two_ image classifiers:
1. A model that predicts whether it is rainy/sunny.
2. A model that predicts realistic/artistic.

You need to use Python.
You can use any deep learning library you like (Pytorch is preferred).
We want to see a deep learning-based solution, i.e., no SVM etc.

You can use Google Colab to have access to a GPU.

# Trained Models

## Description of the SimpleCNN

The SimpleCNN model is a lightweight convolutional neural network for image classification tasks. It consists of two convolutional layers, followed by fully connected layers. 


## Measures against overfitting

Given the small size of the dataset, several steps were taken to prevent/reduce overfitting:

    - Reduced Model Complexity: The number of filters in the convolutional layers was reduced to limit the model's capacity and avoid memorizing the training data.

    - Dropout Layer: A dropout layer with a 50% dropout rate was introduced to prevent the model from relying too heavily on specific neurons during training, encouraging more generalizable feature learning.

    - Data Augmentation: To artificially increase the dataset's diversity, several augmentation techniques were applied, including random resizing, horizontal flipping, rotation, and adjustments in brightness, contrast, and saturation. This helped the model handle various variations in the input images.

Additional Measures

    - Early Stopping: This technique can be implemented to stop training when the model's performance on the validation set stops improving, preventing unnecessary overfitting.

    - Reduced Number of Epochs: Limiting the training epochs ensures the model doesn't overtrain on the limited data


## Training

### rainy_sunny_classifier.pth

```
Epoch 1/9, Train Loss: 0.6165373206138611, Test Loss: 0.577990934252739, Test Accuracy: 0.8
Epoch 2/9, Train Loss: 0.48040019869804385, Test Loss: 0.5908246785402298, Test Accuracy: 0.775
Epoch 3/9, Train Loss: 0.4461625576019287, Test Loss: 0.36448289453983307, Test Accuracy: 0.75
Epoch 4/9, Train Loss: 0.452396959066391, Test Loss: 0.2633468359708786, Test Accuracy: 0.95
Epoch 5/9, Train Loss: 0.38922087848186493, Test Loss: 0.20170123130083084, Test Accuracy: 0.975
Epoch 6/9, Train Loss: 0.3134173691272736, Test Loss: 0.21496391016989946, Test Accuracy: 0.85
Epoch 7/9, Train Loss: 0.30113910138607025, Test Loss: 0.14147231727838516, Test Accuracy: 0.975
Epoch 8/9, Train Loss: 0.22021830677986146, Test Loss: 0.11375889740884304, Test Accuracy: 0.925
Epoch 9/9, Train Loss: 0.2083535984158516, Test Loss: 0.06586712831631303, Test Accuracy: 1.0
```

Comment: This model could be overfitting, as the test accuracy is high, but the small amount of training data might not be enough for the model to generalize well.


### artistic_realistic_classifier.pth

```
Class labels for artstyle: ['artistic', 'realistic']
Epoch 1/9, Train Loss: 1.0520722031593324, Test Loss: 0.6583076417446136, Test Accuracy: 0.5
Epoch 2/9, Train Loss: 0.750455892086029, Test Loss: 0.6833955347537994, Test Accuracy: 0.5
Epoch 3/9, Train Loss: 0.6871476769447327, Test Loss: 0.6521898210048676, Test Accuracy: 0.5
Epoch 4/9, Train Loss: 0.6735546469688416, Test Loss: 0.6466858685016632, Test Accuracy: 0.725
Epoch 5/9, Train Loss: 0.6401363611221313, Test Loss: 0.6612980365753174, Test Accuracy: 0.8
Epoch 6/9, Train Loss: 0.588061785697937, Test Loss: 0.5373858213424683, Test Accuracy: 0.775
Epoch 7/9, Train Loss: 0.5035816490650177, Test Loss: 0.5600416511297226, Test Accuracy: 0.725
Epoch 8/9, Train Loss: 0.45401055812835694, Test Loss: 0.4919136315584183, Test Accuracy: 0.875
Epoch 9/9, Train Loss: 0.4626552015542984, Test Loss: 0.37222427129745483, Test Accuracy: 0.9
```

Comment: This model shows steady improvement throughout the training process. Although the accuracy starts at 50%, it gradually increases to 90% by the final epoch. 