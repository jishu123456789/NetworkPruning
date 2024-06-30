# Network Pruning of Image Captioning Model

## Task
The primary task was to apply network Pruning Techniques to an existing Image Captioning Model so as to decrease the model size while maintaining quality accuracy at the same time

## Datasets
The dataset that was used was Flickr Dataset. It was initially divided into a training set and a hold-out test set. For inference the Batch Size was kept as 1 while for training it was set as 32.

## Model
The Model was made of a three stage process with each process pruning one component of the Image Captioning Model
The Process consisted of a 3 stage pipeline:

### Pruning The Pre-Trained ResNet Backbone:
- BatchNorm Variance was taken as the importance criteria for Pruning the ResNet Model.
- This is a measure of how much of an input affects the ouputs and hence it was taken as the criteria for imprtance.
- Torch Pruning Library was used for Pruning The ResNet.

### Pruning The Transformer Encoder:
- Width Pruning was applied to the transformers Encoder.
- A regularization term was added to the weights of the Encoder which pruned the weights based on their magnitude
- L2 Regularization was used for Pruning the Encoder Weights

### Pruning The Transformer Decoder:
- Depth Pruning was applied for the Transformers Decoder. This was done becuase it was found that the number of Decoder Layers affected the inference speed
- Uniform selection of the Decoder Layers was done and the outputs of the Pruned Model were compared with the corresponding unpruned one.

## Configurations
### Hyperparameters:
- Training Batch size : 32
- Testing Batch size : 1
- Learning Rate : 3e-4
- Epochs : 15
- Numb Encoder Layers : 3
- Num Decoder Layers : 4
- Num Attention Heads : 8
- Dropout Prob : 0.1
- Encoder_Regularization_Coeff : 0.01
- Decoder_weoght_mse_Contribution_Coeff : 0.01
### Results : 
- The model sees a major drop in performance when decoder pruning is considered.
- Encoder Pruning did not bring much degradation in model performance. This may be due to the fact that most of the encoder weights were necessary to encode important information regarding the image.






## DatasetLink 

###
- [https://drive.google.com/drive/folders/1vhu3LJ9eYDDQfmUZB62jFRf6PuqMbqr](https://www.kaggle.com/datasets/adityajn105/flickr8k)-
- This link contains the Flickr Dataset which contains the testing and training data
  
## Mentions

### Much of the codebase was inspired by
- Alaadin Pearson
- Image Captioning using CNN and Transformers
