# RibonanzaNet

Training code for RibonanzaNet

# Data Download

# Environment

# Config 

# README for Configuration File

This README document explains the various parameters and settings in the configuration file for a machine learning model. Understanding each parameter is crucial for effectively training and optimizing the model.

## Model Hyperparameters
- `learning_rate`: 0.001  
  The learning rate for the optimizer. Determines the step size at each iteration while moving toward a minimum of the loss function.

- `batch_size`: 2  
  Number of samples processed before the model is updated.

- `test_batch_size`: 8  
  Batch size used for testing the model.

- `epochs`: 40  
  Total number of training cycles the model goes through.

- `optimizer`: "ranger"  
  The optimization algorithm used for minimizing the loss function. Ranger is a synergistic optimizer combining RAdam and LookAhead.

- `dropout`: 0.05  
  The dropout rate for regularization to prevent overfitting. It represents the proportion of neurons that are randomly dropped out of the neural network during training.

- `weight_decay`: 0.0001  
  Regularization technique to prevent overfitting by penalizing large weights.

- `k`: 9  
  A specific hyperparameter relevant to the model, requiring further context for detailed explanation.

- `ninp`: 256  
  The size of the input dimension.

- `nlayers`: 9  
  Number of layers in the neural network.

- `nclass`: 2  
  Number of classes for classification tasks.

- `ntoken`: 5  
  Number of tokens (AUGC + padding/N token) used in the model.

- `nhead`: 8  
  The number of heads in multi-head attention models.

- `use_bpp`: False  
  A boolean parameter that specifies whether to use base pair probability (BPP) in the model.

- `use_flip_aug`: true  
  Indicates whether flip augmentation is used during training.

- `bpp_file_folder`: "../../input/bpp_files/"  
  The directory where base pair probability files are stored.

- `gradient_accumulation_steps`: 2  
  Number of steps to accumulate gradients before performing a backward/update pass.

- `use_triangular_attention`: false  
  Specifies whether to use triangular attention mechanisms in the model.

- `pairwise_dimension`: 64  
  Dimension of pairwise interactions in the model.

## Data Scaling
- `use_data_percentage`: 1  
  The percentage of data used from the dataset.

- `use_dirty_data`: true  
  Indicates whether to include 'dirty' or noisy data in the training process. Useful for testing robustness or when clean data is limited.

## Other Configurations
- `fold`: 0  
  The current fold in use if the data is split into folds for cross-validation.

- `nfolds`: 6  
  Total number of folds for cross-validation.

- `input_dir`: "../../input/"  
  Directory for input data.

- `gpu_id`: "0"  
  Identifier for the GPU used for training. Useful in multi-GPU setups.

---

Remember, the effectiveness of these parameters can vary depending on the specific nature of the dataset and the model architecture. Experimentation and adjustment might be necessary to achieve optimal results.

