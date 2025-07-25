# Image-classifier
Implement a neural network that classifies images on the CIFAR-10 dataset, which is composed of 60000 small (3 × 32 × 32) color images, each of which belongs to one of 10 classes. There are 6000 images per class. The images are divided into a training dataset composed of 50000 examples and a testing dataset composed of 10000 examples. This dataset is readily available for PyTorch.

<img width="495" height="500" alt="image" src="https://github.com/user-attachments/assets/ca1a8348-ffa8-42db-8007-5c028f4b32a9" />

Figure 1: Examples from CIFAR-10 (classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

#Neural Network Architecture
The baseline model consists of independent convolutional layers followed by fully connected layers. 

#Intermediate Blocks
The network consists of three Intermediate Blocks, each containing a progressively increased number of convolutional layers to capture complex features efficiently:
Block 1: 4 convolutional layers (input: 3 channels, output: 64 channels)
Block 2: 5 convolutional layers (input: 64 channels, output: 128 channels)
Block 3: 6 convolutional layers (input: 128 channels, output: 256 channels)
Each convolutional layer within the blocks is followed by Batch Normalisation and ReLU activation for stabilizing and accelerating training. Dropout layers (probability = 0.3) are added for regularization.

#Transition Layers
Max Pooling layers are applied after each intermediate block to reduce dimensionality, control model complexity, and enhance computational efficiency.

#Output Block
The output block consists of two fully connected layers. The first fully connected layer reduces feature dimension from 256 to 512 units, followed by a Dropout layer, and the final layer outputs logits corresponding to 10 CIFAR-10 classes.

#Training Techniques and Hyperparameters
The following training techniques and hyperparameters were employed to optimize the performance of the model:
•	Loss Function: Cross Entropy Loss
•	Optimizer: Adam (learning rate: 0.001, weight decay: 4e-4 )
•	Learning Rate Scheduler: Cosine Annealing Scheduler (T_max = 50 epochs)
•	Data Augmentation Techniques:
	Random Crop with padding
	Random Horizontal Flip
	Random Rotation (±10 degrees)
	Color Jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)


