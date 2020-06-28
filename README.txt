Implement a simple GAN in Keras to generate MNIST images. Use the GAN given here

https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

as your discriminator and generator. 

You want to train the generator to produce images of numbers between 0 and 9.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the generator model to. 

python train.py MNIST_train_directory <generator model file>

Make test.py take one input: the generator model file. The output
of test.py should be images resembling MNIST digits saved to the output
file.

python test.py <generator model file> <output image filename>