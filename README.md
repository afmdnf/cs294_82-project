# Taylor-expansion Network Pruning for Transfer Learning

This work demonstrates Taylor-expansion pruning on a modified VGG16 network, used for a binary image classification task.
This was able to reduce the CPU runtime by 3x and model size by 5x.

## Usage

The code uses the PyTorch ImageFolder loader, so it assumes that the dataset images are in a different directory for each category. For example,

`
/train
	/dogs
	/cats
`
`
/test
	/dogs
	/cats
`
Test

For the experiment, the dataset was derived from from [here](https://www.kaggle.com/c/dogs-vs-cats) but the code should be able to handle any network and any dataset.


Training:
`python run.py --train`

Pruning:
`python run.py --prune`
