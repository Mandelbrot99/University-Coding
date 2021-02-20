from torchvision import datasets, transforms


import random
import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28

        self.my_data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.data])
        #self.my_data = torch.stack([img.float() for img in self.data])
        
        


    def __getitem__(self, index):

        return self.my_data[index], self.targets[index]

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.my_data[sample_idx]]
    
    def sample_pairs(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.my_data[sample_idx]], [label for label in self.targets[sample_idx]]

    
def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


