from torch.utils.data import Dataset


class TensorIntDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self, x, y):
        # [x: numpy array, y: list of int]

        # convert into Pytorch.torch.tensor
        self.data = x

        # should be list of int
        self.target = y

        self.dim = self.data.shape[0]

        print('Finished reading TensorInt Dataset ({} samples found)'
              .format(len(self.data)))

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
