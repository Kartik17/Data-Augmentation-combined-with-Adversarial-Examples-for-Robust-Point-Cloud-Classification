from torch.utils.data import DataLoader
from data_utils.imbalanced import ImbalancedDatasetSampler
import torch
from data_utils.da_utils import *
from torchvision import transforms

class DataLoaderClass():
    def __init__(self, data_config, train_config):

        self._dataset_name    = data_config['NAME']
        self._data_path       = data_config['DATA_PATH']
        self._categories      = data_config['CATEGORIES']
        self._use_saved_model = train_config['USE_SAVED_MODEL']
        self._batch_size      = train_config['BATCH_SIZE']
        self._num_workers     = train_config['WORKERS']
        self._drop_last       = train_config['DROP_LAST']
        
        self.train_transforms = transforms.Compose([RandomPerturbations((0.0, 0.01)),ScaleAndTranslate((1.2, 0.03)),RandomRotation((0.06,0.18)), ToTensor()])
        self.valid_transforms = transforms.Compose([ToTensor()])

        if(self._dataset_name == "ARGOVERSE"):
            from data_utils.dataset import ArgoverseDataset
            self._train_dataset = ArgoverseDataset(self._data_path, task = 'train', categories = self._categories)
            self._test_dataset  = ArgoverseDataset(self._data_path, task = 'test' , categories = self._categories)

        elif(self._dataset_name == "MODELNET40"):
            from data_utils.dataset import ModelNet40
            self._train_dataset = ModelNet40(self._data_path, task = 'train', categories = self._categories, transform = self.train_transforms)
            self._test_dataset  = ModelNet40(self._data_path, task = 'test' , categories = self._categories, transform = self.valid_transforms)

        elif(self._dataset_name == "MODELNET10"):
            from data_utils.dataset import ModelNetDataLoader, ModelNet10
            self._train_dataset = ModelNet10(self._data_path, task = 'train', categories = self._categories, transform = self.train_transforms)
            self._test_dataset  = ModelNet10(self._data_path, task = 'test' , categories = self._categories, transform = self.valid_transforms)
        
        self._trainloader = DataLoader(self._train_dataset, batch_size = self._batch_size, shuffle = True, num_workers = self._num_workers, drop_last= self._drop_last)
        self._validloader = DataLoader(self._test_dataset , batch_size = self._batch_size, shuffle = True, num_workers = self._num_workers, drop_last= self._drop_last)

    # Dataset_Name
    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        if  isinstance(dataset_name, str):
            self._dataset_name = dataset_name
        else:
            print("Enter valid string. ")

    # Learning_Rate
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, lr):
        if lr > 0:
            self._lr = lr
        else:
            print("Enter valid value of greater than zero. ")

    # Batch Size
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if batch_size > 0:
            self._batch_size= batch_size
        else:
            print("Enter valid value of greater than zero. ")

    @property
    def validloader(self):
    	return self._validloader

    @property
    def trainloader(self):
    	return self._trainloader

    @property
    def weights(self):
        return self._train_weights


