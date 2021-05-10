import torch
from training import PointNetTrain, PointAugmentTrain, Model
#from PointAugment.Augment.config import opts
from data_utils.dataloader import DataLoaderClass
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import yaml

def visualize_batch(pointclouds, pred_labels, labels, categories):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(8, batch_size / 2))

    ncols = 5
    nrows = max(1, batch_size // 5)
    for idx, pc in enumerate(pointclouds):
        label = categories[int(labels[idx].item())]
        pred = categories[int(pred_labels[idx])]
        colour = 'g' if label == pred else 'r'
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colour, s=2)
        ax.axis('off')
        ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.show()


if __name__ == '__main__':
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PointNet
    training_instance_2 = PointNetTrain(config['MODEL']['POINTNET'],  device)
    modelnet10_dataloader = DataLoaderClass(config['DATA']['MODELNET10'], config['MODEL']['POINTNET']['TRAINING'])    
    #training_instance_2.train(modelnet10_dataloader.trainloader, modelnet10_dataloader.validloader, adv = False)
    training_instance_2.test(modelnet10_dataloader.validloader)

    # Point Augment
    #training_instance_1 = PointAugmentTrain(config['MODEL']['POINT_AUGMENT'],  device)
    #modelnet10_dataloader = DataLoaderClass(config['DATA']['MODELNET10'], config['MODEL']['POINTNET']['TRAINING'])    
    #training_instance_1.train(modelnet10_dataloader.trainloader, modelnet10_dataloader.validloader, adv = False)
    #training_instance_1.test(modelnet10_dataloader.validloader)


   

