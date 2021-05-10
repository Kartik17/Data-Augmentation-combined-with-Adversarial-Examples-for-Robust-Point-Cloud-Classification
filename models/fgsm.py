import torch
#import numpy as np
#import torch.nn as nn
#import torch.optim as optim
#from dataset import ModelNet10
import matplotlib.pyplot as plt
#import torch.nn.functional as F
from mpl_toolkits import mplot3d
#from torch.utils.data import DataLoader
#


def visualize_batch(pointclouds, pred_labels, labels, categories):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(8, batch_size / 2))

    ncols = 5
    nrows = max(1, batch_size // 5)
    for idx, pc in enumerate(pointclouds):
        label = categories[int(labels[idx].item())]
        pred = categories[int(pred_labels[idx])]
        colour = 'g' if label == pred else 'r'
        pc = pc.cpu().detach().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colour, s=2)
        ax.axis('off')
        ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.show()

def fgsm_attack(model, criterion, point, labels, eps) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point = point.to(device)
    labels = labels.to(device)
    point.requires_grad = True

    outputs, _ = model(point)
    model.zero_grad()
    cost = criterion(outputs, labels.long()).to(device)
    
    cost.backward()
    
    attack_data = point + eps*point.grad.sign()
    #attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_data

def valid_model(model, validloader, eps):
	model.eval()
	total = 0
	accurate = 0
	
	for batch_idx, batch_data in enumerate(validloader):
		labels = batch_data[1].type('torch.FloatTensor').to(device)
		data = batch_data[0].type('torch.FloatTensor').to(device)
		adv_point   = fgsm_attack(model,  data, labels, eps)
		outputs, _  = model(adv_point)
		pred_label = torch.argmax(outputs, dim = 1)
		#outputs, feature_transform = model(data)
		#pred_label = torch.argmax(outputs, dim=1)
		accurate += np.sum(pred_label.cpu().numpy() == labels.cpu().numpy())
		total += len(pred_label.cpu().numpy())

		#visualize_batch(data, pred_label, labels, categories)	
	accuracy  = accurate/total

	return accuracy
if __name__ == '__main__':
        
    categories = ['chair', 'sofa','bed', 'desk','monitor','night_stand','table', 'toilet','bathtub', 'dresser']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_net = PointNetClassifier(num_classes = len(categories)).to(device)
    
    test_dataset = ModelNet10('../ModelNet10/ModelNet10', task = 'test', categories = categories)
    validloader = DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers = 8, drop_last=True )
    class_net.load_state_dict(torch.load("./saved_model.pth"))
    class_net.eval()
    
    batch_data = next(iter(validloader))
    labels = batch_data[1].type('torch.FloatTensor').to(device)
    data = batch_data[0].type('torch.FloatTensor').to(device)
    loss = nn.CrossEntropyLoss()
    
    #eps = 1.0
    #adv_point   = fgsm_attack(class_net, data, labels, eps)
    #outputs, _  = class_net(adv_point)
    #pred_labels = torch.argmax(outputs, dim = 1)
    #visualize_batch(adv_point, pred_labels, labels, categories)
    
    for eps in [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5]:
    	print("EPS:{}, Accuracy:{}".format(eps, valid_model(class_net,validloader, eps)))  