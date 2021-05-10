import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from point_augment import Augmentor
from feature_extractor import PointNetClassifier
from model import ClassificationPointNet
from dataset import ModelNet10
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

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
	
	tsne = TSNE(n_components=2, verbose=1, perplexity= 50, n_iter=2000, learning_rate = 200)

	categories = ['chair','sofa','bed', 'desk','monitor','night_stand','table', 'toilet','bathtub', 'dresser']
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	workers =  0 if device == 'cuda' else 8
	pin_memory = device != 'cuda'
	batch_size = 10

	train_dataset = ModelNet10('../ModelNet10/ModelNet10', task = 'train', categories = categories)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last =True)
	classifier = PointNetClassifier(num_classes = len(categories)).to(device) 
	augmentor = Augmentor(F_layers = [32, 16], M_layers = [32, 16], T_layers = [32,16], N = 1024).to(device)

	classifier.load_state_dict(torch.load('classifier_correct_10_new.pth'))
	augmentor.load_state_dict(torch.load('augmentor_correct_10_new.pth'))

	batch_data = next(iter(train_loader))
	labels = batch_data[1].type('torch.FloatTensor').to(device)
	data = batch_data[0].type('torch.FloatTensor').to(device)
	outputs, feature_transform = classifier(data)
	pred_label = torch.argmax(outputs, dim=1)

	visualize_batch(data, pred_label, labels, categories)

	data_augmented, T , M = augmentor(data)
	data_augmented = data_augmented.transpose(1,2)
	outputs, feat = classifier(data_augmented)

	pred_label = torch.argmax(outputs, dim=1)
	visualize_batch(data_augmented.detach(), pred_label, labels, categories)

	exit()
	feat_array = np.empty((0,1024))
	feataug_array = np.empty((0,1024))
	label_array = np.empty((0,1))

	for (data, labels) in train_loader:
		data = data.type('torch.FloatTensor').to(device)
		_, feat = classifier(data)
		feat_array = np.vstack( (feat_array,feat.cpu().data.numpy()))
		data_augmented, T , M = augmentor(data)

		data_augmented = data_augmented.transpose(1,2)
		_, feat = classifier(data_augmented)
		feataug_array = np.vstack( (feataug_array,feat.cpu().data.numpy() ))
		label_array = np.append(label_array, labels)


	
	all_label = np.append(label_array, label_array + 10)
	all_label = label_array
	mask = np.array([k in np.array([0,1,2,3]) for k in all_label]) 

	all_feat = np.vstack((feat_array, feataug_array))
	all_feat = feat_array

	pca = PCA(n_components=50)
	X_pca = pca.fit_transform(all_feat) 

	tsne_results = tsne.fit_transform(X_pca)

	fig, ax = plt.subplots(figsize = (12,10))

	scatter = ax.scatter(tsne_results[:,0],tsne_results[:,1], c=all_label)

	# produce a legend with the unique colors from the scatter
	legend1 = ax.legend(*scatter.legend_elements(),
	                    loc="upper right", title="Classes")
	ax.add_artist(legend1)

	plt.show()
