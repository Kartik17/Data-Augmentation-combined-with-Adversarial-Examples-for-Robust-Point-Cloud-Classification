import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import ModelNet10
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits import mplot3d
from torch.utils.data import DataLoader
from pointnet import PointNetClassifier
from pointnet_seg import PointNetSegmenter
from loss import ChamferDistLoss

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



def valid_model(model, point_gen, validloader):
	model.eval()
	total = 0
	accurate = 0
	with torch.no_grad():
		for batch_idx, batch_data in enumerate(validloader):
			labels = batch_data[1].type('torch.FloatTensor').to(device)
			data = batch_data[0].type('torch.FloatTensor').to(device)

			mask = point_gen(data)
			#perturbation = torch.clamp(perturbation, -1.5 , 1.5)
			#adv_point = data + perturbation*mask
			#adv_point = data*mask
			adv_point = mask

			outputs, features = model(adv_point)
			pred_label = torch.argmax(outputs, dim=1)
			
			accurate += np.sum(pred_label.cpu().numpy() == labels.cpu().numpy())
			total += len(pred_label.cpu().numpy())
			#visualize_batch(data, pred_label, labels, categories)
		
	accuracy  = accurate/total

	return accuracy

categories = ['chair', 'sofa','bed', 'desk','monitor','night_stand','table', 'toilet','bathtub', 'dresser']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_net = PointNetClassifier(num_classes = len(categories)).to(device)


test_dataset = ModelNet10('../ModelNet10/ModelNet10', task = 'test', categories = categories)
validloader = DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers = 8, drop_last=True )


model_dataset = ModelNet10('../ModelNet10/ModelNet10', task = 'train', categories = categories)
trainloader = DataLoader(model_dataset, batch_size = 10, shuffle = True, num_workers = 8, drop_last=True )


class_net.load_state_dict(torch.load("./clean_model10.pth"))
class_net.eval()

point_gen = PointNetSegmenter(num_classes = 3).to(device)
point_dis = PointNetClassifier(num_classes = 1).to(device)
#point_gen = PointNetSegmenter(in_channels=3, feat_size = 1024, feat_layer_dims=[32, 64], classifier_layer_dims=[200, 100]).to(device)
criterion = torch.nn.BCELoss()
optimizerG = optim.Adam(point_gen.parameters(), lr = 0.001)
optimizerD = optim.Adam(point_dis.parameters(), lr = 0.01)
lambda1 = 0.5
lambda2 = 50.0
Chamfer_Loss_Func = ChamferDistLoss()

real_label = 1.
fake_label = 0.

for epoch in range(30):
	for idx, batch_data in enumerate(trainloader):
		labels = batch_data[1].type('torch.FloatTensor').to(device)
		data = batch_data[0].type('torch.FloatTensor').to(device)

		# Discriminator
		optimizerD.zero_grad()
		output, real_feature = point_dis(data)
		labels_gan = torch.full((data.shape[0],), real_label, dtype=torch.float, device=device)
		errD_real = criterion(output.view(-1), labels_gan)
		errD_real.backward()
		

		optimizerG.zero_grad()
		perturbation, _ = point_gen(data)
		#perturbation = torch.clamp(perturbation, -1.0 , 1.0)
		adv_pc = perturbation + data
		real, _ = point_dis(adv_pc.detach())
		labels_gan.fill_(fake_label)
		errD_fake = criterion(real.view(-1), labels_gan)
		errD_fake.backward()

		errD = errD_fake + errD_real
		optimizerD.step()

		# Generator
		optimizerG.zero_grad()
		labels_gan.fill_(real_label)
		output, adv_feature = point_dis(adv_pc) 

		output_net, _ = class_net(adv_pc)
		errG = criterion(output.view(-1), labels_gan) + torch.norm(real_feature.detach() - adv_feature, 2) - F.cross_entropy(output_net, labels.long()) + (1/100.0)*torch.abs(torch.norm(perturbation,1))
		
		#adv_loss.backward()
		errG.backward()
		optimizerG.step()


		
	print("Loss(CE, l1, Chamfer) per Epoch - {}: {}, {}, {}".format(epoch, errD_real.item(), errD_fake.item(), errG.item()))

torch.save(point_gen.state_dict(), "./point_gen.pth")


#point_gen.load_state_dict(torch.load("./point_gen.pth"))
with torch.no_grad():
	batch_data = next(iter(validloader))
	labels = batch_data[1].type('torch.FloatTensor').to(device)
	data = batch_data[0].type('torch.FloatTensor').to(device)
	perturbation, mask = point_gen(data)
	#perturbation = torch.clamp(perturbation, -1.5 , 1.5)
	adv_point = data + perturbation*mask
	#adv_point = data*mask
	
	outputs, feature_transform = class_net(adv_point)
	pred_label = torch.argmax(outputs, dim=1)
	
	visualize_batch(adv_point, pred_label, labels, categories)
	visualize_batch(perturbation, pred_label, labels, categories)
	print("Std: {}".format(torch.std(perturbation, dim=0)))
	print("Mean: {}".format(torch.mean(perturbation, dim=0)))