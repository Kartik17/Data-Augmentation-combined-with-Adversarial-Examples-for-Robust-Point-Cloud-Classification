#import open3d as o3d
#import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import trimesh
import os
import glob

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class MNISTDataset(Dataset):
	"""docstring for MNISTDataset"""
	def __init__(self, root_dir, num_points = 1024, task = 'train'):
		super(MNISTDataset, self).__init__()
		self.root_dir = root_dir
		self.data = []
		self.num_points = num_points
		with h5py.File(self.root_dir, 'r') as hf:
			for key in hf.keys():
				self.data.append((np.array(hf[key]['points']), hf[key].attrs['label']))
		 
	def __getitem__(self, idx):
			idx = idx%4999
			pc  = self.data[idx][0]

			#if(pc.shape[0] < self.num_points):
			#	pc = np.vstack((pc, pc[:(self.num_points - pc.shape[0]),:]))
			label = self.data[idx][1]
			return (pc, label)

	def __len__(self):
		return len(self.data)


class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

class ModelNet10(Dataset):
	def __init__(self, root_dir, task = 'train', 
		categories = ['chair', 'sofa','bed','bathtub','desk','dresser','monitor','night_stand','table','toilet'], transform=None):
		super(ModelNet10, self).__init__()
		
		self.root_dir = pathlib.Path(root_dir)
		self.num_points = 1024
		self.max_examples = 500
		self.data = []
		self.id_to_label = {}
		self.total_data = 0
		self.count = {}
		self.transform = transform

		for idx, category in enumerate(categories):
			self.count[idx] = 0
			path = self.root_dir / category / task
			self.id_to_label[idx] = category
			
			for mesh_path in os.listdir(path):
				if(task == 'train' and self.count[idx] > self.max_examples):
					break					
				if mesh_path.endswith('.npy'):
					mesh_path = path / mesh_path
					points = np.load(mesh_path)
					points = points - np.mean(points, axis = 0)
				#elif mesh_path.endswith('.off'):
				#	arr_name = str(mesh_path).split('.')[0] + ".npy"
				#	mesh_path = path / mesh_path
				#	try:
				#		mesh = trimesh.load(mesh_path)	
				#	except Exception as e:
				#		raise e
				#	points = mesh.sample(self.num_points)
				#	#print("Saving File: {}".format(arr_name))
				#	np.save(path / arr_name, np.array(points))
					label = idx
					self.count[idx] += 1
					self.total_data = self.total_data + 1 
					self.data.append((points, label))

		self.data = np.array(self.data)
		np.random.shuffle(self.data)
		print(self.count)
	def __getitem__(self, idx):
		idx = idx%self.total_data

		pc  = np.array(self.data[idx][0])
		label = self.data[idx][1]

		sample = (pc, label)
		if self.transform:
			sample = self.transform(sample)
		return sample

	def __len__(self):
		return len(self.data) 

class ModelNet40(Dataset):
	def __init__(self, root_dir, task = 'train', 
		categories = ['car', 'desk', 'bookshelf', 'chair', 'night_stand', 'flower_pot', 'tent', 'bathtub', 'range_hood', 
					  'toilet', 'piano', 'bowl', 'guitar', 'person', 'monitor', 'sofa', 'cup', 'table', 'curtain', 'wardrobe', 
					  'laptop', 'plant', 'vase', 'radio', 'tv_stand', 'glass_box', 'lamp', 'airplane', 'cone', 
					  'bed', 'bottle', 'xbox', 'sink', 'mantel', 'dresser', 'keyboard', 'bench', 'stairs', 'door', 'stool'], transform = None):
		
		super(ModelNet40, self).__init__()
		self.root_dir = pathlib.Path(root_dir)
		self.num_points = 1024
		self.data = []
		self.id_to_label = {}
		self.total_data = 0
		self.count = {}
		self.max_examples = 500
		self.transform = transform

		for idx, category in enumerate(categories):
			self.count[idx] = 0
			path = self.root_dir / category / task
			self.id_to_label[idx] = category
			
			for mesh_path in os.listdir(path):
				if(task == 'train' and self.count[idx] > self.max_examples):
					break
				if mesh_path.endswith('.npy'):
					mesh_path = path / mesh_path
					points = np.load(mesh_path)
					points = points - np.mean(points, axis = 0)
				#elif mesh_path.endswith('.off'):
				#	arr_name = str(mesh_path).split('.')[0] + ".npy"
				#	mesh_path = path / mesh_path
				#	try:
				#		mesh = trimesh.load(mesh_path)	
				#	except Exception as e:
				#		raise e
				#	points = mesh.sample(self.num_points)
				#	points = points - np.mean(points, axis = 0)
				#	print("Saving File: {}".format(arr_name + ".npy"))
				#	np.save(path / arr_name, np.array(points))

					label = idx
					self.count[idx] += 1
	
					self.total_data = self.total_data + 1 
					self.data.append((points, label))
		print(self.count)
	def __getitem__(self, idx):
		idx = idx%self.total_data

		pc  = np.array(self.data[idx][0])
		label = self.data[idx][1]

		sample = (pc, label)
		if self.transform:
			sample = self.transform(sample)
		return sample

	def __len__(self):
		return len(self.data) 


class ArgoverseDataset(Dataset):
	def __init__(self, root_dir, task = 'train', 
		categories = ['VEHICLE', 'PEDESTRIAN', 'BUS', 'LARGE_VEHICLE', 'TRAILER']):
		super(ArgoverseDataset, self).__init__()
		
		self.root_dir = root_dir
		self.num_points = 1024
		self.max_examples = 300
		if(task == 'train'):
			self.max_examples = 500
		self.data = []
		self.id_to_label = {}
		self.total_data = 0

		for idx, category in enumerate(categories):
			path = self.root_dir + "/" + task + "/" + category 
			self.id_to_label[idx] = category
			label_cnt = 0

			for pcd_path in os.listdir(path):
				if(label_cnt >= self.max_examples):
					break					

				if pcd_path.endswith('.pcd'):
					pcd_path = path + "/" + pcd_path
					try:
						pcd_obj = o3d.io.read_point_cloud(pcd_path)
						points  = np.asarray(pcd_obj.points)	
						if(points.shape[0] <= 120):
							continue
					except Exception as e:
						raise e

					while(points.shape[0] < self.num_points):
						points = np.vstack((points,points))

					points = points[:self.num_points,:]
					label = idx	
					label_cnt = label_cnt + 1
					self.total_data = self.total_data + 1 
					self.data.append((points, label))

			print("Category: {}, Count: {}".format(category, label_cnt))

	def __getitem__(self, idx):
		idx = idx%self.total_data
		pc  = np.array(self.data[idx][0])
		
		label = self.data[idx][1]
		return (pc, label)

	def __len__(self):
		return len(self.data) 

