from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

def angle_axis(angle: float, axis: np.ndarray):
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = cosval * np.eye(3) + sinval * cross_prod_mat + (1.0 - cosval) * np.outer(u, u)
  
    # yapf: enable
    return R

class ToTensor(object):
	def __init__(self):
		super(ToTensor, self).__init__()
	def __call__(self, arg):
		pc, label = arg
		pc = torch.tensor(pc).type('torch.FloatTensor')
		label = torch.tensor(label).type('torch.LongTensor')

		return (pc, label)

class RandomPerturbations(object):
	"""docstring for RandomPerturbations"""
	def __init__(self, arg):
		super(RandomPerturbations, self).__init__()
		assert isinstance(arg, tuple)
		self.mean, self.std = arg
	
	def __call__(self, sample):
		pc , label = sample
		pc_size = pc.shape

		noise = np.random.normal(self.mean, self.std, (pc_size))


		pc = pc + noise

		return (pc, label)

class ScaleAndTranslate(object):
	"""docstring for ScaleAndTranslate"""
	def __init__(self, arg):
		super(ScaleAndTranslate, self).__init__()
		assert isinstance(arg, tuple) 
		self.scale_high, self.translate_high = arg
		self.scale = np.random.uniform(2/3., self.scale_high)
		self.translate = np.random.uniform(-self.translate_high, self.translate_high)

	def __call__(self, sample):
		pc, label = sample

		pc = self.scale*pc + self.translate

		return (pc, label)

class RandomRotation(object):
	"""docstring for RandomRotation"""
	def __init__(self, arg):
		super(RandomRotation, self).__init__()
		assert isinstance(arg, tuple) 
		self.sig, self.clip = arg
		self.angles = np.clip(self.sig * np.random.randn(3), -self.clip, self.clip)
		Rx = angle_axis(self.angles[0], np.array([1.0, 0.0, 0.0]))
		Ry = angle_axis(self.angles[1], np.array([0.0, 1.0, 0.0]))
		Rz = angle_axis(self.angles[2], np.array([0.0, 0.0, 1.0]))
		

		self.rot = (Rz@Ry)@Rx
	def __call__(self, sample):
		pc, label = sample

		pc = (self.rot@pc.T).T

		return (pc,label)
		
		

