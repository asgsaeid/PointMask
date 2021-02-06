
import numpy as np
import h5py
import random
from keras.utils import np_utils
from pyquaternion import Quaternion as Q


class DataGenerator:
    def __init__(self, file_name, batch_size, nb_classes=40, train=True):
        self.fie_name = file_name
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.train = train

    @staticmethod
    def rotate_point_cloud(data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    @staticmethod
    def rotate_point_cloud_rnd(data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along random directions
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        quat = Q.random()
        quat = quat.rotation_matrix
        rotated_data = np.matmul(data, quat)
        return rotated_data

    @staticmethod
    def jitter_point_cloud(data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    @staticmethod
    def translate_point_cloud(data):
        tr = np.random.random((1, 3))
        tr = tr / np.linalg.norm(tr) * 0.5
        data[:, :3] += tr
        return data

    def generator(self):
        f = h5py.File(self.fie_name, mode='r')
        nb_sample = f['data'].shape[0]
        while True:
            index = [n for n in range(nb_sample)]
            random.shuffle(index)
            for i in range(nb_sample // self.batch_size):
                batch_start = i * self.batch_size
                batch_end = (i + 1) * self.batch_size
                batch_index = index[batch_start: batch_end]
                X = []
                Y = []
                for j in batch_index:
                    item = f['data'][j]
                    label = f['label'][j]
                    if self.train:
                        is_rotate = random.randint(0, 1)
                        is_jitter = random.randint(0, 1)
                        if is_rotate == 1:
                            item = self.rotate_point_cloud_rnd(item)
                        if is_jitter == 1:
                            item = self.jitter_point_cloud(item)
                    X.append(item)
                    Y.append(label)
                Y = np_utils.to_categorical(np.array(Y), self.nb_classes)
                yield np.array(X), Y

