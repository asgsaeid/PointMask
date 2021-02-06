# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib
matplotlib.use('TkAgg')
import h5py
from data_loader import *
from model_cls import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
from sklearn.metrics import accuracy_score


f = h5py.File('./ModelNet40/ply_data_test.h5', mode='r')
probs = []
y_pred = np.zeros(len(f['data']))
labels = f['label']

model = point_mask(40)
model.load_weights('PATH/results/pointmask.h5')


for i in range(0, len(f['data'])):
    print(str(i) + ' / ' + str(len(f['data'])))
    ps = f['data'][i]
    l_ps = f['label'][i]

    model_output = model.predict(np.expand_dims(ps, 0))
    y_pred[i] = np.argmax(model_output, axis=1)


print(accuracy_score(labels, y_pred))

