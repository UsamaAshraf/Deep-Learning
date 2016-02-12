from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# ------------------------------- Training ---------------------------------------

_data = pickle.load(open("notMNIST.pickle", "rb"))

# Create linear regression object
regr = LogisticRegression()

_data['train_dataset'] = np.reshape(_data['train_dataset'], (50000, 28*28))
_data['test_dataset'] = np.reshape(_data['test_dataset'], (10000, 28*28))

# Train the model using the training sets
regr.fit(_data['train_dataset'], _data['train_labels'])

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
print("Residual sum of squares: %.2f" % np.mean((regr.predict(_data['test_dataset']) - _data['test_labels']) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(_data['test_dataset'], _data['test_labels']))

# Plot outputs
plt.scatter(regr.predict(_data['test_dataset']), _data['test_labels'], color='blue')
plt.plot(regr.predict(_data['test_dataset']), _data['test_labels'], color='red', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()

# ------------------------ Saving the model for later reuse -------------------------------

pickle_file = 'LogRegrnotMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {regr}
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)