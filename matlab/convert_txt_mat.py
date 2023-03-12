import numpy as np
import scipy.io

d = np.loadtxt("trainData.txt",delimiter=',',comments='#')
scipy.io.savemat('trainData.mat', dict(data=d))

d = np.loadtxt("testData.txt",delimiter=',',comments='#')
scipy.io.savemat('testData.mat', dict(data=d))

