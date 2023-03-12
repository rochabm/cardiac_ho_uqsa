import sys
import numpy as np
import chaospy as cp
from math import factorial

def shift_samples(s, ss, k):
	j = 0
	for i in range(len(s)):
		if i != k:
			ss[j] = s[i]
			j += 1

def calcula_loo(y, poly_exp, samples, distr):
	print ("\n\nPerforming Leave One Out cross validation.")
	looacc = np.zeros(len(y))

	nsamp = np.shape(y)[0]
	print(nsamp)
	print(y)

	deltas = np.empty(nsamp)
	samps = samples.T

	for i in range(nsamp):
		indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
		#print(indices)
		indices = np.delete(indices,i)
		#print(indices)

		#subs_samples = np.concatenate( (samps[:i,:], samps[i+1:,:]) , axis=0)
		subs_samples = samps[indices, :].copy()
		#subs_y = np.concatenate( (y[:i], y[i+1]) )
		subs_y = y[indices].copy()

		#subs_poly = cp.fit_regression(poly_exp, subs_samples, y_loo[k][0])
		subs_poly = cp.fit_regression(poly_exp, subs_samples.T, subs_y)
		yhat = cp.call(subs_poly, samps[i,:])
		#y_loo[k][1][i] = cp.Std(subs_poly, distr)			
		#y_loo[k][2][i] = abs(y[k] - def_poly)
		deltas[i] = abs(y[i] - yhat)

	#k = int(k)
	y_std = np.std(y)
	print("deltas",deltas)
	print("var", y_std)
	acc = 1.0 - np.mean(deltas)/np.mean(y_std)
	print(acc)
	return acc
# end_of_loo_cv