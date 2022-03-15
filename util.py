import sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp

plt.style.use('ieee')

# -----------------------------------------------------------------------------

def pce_prediction(surrogate, samples, test_data, out_index, out_label):

	n = np.shape(samples)[1]
	qoi_true = np.zeros((n))
	qoi_pred = np.zeros((n))
	for i in range(n):
		qoi_pred[i] = cp.numpoly.call(surrogate, samples[:,i])
		qoi_true[i] = test_data[i,out_index]
	
	a,b = qoi_true.min(), qoi_true.max()
	xx = np.linspace(a,b,1000)
	yy = xx.copy()
	
	plt.figure()
	plt.plot(xx,yy,'k-')
	plt.plot(qoi_true, qoi_pred, 'o')
	plt.xlabel('true')
	plt.ylabel('predicted')
	plt.title(out_label)
	plt.tight_layout()
	plt.savefig('fig_pred_%s.png' % out_label)

	corr_matrix = np.corrcoef(qoi_true, qoi_pred)
	corr = corr_matrix[0,1]
	# R2 coefficient of determination
	rsq = corr**2
	return rsq

# -----------------------------------------------------------------------------

def plot_qoi(dist_qoi, outf, x_label):
	plt.figure()
	values = dist_qoi.sample(10000).round(6)
	plt.hist(values, 50, density=True)
	plt.xlabel(x_label)
	plt.ylabel('density [-]')
	plt.tight_layout()
	plt.savefig('fig_' + outf + '.pdf')
	plt.savefig('fig_' + outf + '.png', dpi=300)
	np.savetxt('data_' + outf + '.txt', values)

# -----------------------------------------------------------------------------

def scatter_inputs(samples,labels_samples):
	n = np.shape(samples)[0]
	s = 3
	
	plt.figure()
	fig ,axs = plt.subplots(n,n,figsize=(n*s,n*s))
	for j in range(n):
		for i in range(n):
			if(j>=i):
				axs[i,j].axis('off')
			else:
				axs[i,j].scatter(samples[j,:],samples[i,:])#,marker = 'o')
				axs[i,j].set_xlabel(labels_samples[j])
				axs[i,j].set_ylabel(labels_samples[i])
	plt.tight_layout()
	plt.savefig('fig_scatter_inputs.png')
	plt.savefig('fig_scatter_inputs.svg')
	plt.savefig('fig_scatter_inputs.pdf')

# end scatter_inputs

# -----------------------------------------------------------------------------

def scatter_inputs_outputs(samples,evals,labels_samples,labels_evals):
	ni = np.shape(samples)[0] # number of inputs
	no = np.shape(evals)[0] # number of outputs
	s = 4
	plt.figure()
	fig ,axs = plt.subplots(no,ni,figsize=(no*s,no*s))
	for i in range(no):
		for j in range(ni):
				axs[i,j].scatter(samples[j,:], evals[i,:], marker = 'o')
				axs[i,j].set_xlabel( labels_samples[j] )
				axs[i,j].set_ylabel( labels_evals[i] )
	plt.tight_layout()
	plt.savefig('fig_scatter_inputs_outputs.png')
	plt.savefig('fig_scatter_inputs_outputs.svg')
	plt.savefig('fig_scatter_inputs_outputs.pdf')

	return

# end scatter_inputs_outputs

# -----------------------------------------------------------------------------
