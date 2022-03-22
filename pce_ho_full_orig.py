import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from loo import calcula_loo
from util import *

plt.style.use(['science','no-latex'])

if __name__ == "__main__":

	# parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', type=int, default=2, help="PCE polynomial degree")
	parser.add_argument('-m', type=int, default=2, help="PCE multiplicative factor")
	parser.add_argument('-uq', dest='uq', action='store_true', help='perform UQ ')
	parser.add_argument('-sa', dest='sa', action='store_true', help='perform SA (Sobol indices)')
	parser.add_argument('-qoi', dest='qoi', action='store_true',help='obtain QoIs dist')
	parser.add_argument('-all', dest='all', action='store_true', help='perform all tasks (uq/sa/qoi/test)')
	parser.add_argument('-test', dest='test', action='store_true',help='check test data and prediction accuracy')
	args = parser.parse_args()

	if(args.all):
		args.uq = True
		args.sa = True
		args.qoi = True
		args.test = True

	# Holzapfel-Ogden reference values (a0 = 150, b0 = 6.0 -> EF = 56)
	a0 = 150 #228.0	 		# Pa
	b0 = 6.0 #7.780			# dimensionless
	af0 = 116.85	 		# Pa
	bf0 = 11.83425			# dimensionless
	as0 = 372
	bs0 = 5.16
	afs0 = 410
	bfs0 = 11.3

	# Holzapfel-Ogden original parametrization
	a   = cp.Uniform(0.7*a0,   1.3*a0)
	b   = cp.Uniform(0.7*b0,   1.3*b0)
	af  = cp.Uniform(0.7*af0,  1.3*af0)
	bf  = cp.Uniform(0.7*bf0,  1.3*bf0)
	as1 = cp.Uniform(0.7*as0,  1.3*as0)
	bs  = cp.Uniform(0.7*bs0,  1.3*bs0)
	afs = cp.Uniform(0.7*afs0, 1.3*afs0)
	bfs = cp.Uniform(0.7*bfs0, 1.3*bfs0)
	distribution = cp.J(a, b, af, bf, as1, bs, afs, bfs)

	npar = 8            # number of parameters
	nout = 6            # (alfa1,beta1,alfa2,beta2,vol,def)
	pce_degree = args.d # polinomial degree
	pce_mult = args.m   # multiplicative factor
	Np = int(factorial(npar+pce_degree)/(factorial(npar)*factorial(pce_degree)))
	Ns = pce_mult * Np
	print("numero de parametros de entrada", npar)
	print("numero de saidas", nout)
	print("grau do polinomio", pce_degree)
	print("fator multiplicativo", pce_mult)
	print("numero de amostras minimo", Np)
	print("numero de amostras", Ns)

	# dados
	outdir_train = '../results/output_ho_full_orig_train/'
	outdir_test  = '../results/output_ho_full_orig_test/'
	datafile_train = 'trainData.txt'
	datafile_test  = 'testData.txt'	

	# train data
	arq = outdir_train + datafile_train
	data = np.loadtxt(arq, comments='#', delimiter=',')
	samples = data[:Ns,0:8] # trunca ate o numero de amostras
	samples = samples.transpose()
	outputs = data[:Ns,8:14] 
	outputs = outputs.transpose()
	print("data",np.shape(data))
	print("samples",np.shape(samples))
	print("respostas",np.shape(outputs))

	# test data
	arq_test = outdir_test + datafile_test
	data_test = np.loadtxt(arq_test, comments='#', delimiter=',')
	n_test = 100
	samples_test = data_test[:n_test,0:8] # trunca ate n_test
	samples_test = samples_test.transpose()
	outputs_test = data_test[:n_test,8:14] 

	# plots
	labels_samples = ['a','b','af','bf','as','bs','afs','bfs']
	labels_evals = ['alpha1','beta1','alpha2','beta2','vol','fiberStretch']
	scatter_inputs(samples,labels_samples)
	scatter_inputs_outputs(samples,outputs,labels_samples,labels_evals)	

	# estatisticas descritivas
	print("estatisticas dos outputs (mean,std)")
	print(' alfa1: %.2f %.2f' % (np.mean(outputs[0,:]),np.std(outputs[0,:])))
	print(' beta1: %.2f %.2f' % (np.mean(outputs[1,:]),np.std(outputs[1,:])))
	print(' alfa2: %.2f %.2f' % (np.mean(outputs[2,:]),np.std(outputs[2,:])))
	print(' beta2: %.2f %.2f' % (np.mean(outputs[3,:]),np.std(outputs[3,:])))
	print(' vol: %.2f %.2f' % (np.mean(outputs[4,:]),np.std(outputs[4,:])))
	print(' def: %.2f %.2f' % (np.mean(outputs[5,:]),np.std(outputs[5,:])))

	# create the pce emulator
	poly_exp = cp.orth_ttr(pce_degree, distribution)
	#poly_exp = cp.expansion.stieltjes(pce_degree, distribution)
	
	# emuladores
	print('criando emuladores PCE')
	surr_model_alfa1 = cp.fit_regression(poly_exp, samples, outputs[0,:])
	surr_model_beta1 = cp.fit_regression(poly_exp, samples, outputs[1,:])
	surr_model_alfa2 = cp.fit_regression(poly_exp, samples, outputs[2,:])
	surr_model_beta2 = cp.fit_regression(poly_exp, samples, outputs[3,:])	
	surr_model_vol   = cp.fit_regression(poly_exp, samples, outputs[4,:])	
	surr_model_def   = cp.fit_regression(poly_exp, samples, outputs[5,:])

	surrogates = {'alfa1': surr_model_alfa1, 'beta1': surr_model_beta1, 
	 			  'alfa2': surr_model_alfa2, 'beta2': surr_model_beta2, 
				  'edvol': surr_model_vol,   'eddef': surr_model_def}

	tex_labels = {'alfa1': r'$\alpha_1$', 'beta1': r'$\beta_1$', 
	 			  'alfa2': r'$\alpha_2$', 'beta2': r'$\beta_2$', 
				  'edvol': 'volume [mL]', 'eddef': 'fiber stretch [-]'}

	#
	# uncertainty quantification
	#
	if(args.uq):
		print("dados dos emuladores (alfa1,beta1,alfa2,beta2,vol,def)")
		perform_uq(surrogates, distribution)

	#
	# plot QoI distributions
	#
	if(args.qoi):
		print('criando e calculando distribuicoes das QoIs')
		plot_qois(surrogates, distribution, tex_labels)

	#
	# check prediction accuracy
	#
	if(args.test):
		print('previsao dos emuladores')
		r2coef = np.zeros((6))
		for index, skey in enumerate(surrogates):
			surr = surrogates[skey]
			r2coef[index] = pce_prediction(surr, samples_test, outputs_test, index, skey)
		print(' R2 coefs:', r2coef)

	#
	# sensitivity analysis
	#
	if(args.sa):
		print('calculando indices de Sobol (main/total)')
		sobol_m = np.zeros((6,8))
		sobol_t = np.zeros((6,8))
		for index, skey in enumerate(surrogates):
			print(' ' + str(index) + ' qoi: ' + skey)
			surr = surrogates[skey]		 	
			sobol_m[index,:] = cp.Sens_m(surr, distribution)
			sobol_t[index,:] = cp.Sens_t(surr, distribution)

		# salva os indices de sobol em arquivo
		np.savetxt("data_sobol_main.txt", sobol_m, header='a b af bf as bs afs bfs')
		np.savetxt("data_sobol_total.txt", sobol_t, header='a b af bf as bs afs bfs')

# end