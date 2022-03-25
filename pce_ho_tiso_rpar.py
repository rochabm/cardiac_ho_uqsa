import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from uqsa_utils import *

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

	# Holzapfel-Ogden reduced parametrization
	Z1 = cp.Uniform(0.7, 1.3)   # q1
	Z2 = cp.Uniform(0.7, 1.3)   # q2
	Z3 = cp.Uniform(0.7, 1.3)   # q3
	distribution = cp.J(Z1, Z2, Z3)

	npar = 3            # number of parameters
	nout = 6            # (alfa1,beta1,alfa2,beta2,vol,def)
	pce_degree = args.d # polinomial degree
	pce_mult = args.m   # multiplicative factor
	Np = int(factorial(npar+pce_degree)/(factorial(npar)*factorial(pce_degree)))
	Ns = pce_mult * Np
	print("numero de parametros de entrada", npar)
	print("numero de saidas", nout)
	print("grau do polinomio", pce_degree)
	print("fator multiplicativo", pce_mult)
	print("numero de amostras", Ns)

    # dados
	outdir_train = '../results/output_ho_tiso_rpar_train/'
	outdir_test  = '../results/output_ho_tiso_rpar_test/'
	datafile_train = 'trainData.txt'
	datafile_test  = 'testData.txt'

	# train data
	arq = outdir_train + datafile_train
	data = np.loadtxt(arq, comments='#', delimiter=',')
	samples = data[:Ns,0:3] # trunca ate o numero de amostras
	samples = samples.transpose()
	outputs = data[:Ns,3:9] 
	outputs = outputs.transpose()
	print("data",np.shape(data))
	print("samples",np.shape(samples))
	print("outputs",np.shape(outputs))

	# test data
	arq_test = outdir_test + datafile_test
	data_test = np.loadtxt(arq_test, comments='#', delimiter=',')
	n_test = 100
	samples_test = data_test[:n_test,0:3] # trunca ate n_test
	samples_test = samples_test.transpose()
	outputs_test = data_test[:n_test,3:9] 

	# scatter plots
	labels_samples = ['q1','q2','q3']
	labels_outputs = ['alpha1','beta1','alpha2','beta2','vol','fiberStretch']
	scatter_inputs(samples,labels_samples)
	scatter_inputs_outputs(samples,outputs,labels_samples,labels_outputs)	

	print("estatisticas dos outputs (mean,std)")
	print(' alfa1: %.2f %.2f' % (np.mean(outputs[0,:]),np.std(outputs[0,:])))
	print(' beta1: %.2f %.2f' % (np.mean(outputs[1,:]),np.std(outputs[1,:])))
	print(' alfa2: %.2f %.2f' % (np.mean(outputs[2,:]),np.std(outputs[2,:])))
	print(' beta2: %.2f %.2f' % (np.mean(outputs[3,:]),np.std(outputs[3,:])))
	print(' vol: %.2f %.2f' % (np.mean(outputs[4,:]),np.std(outputs[4,:])))
	print(' def: %.2f %.2f' % (np.mean(outputs[5,:]),np.std(outputs[5,:])))

	# create the pce emulator
	poly_exp = cp.orth_ttr(pce_degree, distribution)

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
		print(' R2 min:', r2coef.min())
		print(' R2 max:', r2coef.max())

	#
	# sensitivity analysis
	#
	if(args.sa):
		print('calculando indices de Sobol (main/total)')
		sobol_m = np.zeros((6,3))
		sobol_t = np.zeros((6,3))
		for index, skey in enumerate(surrogates):
			print(' ' + str(index) + ' qoi: ' + skey)
			surr = surrogates[skey]		 	
			sobol_m[index,:] = cp.Sens_m(surr, distribution)
			sobol_t[index,:] = cp.Sens_t(surr, distribution)

		# salva os indices de sobol em arquivo
		np.savetxt("data_sobol_main.txt", sobol_m, header='q1 q2 q3')
		np.savetxt("data_sobol_total.txt", sobol_t, header='q1 q2 q3')
