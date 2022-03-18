import sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
import argparse
from math import factorial
from loo import calcula_loo
from util import pce_prediction,plot_qoi,scatter_inputs, scatter_inputs_outputs

plt.style.use(['science','no-latex'])

def klotz_pv(p,a,b):
	v = np.zeros(100)
	v = (p/a)**(1.0/b)
	return v

def residual(pars, y_resp):
	p = np.zeros(3)
	p[0] = pars['q1']
	p[1] = pars['q2']
	p[2] = pars['q3']
	y_pol = np.zeros(len(y_resp))
	y_pol[0] = cp.numpoly.call(surr_model_alfa1, p)
	y_pol[1] = cp.numpoly.call(surr_model_beta1, p)
	y_pol[2] = cp.numpoly.call(surr_model_alfa2, p)
	y_pol[3] = cp.numpoly.call(surr_model_beta2, p)
	return y_pol - y_resp

if __name__ == "__main__":

	# parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-uq', dest='uq', action='store_true', help='perform UQ ')
	parser.add_argument('-sa', dest='sa', action='store_true', help='perform SA (Sobol indices)')
	parser.add_argument('-qoi', dest='qoi', action='store_true',help='obtain QoIs dist')
	parser.add_argument('-all', dest='all', action='store_true', help='perform all tasks (UQ/SA/QoI/test)')
	parser.add_argument('-test', dest='test', action='store_true',help='check test data and prediction accuracy')
	args = parser.parse_args()

	if(args.all):
		args.uq = True
		args.sa = True
		args.qoi = True
		args.test = True

	# Holzapfel-Ogden reference values
	a0 = 150 #228.0	 		# Pa
	b0 = 6.0 #7.780			# dimensionless
	af0 = 116.85	 		# Pa
	bf0 = 11.83425			# dimensionless

	# Parameter distributions
	perc = 0.3
	a = cp.Uniform((1-perc)*a0,   (1+perc)*a0)
	b = cp.Uniform((1-perc)*b0,   (1+perc)*b0)
	af = cp.Uniform((1-perc)*af0, (1+perc)*af0)
	bf = cp.Uniform((1-perc)*bf0, (1+perc)*bf0)
	distribution = cp.J(a, b, af, bf)

	npar = 4       # number of parameters
	nout = 6       # (alfa1,beta1,alfa2,beta2,vol,def)
	pce_degree = 5 # polinomial degree
	pce_mult = 2   # multiplicative factor
	Np = int(factorial(npar+pce_degree)/(factorial(npar)*factorial(pce_degree)))
	Ns = pce_mult * Np
	print("numero de parametros de entrada", npar)
	print("numero de saidas", nout)
	print("grau do polinomio", pce_degree)
	print("fator multiplicativo", pce_mult)
	print("numero de amostras", Ns)

    # dados
	outdir_train = 'output_ho_tiso_orig_train/'
	outdir_test  = 'output_ho_tiso_orig_test/'
	datafile_train = 'trainData.txt'
	datafile_test  = 'testData.txt'

	# train data
	arq = outdir_train + datafile_train
	data = np.loadtxt(arq, comments='#', delimiter=',')
	samples = data[:Ns,0:4] # trunca ate o numero de amostras
	samples = samples.transpose()
	outputs = data[:Ns,4:10] 
	outputs = outputs.transpose()
	print("data", np.shape(data))
	print("samples", np.shape(samples))
	print("outputs", np.shape(outputs))

	# test data
	arq_test = outdir_test + datafile_test
	data_test = np.loadtxt(arq_test, comments='#', delimiter=',')
	n_test = 100
	samples_test = data_test[:n_test,0:4] # trunca ate n_test
	samples_test = samples_test.transpose()
	respostas_test = data_test[:n_test,4:10] 
	
	# scatter plots
	labels_samples = ['a','b','af','bf']
	labels_outputs = ['alpha1','beta1','alpha2','beta2','vol','fiberStretch']
	scatter_inputs(samples,labels_samples)
	scatter_inputs_outputs(samples,outputs,labels_samples,labels_outputs)	
	
	# estatisticas descritivas
	print("estatisticas dos outputs (mean,std)")
	print(' alfa1: %.2f %.2f' % (np.mean(outputs[0,:]),np.std(outputs[0,:])))
	print(' beta1: %.2f %.2f' % (np.mean(outputs[1,:]),np.std(outputs[1,:])))
	print(' alfa2: %.2f %.2f' % (np.mean(outputs[2,:]),np.std(outputs[2,:])))
	print(' beta2: %.2f %.2f' % (np.mean(outputs[3,:]),np.std(outputs[3,:])))
	print(' edvol: %.2f %.2f' % (np.mean(outputs[4,:]),np.std(outputs[4,:])))
	print(' eddef: %.2f %.2f' % (np.mean(outputs[5,:]),np.std(outputs[5,:])))

	# create the pce emulator
	poly_exp = cp.orth_ttr(pce_degree, distribution)

	# emuladores
	print('criando emuladores PCE')
	surr_model_alfa1 = cp.fit_regression(poly_exp, samples, outputs[0,:])
	surr_model_beta1 = cp.fit_regression(poly_exp, samples, outputs[1,:])
	surr_model_alfa2 = cp.fit_regression(poly_exp, samples, outputs[2,:])
	surr_model_beta2 = cp.fit_regression(poly_exp, samples, outputs[3,:])	
	surr_model_edvol = cp.fit_regression(poly_exp, samples, outputs[4,:])	
	surr_model_eddef = cp.fit_regression(poly_exp, samples, outputs[5,:])	

	surrogates = {'alfa1': surr_model_alfa1, 'beta1': surr_model_beta1, 
	 			  'alfa2': surr_model_alfa2, 'beta2': surr_model_beta2, 
				  'edvol': surr_model_edvol, 'eddef': surr_model_eddef}

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
		dist_alfa1 = cp.QoI_Dist(surr_model_alfa1, distribution)
		dist_beta1 = cp.QoI_Dist(surr_model_beta1, distribution)
		dist_alfa2 = cp.QoI_Dist(surr_model_alfa2, distribution)
		dist_beta2 = cp.QoI_Dist(surr_model_beta2, distribution)
		dist_edvol = cp.QoI_Dist(surr_model_edvol, distribution)
		dist_eddef = cp.QoI_Dist(surr_model_eddef, distribution)
		plot_qoi(dist_alfa1, 'hist_alfa1', r'$\alpha_1$')
		plot_qoi(dist_beta1, 'hist_beta1', r'$\beta_1$')
		plot_qoi(dist_alfa2, 'hist_alfa2', r'$\alpha_2$')
		plot_qoi(dist_beta2, 'hist_beta2', r'$\beta_2$')
		plot_qoi(dist_edvol, 'hist_vol', 'volume [mL]')
		plot_qoi(dist_eddef, 'hist_def', 'fiber stretch [-]')

	#
	# check prediction accuracy
	#
	if(args.test):
		print('previsao dos emuladores')
		r2coef = np.zeros((6))
		r2coef[0] = pce_prediction(surr_model_alfa1, samples_test, respostas_test, 0, 'alfa1')
		r2coef[1] = pce_prediction(surr_model_beta1, samples_test, respostas_test, 1, 'beta1')
		r2coef[2] = pce_prediction(surr_model_alfa2, samples_test, respostas_test, 2, 'alfa2')
		r2coef[3] = pce_prediction(surr_model_beta2, samples_test, respostas_test, 3, 'beta2')
		r2coef[4] = pce_prediction(surr_model_edvol, samples_test, respostas_test, 4, 'vol')
		r2coef[5] = pce_prediction(surr_model_eddef, samples_test, respostas_test, 5, 'def')
		print(' R2 min:', r2coef.min())
		print(' R2 max:', r2coef.max())

	#
	# sensitivity analysis
	#
	if(args.sa):

		# main sobol indices
		print('calculando indices de Sobol (main/total)')
		np.set_printoptions(precision=2)

		sobol_m = np.zeros((6,4))
		sobol_t = np.zeros((6,4))

		for index, skey in enumerate(surrogates):
			print(' ' + str(index) + ' qoi: ' + skey)
			surr = surrogates[skey]		 	
			sobol_m[index,:] = cp.Sens_m(surr, distribution)
			sobol_t[index,:] = cp.Sens_t(surr, distribution)

		# salva os indices de sobol em arquivo
		sm_file = outdir_train + "sobol_main.txt"
		st_file = outdir_train + "sobol_total.txt"
		np.savetxt(sm_file, sobol_m, header='a b af bf')
		np.savetxt(st_file, sobol_t, header='a b af bf')

