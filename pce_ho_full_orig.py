import sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
import lmfit
import time
import argparse
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
	print("numero de amostras", Ns)

	# dados
	outdir = '../results/output_ho_full_orig_train/'
	datafile = 'trainData.txt'

	arq = outdir + datafile

	data = np.loadtxt(arq, comments='#', delimiter=',')
	samples = data[:Ns,0:8] # trunca ate o numero de amostras
	samples = samples.transpose()
	outputs = data[:Ns,8:14] 
	outputs = outputs.transpose()
	print("data",np.shape(data))
	print("samples",np.shape(samples))
	print("respostas",np.shape(outputs))

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
		print(' R2 min:', r2coef.min())
		print(' R2 max:', r2coef.max())

	#
	# sensitivity analysis
	#
	if(args.sa):
		print('calculando indices de Sobol (main/total)')
		sobol_m = np.zeros((6,4))
		sobol_t = np.zeros((6,4))
		for index, skey in enumerate(surrogates):
			print(' ' + str(index) + ' qoi: ' + skey)
			surr = surrogates[skey]		 	
			sobol_m[index,:] = cp.Sens_m(surr, distribution)
			sobol_t[index,:] = cp.Sens_t(surr, distribution)

		# salva os indices de sobol em arquivo
		np.savetxt("data_sobol_main.txt", sobol_m, header='q1 q2 q3 q4')
		np.savetxt("data_sobol_total.txt", sobol_t, header='q1 q2 q3 q4')
	
	"""
	#
	# sensitivity analysis
	# 
	if(args.sa == True):

		# main sobol indices	
		print('calculando indices de Sobol main')
		np.set_printoptions(precision=2)

		sobol_m = np.zeros((8,6))
		sobol_t = np.zeros((8,6))

		sens_alpha1 = cp.Sens_m(surr_model_alfa1, distribution)
		print(" main sobol alpha1:", sens_alpha1)
		sobol_m[:,0] = sens_alpha1
		
		sens_beta1 = cp.Sens_m(surr_model_beta1, distribution)
		print(" main sobol beta1:", sens_beta1)
		sobol_m[:,1] = sens_beta1
		
		sens_alpha2 = cp.Sens_m(surr_model_alfa2, distribution)
		print(" main sobol alpha2:", sens_alpha2)
		sobol_m[:,2] = sens_alpha2
		
		sens_beta2 = cp.Sens_m(surr_model_beta2, distribution)
		print(" main sobol beta2:", sens_beta2)
		sobol_m[:,3] = sens_beta2

		sens_vol = cp.Sens_m(surr_model_vol, distribution)
		print(" main sobol vol:", sens_vol)
		sobol_m[:,4] = sens_vol

		sens_def = cp.Sens_m(surr_model_def, distribution)
		print(" main sobol def:", sens_def)
		sobol_m[:,5] = sens_beta2

		# total sobol indices
		print('calculando indices de Sobol total')

		tsens_alpha1 = cp.Sens_t(surr_model_alfa1, distribution)
		print(" total sobol alpha1:", sens_alpha1)
		sobol_t[:,0] = sens_alpha1
		
		tsens_beta1 = cp.Sens_t(surr_model_beta1, distribution)
		print(" total sobol beta1:", sens_beta1)
		sobol_t[:,1] = sens_beta1

		tsens_alpha2 = cp.Sens_t(surr_model_alfa2, distribution)
		print(" total sobol alpha2:", sens_alpha2)
		sobol_t[:,2] = sens_alpha2
		
		tsens_beta2 = cp.Sens_t(surr_model_beta2, distribution)
		print(" total sobol beta2:", sens_beta2)
		sobol_t[:,3] = sens_beta2

		tsens_vol = cp.Sens_t(surr_model_vol, distribution)
		print(" total sobol vol:", tsens_vol)
		sobol_t[:,4] = sens_vol

		tsens_def = cp.Sens_t(surr_model_beta2, distribution)
		print(" total sobol def:", tsens_def)
		sobol_t[:,5] = sens_beta2
		
		np.savetxt("sobol_main.txt", sobol_m)
		np.savetxt("sobol_total.txt", sobol_t)
	"""
	
	sys.exit()

	# plots
	y = np.linspace(0,2.7,100)
	model1 = expo(mean_alfa1, mean_beta1,y)
	model_s1 = expo(std_alfa1, std_beta1,y)
	plt.fill_between(y,model1 + 3*model_s1, model1 - 3*model_s1, alpha = 0.4)
	plt.plot(y,model1)
	plt.xlabel("Pressão")
	plt.ylabel("Volume")
	plt.show()

	model2 = expo(mean_alfa2, mean_beta2,y)
	model_s2 = expo(std_alfa2, std_beta2,y)
	plt.figure()
	#plt.fill_between(y,model2 + 3*model_s2, model2 - 3*model_s2, alpha = 0.4)
	plt.plot(y, model2)
	plt.xlabel("Pressão")
	plt.ylabel("Deformação")
	plt.show()

	#
	# avaliação de erro do modelo
	#
	pol_matrix = np.zeros((nout, tam))
	error_matrix = np.zeros((nout, tam))

	for i in range(len(samples[0,:])):
		pol_matrix[0,i] = cp.numpoly.call(surr_model_alfa1, samples[:,i])
		pol_matrix[1,i] = cp.numpoly.call(surr_model_beta1, samples[:,i])
		pol_matrix[2,i] = cp.numpoly.call(surr_model_alfa2, samples[:,i])
		pol_matrix[3,i] = cp.numpoly.call(surr_model_beta2, samples[:,i])

	for i in range(nout):
		for j in range(tam):
			error_matrix[i,j] = abs(pol_matrix[i,j] - evals[i,j])/abs(pol_matrix[i,j])
			#if(i==0 and error_matrix[i,j] > 0.10):
			#	print("aqui")
			#print(i, j, pol_matrix[i,j], evals[i,j], error_matrix[i,j])
		#print("")

	#print(error_matrix)	

	alfa1_max_error = np.max(error_matrix[0,:])*100
	beta1_max_error = np.max(error_matrix[1,:])*100
	alfa2_max_error = np.max(error_matrix[2,:])*100
	beta2_max_error = np.max(error_matrix[3,:])*100

	#print( np.max(error_matrix[0,:]) )
	#print( np.argmax(error_matrix[0,:]) )

	print("\nMax Erro alfa1: ", alfa1_max_error)
	print("Max Erro beta1: ", beta1_max_error)
	print("Max Erro alfa2: ", alfa2_max_error)
	print("Max Erro beta2: ", beta2_max_error)
	
	alfa1_avg_error = np.mean(error_matrix[0,:])*100
	beta1_avg_error = np.mean(error_matrix[1,:])*100
	alfa2_avg_error = np.mean(error_matrix[2,:])*100
	beta2_avg_error = np.mean(error_matrix[3,:])*100

	print("\nAvg Erro alfa1: ", alfa1_avg_error)
	print("Avg Erro beta1: ", beta1_avg_error)
	print("Avg Erro alfa2: ", alfa2_avg_error)
	print("Avg Erro beta2: ", beta2_avg_error)

	alfa1_std_error = np.std(error_matrix[0,:])*100
	beta1_std_error = np.std(error_matrix[1,:])*100
	alfa2_std_error = np.std(error_matrix[2,:])*100
	beta2_std_error = np.std(error_matrix[3,:])*100

	print("\nStd Erro alfa1: ", alfa1_std_error)
	print("Std Erro beta1: ", beta1_std_error)
	print("Std Erro alfa2: ", alfa2_std_error)
	print("Std Erro beta2: ", beta2_std_error)
	
	data_evals = np.loadtxt("respostas_para_avaliacao.txt")
	print(data_evals)
	N = len(data_evals[:,0])
	par_q = np.zeros((N, 4))
	par_de = np.zeros((N, 4))
	resp_de = np.zeros((N, 4))
	func_evals = np.zeros(N)
	tempo_exec = np.zeros(N)
	for j in range(N):
		print("\n Avaliando a resposta", j)
		start = time.time()
		y_resp = data_evals[j,:]
		
		# Holzapfel-Ogden reference values
		a0 = 228.0	 			# Pa
		b0 = 7.780				# dimensionless
		a_f0 = 116.85	 		# Pa
		b_f0 = 11.83425			# dimensionless
			
		print("computing LOO xx")
		print(outputs)
		#yy = np.linspace(0,Ns-1,Ns)
		
		#calc_loo_alpha1 = calc_loo(yy, npar, poly_exp, yy_std, samples, distribution)
		calc_loo_alpha1 = calcula_loo(outputs[0,:], poly_exp, samples, distribution)
		print(calc_loo_alpha1)
		print("Q2 alpha 1: ", np.mean(calc_loo_alpha1))

		calc_loo_beta1 = calcula_loo(outputs[1,:], poly_exp, samples, distribution)
		#calc_loo_beta1 = calc_loo(yy, npar, poly_exp, yy_std, samples, distribution)
		print(calc_loo_beta1)
		print("Q2 beta 1: ", np.mean(calc_loo_beta1))
		
		calc_loo_alpha2 = calcula_loo(outputs[2,:], poly_exp, samples, distribution)
		#calc_loo_alpha2 = calc_loo(yy, npar, poly_exp, yy_std, samples, distribution)
		print(calc_loo_alpha2)
		print("Q2 alpha 2: ", np.mean(calc_loo_alpha2))

		calc_loo_beta2 = calcula_loo(outputs[3,:], poly_exp, samples, distribution)
		#calc_loo_beta2 = calc_loo(yy, npar, poly_exp, yy_std, samples, distribution)
		print(calc_loo_beta2)
		print("Q2 beta 2: ", np.mean(calc_loo_beta2))
		
		#
		# AJUSTE COM EMULADOR
		#

		fit_params = lmfit.Parameters()
		# parametros 
		fit_params.add("q1", vary=True, min=0.1, max=5)
		fit_params.add("q2", vary=True, min=0.1, max=5)
		fit_params.add("q3", vary=True, min=0.1, max=5)
		
		result_de = lmfit.minimize(residual, fit_params, args = (y_resp,), method = "differential_evolution")  # fitting
		lmfit.report_fit(result_de)
		func_evals[j] = result_de.nfev
		qq1 = result_de.params["q1"].value
		qq2 = result_de.params["q2"].value
		qq3 = result_de.params["q3"].value
		a_de = a0 * qq1
		b_de = b0 * qq1
		af_de = a_f0 * qq2
		bf_de = b_f0 * qq3
		#print("Parâmetros recuperados differential evolution:")
		print("a: ", a_de)
		print("b: ", b_de)
		print("af: ", af_de)
		print("bf: ", bf_de)

		par_q[j,0] = qq1
		par_q[j,1] = qq2
		par_q[j,2] = qq3

		par_de[j,0] = a_de
		par_de[j,1] = b_de
		par_de[j,2] = af_de
		par_de[j,3] = bf_de
		
		resp_de[j,0] =  cp.numpoly.call(surr_model_alfa1, par_q[j,:])
		resp_de[j,1] =  cp.numpoly.call(surr_model_beta1, par_q[j,:])
		resp_de[j,2] =  cp.numpoly.call(surr_model_alfa2, par_q[j,:])
		resp_de[j,3] =  cp.numpoly.call(surr_model_beta2, par_q[j,:])
		end = time.time()
		tempo_exec[j] = end - start
		print(result_de.chisqr)
		print("tempo de execução:",tempo_exec[j])
	
	np.savetxt("test_inp_param.txt", par_de)
	np.savetxt("test_out_pce.txt", resp_de)
	np.savetxt("funcoes_avaliadas.txt", func_evals)
	np.savetxt("tempo_execucao.txt", tempo_exec)
	
# 4. TO-DO: criar outro arquivo python (plot_predictions.py)
# plotar test_out_pce VERSUS respostas.txt