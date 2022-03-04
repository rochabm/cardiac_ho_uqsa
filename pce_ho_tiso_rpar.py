#import dolfin
#from fenics import *
#from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
import lmfit
import time
from math import factorial
from loo import calcula_loo

#plt.style.use('ieee')

def expo(a,b,y):
	x = np.zeros(100)
	x = (y/a)**(1/b)
	return x

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

def scatter_inputs(samples):
	#scatter plot
	fig ,axs = plt.subplots(2,2)
	axs[0,0].scatter(samples[0,:],samples[1,:])#,marker = 'o')
	axs[0,0].set_xlabel("q1")
	axs[0,0].set_ylabel("q2")
	axs[0,1].axis('off')
	axs[1,0].scatter(samples[0,:],samples[2,:],marker = 'o')
	axs[1,0].set_xlabel("q1")
	axs[1,0].set_ylabel("q3")
	axs[1,1].scatter(samples[1,:],samples[2,:],marker = 'o')
	axs[1,1].set_xlabel("q2")
	axs[1,1].set_ylabel("q3")
	plt.tight_layout()
	plt.savefig('fig_scatter_inputs.png')
	#plt.show()

def scatter_inputs_outputs(samples,evals):
	fig ,axs = plt.subplots(6,3)
	axs[0,0].scatter(samples[0,:], evals[0,:], marker = 'o')
	axs[0,0].set_ylabel("alpha1")
	axs[1,0].scatter(samples[0,:], evals[1,:], marker = 'o')
	axs[1,0].set_ylabel("beta1")
	axs[2,0].scatter(samples[0,:], evals[2,:], marker = 'o')
	axs[2,0].set_ylabel("alpha2")
	axs[3,0].scatter(samples[0,:], evals[3,:], marker = 'o')
	axs[3,0].set_ylabel("beta2")
	axs[4,0].scatter(samples[2,:], evals[4,:], marker = 'o')
	axs[3,0].set_ylabel("vol")
	axs[5,0].scatter(samples[2,:], evals[5,:], marker = 'o')
	axs[3,0].set_ylabel("def")
	axs[3,0].set_xlabel("q1")

	axs[0,1].scatter(samples[1,:], evals[0,:], marker = 'o')
	axs[1,1].scatter(samples[1,:], evals[1,:], marker = 'o')
	axs[2,1].scatter(samples[1,:], evals[2,:], marker = 'o')
	axs[3,1].scatter(samples[1,:], evals[3,:], marker = 'o')
	axs[4,1].scatter(samples[2,:], evals[4,:], marker = 'o')
	axs[5,1].scatter(samples[2,:], evals[5,:], marker = 'o')
	axs[3,1].set_xlabel("q2")

	axs[0,2].scatter(samples[2,:], evals[0,:], marker = 'o')
	axs[1,2].scatter(samples[2,:], evals[1,:], marker = 'o')
	axs[2,2].scatter(samples[2,:], evals[2,:], marker = 'o')
	axs[3,2].scatter(samples[2,:], evals[3,:], marker = 'o')
	axs[4,2].scatter(samples[2,:], evals[4,:], marker = 'o')
	axs[5,2].scatter(samples[2,:], evals[5,:], marker = 'o')
	axs[3,2].set_xlabel("q3")
	plt.tight_layout()
	plt.savefig('fig_scatter_inputs_outputs.png')
	#plt.show()

if __name__ == "__main__":

	# Holzapfel-Ogden reduced parametrization
	Z1 = cp.Uniform(0.7, 1.3)   # q1
	Z2 = cp.Uniform(0.7, 1.3)   # q2
	Z3 = cp.Uniform(0.7, 1.3)   # q3
	distribution = cp.J(Z1, Z2, Z3)

	npar = 3  # number of parameters
	nout = 6 # (alfa1,beta1,alfa2,beta2,vol,def)

	pce_degree = 5 # polinomial degree
	pce_mult = 2 # multiplicative factor
	Np = int(factorial(npar+pce_degree)/(factorial(npar)*factorial(pce_degree)))
	Ns = pce_mult * Np
	print("numero de parametros de entrada", npar)
	print("numero de saidas", nout)
	print("grau do polinomio", pce_degree)
	print("fator multiplicativo", pce_mult)
	print("numero de amostras", Ns)

	# dados
	outdir = 'output_ho_tiso_rpar_train/'
	datafile = 'trainData.txt'

	arq = outdir + datafile

	data = np.loadtxt(arq, comments='#', delimiter=',')
	samples = data[:Ns,0:3] # trunca ate o numero de amostras
	samples = samples.transpose()
	respostas = data[:Ns,3:9] 
	print("data",np.shape(data))
	print("samples",np.shape(samples))
	print("respostas",np.shape(respostas))
	
	tam = np.shape(respostas)[0]
	evals = np.zeros((nout,tam))
	evals[0,:] = respostas[:,0]
	evals[1,:] = respostas[:,1]
	evals[2,:] = respostas[:,2]
	evals[3,:] = respostas[:,3]
	evals[4,:] = respostas[:,4]
	evals[5,:] = respostas[:,5]

	outputs = evals.copy()

	scatter_inputs(samples)
	scatter_inputs_outputs(samples,evals)	

	print("estatisticas dos outputs (mean,std)")
	print(' alfa1: %.2f %.2f' % (np.mean(evals[0,:]),np.std(evals[0,:])))
	print(' beta1: %.2f %.2f' % (np.mean(evals[1,:]),np.std(evals[1,:])))
	print(' alfa2: %.2f %.2f' % (np.mean(evals[2,:]),np.std(evals[2,:])))
	print(' beta2: %.2f %.2f' % (np.mean(evals[3,:]),np.std(evals[3,:])))
	print(' vol: %.2f %.2f' % (np.mean(evals[4,:]),np.std(evals[4,:])))
	print(' def: %.2f %.2f' % (np.mean(evals[5,:]),np.std(evals[5,:])))

	# create the pce emulator
	poly_exp = cp.orth_ttr(pce_degree, distribution)

	# emuladores
	print('criando emuladores PCE')
	surr_model_alfa1 = cp.fit_regression(poly_exp, samples, evals[0,:])
	surr_model_beta1 = cp.fit_regression(poly_exp, samples, evals[1,:])
	surr_model_alfa2 = cp.fit_regression(poly_exp, samples, evals[2,:])
	surr_model_beta2 = cp.fit_regression(poly_exp, samples, evals[3,:])	
	surr_model_vol   = cp.fit_regression(poly_exp, samples, evals[4,:])	
	surr_model_def   = cp.fit_regression(poly_exp, samples, evals[5,:])	

	# dados estatisticos	
	mean_alfa1 = cp.E(surr_model_alfa1, distribution)
	std_alfa1 = cp.Std(surr_model_alfa1, distribution)

	mean_beta1 = cp.E(surr_model_beta1, distribution)
	std_beta1 = cp.Std(surr_model_beta1, distribution)

	mean_alfa2 = cp.E(surr_model_alfa2, distribution)
	std_alfa2 = cp.Std(surr_model_alfa2, distribution)

	mean_beta2 = cp.E(surr_model_beta2, distribution)
	std_beta2 = cp.Std(surr_model_beta2, distribution)

	mean_vol = cp.E(surr_model_vol, distribution)
	std_vol = cp.Std(surr_model_vol, distribution)

	mean_def = cp.E(surr_model_def, distribution)
	std_def = cp.Std(surr_model_def, distribution)
	
	print("dados dos emuladores (alfa1,beta1,alfa2,beta2,vol,def)")	
	print(" alfa1: %.2f %.2f" % (mean_alfa1, std_alfa1))
	print(" beta1: %.2f %.2f" % (mean_beta1, std_beta1))
	print(" alfa2: %.2f %.2f" % (mean_alfa2, std_alfa2))
	print(" beta2: %.2f %.2f" % (mean_beta2, std_beta2))
	print(" edvol: %.2f %.2f" % (mean_vol, std_vol))
	print(" eddef: %.2f %.2f" % (mean_def, std_def))

	# main sobol indices
	print('calculando indices de Sobol main')
	np.set_printoptions(precision=2)

	sobol_m = np.zeros((3,8))
	sobol_t = np.zeros((3,8))

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