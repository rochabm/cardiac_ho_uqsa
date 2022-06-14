import sys
import lmfit
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from uqsa_utils import *

plt.style.use(['science','no-latex'])

# -----------------------------------------------------------------------------

def residual_pce(pars, y_resp, surrogates, monitor):
	npars = len(pars)
	nouts = len(surrogates)
	p = np.zeros(npars)
	p[0] = pars['q1']
	p[1] = pars['q2']
	p[2] = pars['q3']
	y_pol = np.zeros(nouts)
	for ind, skey in enumerate(surrogates):
		surr = surrogates[skey]
		y_pol[ind] = cp.numpoly.call(surr, p)
	#print(npars, nouts, p, y_pol, y_resp)
	# compute residual vector
	res = y_pol - y_resp
	# monitor convergence
	monitor.append(np.linalg.norm(res))
	return res

# -----------------------------------------------------------------------------

def residual_fem(pars, y_resp):
    npars = len(pars)
    p = np.zeros(npars)
    p[0] = pars['q1']
    p[1] = pars['q2']
    p[2] = pars['q3']
    y_fem = np.zeros(nouts)
    
	# TO-DO: chamar o LVPassiveFilling aqui

    #print(npars, nouts, p, y_pol, y_resp)
    return y_fem - y_resp	

# -----------------------------------------------------------------------------

if __name__ == "__main__":

	# parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', type=int, default=2, help="PCE polynomial degree")
	parser.add_argument('-m', type=int, default=2, help="PCE multiplicative factor")
	args = parser.parse_args()

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
	outdir_train = 'results/output_ho_tiso_rpar_train/'
	outdir_test  = 'results/output_ho_tiso_rpar_test/'
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
	n_test = 100
	arq_test = outdir_test + datafile_test
	data_test = np.loadtxt(arq_test, comments='#', delimiter=',')
	samples_test = data_test[:n_test,0:3] # trunca ate n_test
	samples_test = samples_test.transpose()
	outputs_test = data_test[:n_test,3:9] 

	# scatter plots
	labels_samples = ['q1','q2','q3']
	labels_outputs = ['alpha1','beta1','alpha2','beta2','vol','fiberStretch']

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

	# -------------------------------------------------------------------------
    # AJUSTE usando os emuladores
    # -------------------------------------------------------------------------

	print("fitting using the PCE emulators")

	vec_chisqr = np.zeros(Ns)
	vec_nfev = np.zeros(Ns)

	metodo = 'nelder'
	#metodo = 'leastsq'
	#metodo = 'differential_evolution'
	print('lmfit method: %s' % metodo)
	
	itest = 0
	ntest = n_test

	fitted_params = np.empty((ntest, npar))
	predic_output = np.empty((ntest, nout))
	file_monitor = open('out_convergence.txt','w')
    #outputs_full = respostas_full.transpose()

	for j in range(itest, itest + ntest):
		k = j - itest
		print(" case %d, sample %d:" % (k,j), end='')
		in_true = samples_test[:,j]
		out_true = outputs_test[j,:]

		fit_params = lmfit.Parameters()
		fit_params.add("q1", vary=True, value=1, min=0.7, max=1.3)
		fit_params.add("q2", vary=True, value=1, min=0.7, max=1.3)
		fit_params.add("q3", vary=True, value=1, min=0.7, max=1.3)

		done = False
		cont = 1
		metodo = 'leastsq'
		#metodo = 'differential_evolution'
		conv_monitor = []

		while(not done):

			#print(fit_params)
			result_fit = lmfit.minimize(residual_pce, fit_params,
                                       args = (out_true, surrogates, conv_monitor),
                                       method = metodo, 
									   max_nfev=5000)
									   #, 
                                       #max_nfev=50000)
			
			file_monitor.write("case %d: " % k)
			file_monitor.write(" ".join(str(item) for item in conv_monitor))

			if(result_fit.chisqr > 1e-1):
				#if(cont % 1) != 0:
					#fit_params['q1'].value = fit_params['q1'].value + np.random.normal(0,0.2)
					#fit_params['q2'].value = fit_params['q2'].value + np.random.normal(0,0.2)
					#fit_params['q3'].value = fit_params['q3'].value + np.random.normal(0,0.2)
				print(result_fit.chisqr)
				print(fit_params)
				print(out_true)
				print('skipping')
				done = True
				continue
				#else:
					#metodo = 'differential_evolution'
					#fit_params['a'].value = samples_full[j, 0]
					#fit_params['b'].value = samples_full[j, 1]
					#fit_params['af'].value = samples_full[j, 2]
			else:
				done = True
				print('  nfev: %d, chisqr: %e' % (result_fit.nfev, result_fit.chisqr))
				vec_nfev[k] = result_fit.nfev
				vec_chisqr[k] = result_fit.chisqr
				#lmfit.report_fit(result_fit)

			cont = cont + 1
			#done = True
        # end of fitting

        #if(cont>15):
        #    print("Nao fez um bom ajuste")
        #    sys.exit(0)

		par_fit = np.zeros(3)
		par_fit[0] = result_fit.params["q1"].value
		par_fit[1] = result_fit.params["q2"].value
		par_fit[2] = result_fit.params["q3"].value

		fitted_params[k, :] = par_fit[:]

        # predicted output with fitted parameters
		y_pred = np.zeros(nout)
		for ind, skey in enumerate(surrogates):
			surr = surrogates[skey]
			y_pred[ind] = cp.numpoly.call(surr, par_fit)
        
		# salva na matriz
		predic_output[k, :] = y_pred[:]

		#print('  pred / true / rel error')
		#print(y_pred)
		#print(outputs_test[j,:])
		#print(np.abs(y_pred-outputs_test[j,:])/outputs_test[j,:])
		file_monitor.write('\n')
	
	file_monitor.close()

	print('plotting true/pred')
	for ind, skey in enumerate(tex_labels):
		lab = tex_labels[skey]
		xmin,xmax = outputs_test[:,ind].min(), outputs_test[:,ind].max()
		xx = np.linspace(xmin,xmax,1000)
		print('',ind,skey,lab,xmin,xmax)
		plt.figure()
		plt.plot(xx,xx,'k-')
		for j in range(itest, itest + ntest):
			k = j - itest
			plt.plot(outputs_test[j,ind], predic_output[k,ind], 'bo', markersize=4)
		plt.margins(0.1)
		plt.title(lab)
		plt.xlabel('true')
		plt.ylabel('fitted')
		plt.tight_layout()
		plt.savefig('fig_%s_true_fitted.png' % skey)
		plt.savefig('fig_%s_true_fitted.pdf' % skey)

	print("saving data")
	np.savetxt("fitted_params.txt", fitted_params)
	np.savetxt("predicted_output.txt", predic_output)

	print("done")