import sys
import lmfit
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from uqsa_utils import *
from sklearn.metrics import r2_score, mean_squared_error
from lv_passive_filling import *

#plt.style.use(['science','no-latex'])

# -----------------------------------------------------------------------------

def residual_pce(pars, y_resp, surrogates, monitor):
	npars = len(pars)
	nouts = len(surrogates)
	p = np.zeros(npars)
	p[0] = pars['a']
	p[1] = pars['b']
	p[2] = pars['af']
	p[3] = pars['bf']
	p[4] = pars['a_s']
	p[5] = pars['bs']
	p[6] = pars['afs']
	p[7] = pars['bfs']
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

def residual_fem(pars, y_resp, geoparams, monitor):
    
	# TO-DO: chamar o LVPassiveFilling aqui
	y_fem = LVPassiveFilling(geoparams, pars)
	res = y_fem - y_resp
	monitor.append(np.linalg.norm(res))
	return res	

# -----------------------------------------------------------------------------

if __name__ == "__main__":

	# create mesh and fiber field only once
	print('Creating mesh and fiber fields (f0,s0,n0)')
	mesh, base, endo, epi, ds, nmesh = CreateMesh()
	f0, s0, n0 = CreateFiberfield(mesh)

	geoparams = {'mesh': mesh, 'ds': ds, 'nmesh': nmesh, 
	             'base': base, 'endo': endo, 'epi': epi, 
				 'f0': f0, 's0': s0, 'n0': n0}

	# Holzapfel-Ogden reference values (a0 = 150, b0 = 6.0 -> EF = 56)
	a0 = 150 #228.0	 		# Pa
	b0 = 6.0 #7.780			# dimensionless
	af0 = 116.85	 		# Pa
	bf0 = 11.83425			# dimensionless
	as0 = 372
	bs0 = 5.16
	afs0 = 410
	bfs0 = 11.3

	hoparams = {'a': a0, 'b': b0, 
                'af': af0, 'bf': bf0, 
                'as': as0, 'bs': bs0, 
                'afs': afs0, 'bfs': bfs0} 

    #teste avalaicao residual_fem
	#y_resp = np.zeros(6)
	#residual_fem(hoparams, y_resp, geoparams)

	# parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', type=int, default=2, help="PCE polynomial degree")
	parser.add_argument('-m', type=int, default=2, help="PCE multiplicative factor")
	args = parser.parse_args()


	perc = 0.3

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
	outdir_train = 'results/output_ho_full_orig_train/'
	outdir_test  = 'results/output_ho_full_orig_test/'
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
	print("outputs",np.shape(outputs))

	# test data
	n_test = 1
	arq_test = outdir_test + datafile_test
	data_test = np.loadtxt(arq_test, comments='#', delimiter=',')
	print(np.shape(data_test))
	samples_test = data_test[:n_test,0:8] # trunca ate n_test
	samples_test = samples_test.transpose()
	outputs_test = data_test[:n_test,8:14] 

	# scatter plots
	labels_samples = ['a','b','af','bf','as','bs','afs','bfs']
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

	metodo = 'nelder'                  # Nelder-Mead  
	#metodo = 'leastsq'                 # LM
	#metodo = 'differential_evolution' # DE
	print('lmfit method: %s' % metodo)

	vec_chisqr = np.zeros(Ns)
	vec_nfev = np.zeros(Ns)

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
		fit_params.add("a",   vary=True, value=a0,   min=(1-perc)*a0,   max=(1+perc)*a0)
		fit_params.add("b",   vary=True, value=b0,   min=(1-perc)*b0,   max=(1+perc)*b0)
		fit_params.add("af",  vary=True, value=af0,  min=(1-perc)*af0,  max=(1+perc)*af0)
		fit_params.add("bf",  vary=True, value=bf0,  min=(1-perc)*bf0,  max=(1+perc)*bf0)
		fit_params.add("a_s", vary=True, value=as0,  min=(1-perc)*as0,  max=(1+perc)*as0)
		fit_params.add("bs",  vary=True, value=bs0,  min=(1-perc)*bs0,  max=(1+perc)*bs0)
		fit_params.add("afs", vary=True, value=afs0, min=(1-perc)*afs0, max=(1+perc)*afs0)
		fit_params.add("bfs", vary=True, value=bfs0, min=(1-perc)*bfs0, max=(1+perc)*bfs0)

		done = False
		cont = 1
		conv_monitor = []

		while(not done):

			#print(fit_params)
			#result_fit = lmfit.minimize(residual_pce, fit_params,
            #                           args = (out_true, surrogates, conv_monitor),
            #                           method = metodo, 
			#						   max_nfev=5000)
			result_fit = lmfit.minimize(residual_fem, fit_params, 
										args = (out_true, geoparams, conv_monitor), 
										method = metodo, 
										max_nfev=5000)
			
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

		par_fit = np.zeros(8)
		par_fit[0] = result_fit.params["a"].value
		par_fit[1] = result_fit.params["b"].value
		par_fit[2] = result_fit.params["af"].value
		par_fit[3] = result_fit.params["bf"].value
		par_fit[4] = result_fit.params["a_s"].value
		par_fit[5] = result_fit.params["bs"].value
		par_fit[6] = result_fit.params["afs"].value
		par_fit[7] = result_fit.params["bfs"].value

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

	print('fitting statistics')
	print(' chisqr min: %e' % np.min(vec_chisqr))
	print(' chisqr max: %e' % np.max(vec_chisqr))
	print(' chisqr mean: %e' % np.mean(vec_chisqr))
	print(' nfev min: %e' % np.min(vec_nfev))
	print(' nfev max: %e' % np.max(vec_nfev))
	print(' nfev mean: %e' % np.mean(vec_nfev))

	print('plotting true/pred')
	for ind, skey in enumerate(tex_labels):
		lab = tex_labels[skey]
		xmin,xmax = outputs_test[:,ind].min(), outputs_test[:,ind].max()
		xx = np.linspace(xmin,xmax,1000)
		r2 = r2_score(outputs_test[:,ind], predic_output[:,ind])
		mse = mean_squared_error(outputs_test[:,ind], predic_output[:,ind])
		print('',ind,skey,lab,xmin,xmax,'r2_score:',r2,'mse:',mse)
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