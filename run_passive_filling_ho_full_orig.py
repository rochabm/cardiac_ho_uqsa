import os, sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from lv_passive_filling import *

if __name__ == "__main__":
	
	os.makedirs("output/", exist_ok=True)

	# Holzapfel-Ogden reference values (a0 = 150, b0 = 6.0 -> EF = 56)
	a0 = 150 #228.0	 		# Pa
	b0 = 6.0 #7.780			# dimensionless
	af0 = 116.85	 		# Pa
	bf0 = 11.83425			# dimensionless
	as0 = 372
	bs0 = 5.16
	afs0 = 410
	bfs0 = 11.3

	a   = cp.Uniform(0.7*a0,   1.3*a0)
	b   = cp.Uniform(0.7*b0,   1.3*b0)
	af  = cp.Uniform(0.7*af0,  1.3*af0)
	bf  = cp.Uniform(0.7*bf0,  1.3*bf0)
	as1 = cp.Uniform(0.7*as0,  1.3*as0)
	bs  = cp.Uniform(0.7*bs0,  1.3*bs0)
	afs = cp.Uniform(0.7*afs0, 1.3*afs0)
	bfs = cp.Uniform(0.7*bfs0, 1.3*bfs0)
	distribution = cp.J(a, b, af, bf, as1, bs, afs, bfs)

	# Code for creating gPC expansion
	npar = 8        # number of input parameter  
	nout = 6        # number of outputs
	pce_degree = 5  # polynomial degree	
	pce_mult = 2	# multiplicative factor
	Np = int(factorial(npar+pce_degree)/(factorial(npar)*factorial(pce_degree))) # min number of terms for PCE
	Ns = pce_mult * Np		#number of samples
	
	# number of sample to start 
	start = 0

	print("numero de parametros de entrada", npar)
	print("numero de saidas", nout)
	print("grau do polinomio", pce_degree)
	print('numero minimo de amostras', Np)
	print("fator multiplicativo", pce_mult)
	print("numero de amostras", Ns)

	# take samples from joint distribution
	samples = distribution.sample(Ns)

	# prepare for output 
	shutil.rmtree("output/") 
	os.mkdir("output/")

	# output data for inputs/outputs
	fd = "output/trainData.txt"
	#fd = "output/testData.txt"
	f_data = open(fd,'w')
	f_data.write('# a, b, af, bf, as, bs, afs, bfs, alfa1, beta1, alfa2, beta2, vol, fibstretch \n')

	# create mesh and fiber field only once
	print('Creating mesh and fiber fields (f0,s0,n0)')
	mesh, base, endo, epi, ds, nmesh = CreateMesh()
	f0, s0, n0 = CreateFiberfield(mesh)

	geoparams = {'mesh': mesh, 'ds': ds, 'nmesh': nmesh, 
	             'base': base, 'endo': endo, 'epi': epi, 
				 'f0': f0, 's0': s0, 'n0': n0}

	# list to append outputs
	evals = []

	# solve model for each sample
	for i in range(len(samples.T)):
		Z = samples.T[i,:]

		# reduced parametrization
		a,   b   = float(Z[0]), float(Z[1])
		af,  bf  = float(Z[2]), float(Z[3])
		as1, bs  = float(Z[4]), float(Z[5])
		afs, bfs = float(Z[6]), float(Z[7])

		hoparams = {'a': a, 'b': b, 
		            'af': af, 'bf': bf, 
					'as': as1, 'bs': bs, 
					'afs': afs, 'bfs': bfs} 

		qin = np.array([a,b,af,bf,as1,bs,afs,bfs])
		aux = i + start

		print("\nRodando caso %d: (%f,%f,%f,%f,%f,%f,%f,%f)" % (aux, a, b, af, bf, as1, bs, afs, bfs))
		
		#out = LVPassiveFilling(i, a, b, af, bf, a_s, bs, afs, bfs)
		#out = LVPassiveFilling(mesh, base, endo, epi, ds, nmesh, f0, s0, n0, i, a, b, af, bf, as1, bs, afs, bfs)
		out = LVPassiveFilling(geoparams, hoparams, i)

		# senao deu erro, salva os dados
		if out is not None:
			evals.append(out)
			np.savetxt(f_data, np.concatenate((qin,out)).reshape((1,14)), fmt='%16.8e', delimiter=',')
			f_data.flush()
		else:
			print("\n\Erro na execucao da amostra\n\n")

		# senao deu erro, salva os dados
		#if out is not None:
			#evals.append(out)

			#np.savetxt(f_samp_q, samples[i,:], fmt='%16.8f', newline=" ")
			#f_samp_q.write("\n")
			#f_samp_q.flush()

			#np.savetxt(f_samp_ho, qin, fmt='%16.8f', newline=" ")
			#f_samp_ho.write("\n")
			#f_samp_ho.flush()	

			#np.savetxt(f_outputs, out[0], fmt='%16.8f', newline=" ")
			#f_outputs.write("\n")
			#f_outputs.flush()

			#name_q = "output_q/samp_q_%04d.txt" % aux
			#name_ho = "output_ho/samp_ho_%04d.txt" % aux
			#name_resp = "output_resp/resp_%04d.txt" % aux
			#name_vol = "output_vol/volume_%04d.txt" % aux
			#name_def = "output_def/deformation_%04d.txt" % aux
			#np.savetxt(name_q, samples[i,:], fmt='%16.8f', newline=" ")
			#np.savetxt(name_ho, qin, fmt='%16.8f', newline=" ")
			#np.savetxt(name_resp, out[0], fmt='%16.8f', newline=" ")
			#np.savetxt(name_vol, np.array([out[1]]), fmt='%16.8f', newline=" ")
			#np.savetxt(name_def, np.array([out[2]]), fmt='%16.8f', newline=" ")

		#else:
			#print("\n\nERRO na execucao da amostra\n\n")
			#evals.append(out)

			#zeros = np.array([0,0,0,0,0,0,0,0])
			#zeros = zeros.T
			#zero = np.array([0])

			#np.savetxt(f_samp_q, zeros, fmt='%16.8f', newline=" ")
			#f_samp_q.write("\n")
			#f_samp_q.flush()

			#np.savetxt(f_samp_ho, zeros, fmt='%16.8f', newline=" ")
			#f_samp_ho.write("\n")
			#f_samp_ho.flush()	

			#np.savetxt(f_outputs, zeros, fmt='%16.8f', newline=" ")
			#f_outputs.write("\n")
			#f_outputs.flush()
			
			#name_q = "output_q/samp_q_%04d.txt" % aux
			#name_ho = "output_ho/samp_ho_%04d.txt" % aux
			#name_resp = "output_resp/resp_%04d.txt" % aux
			#name_vol = "output_vol/volume_%04d.txt" % aux
			#name_def = "output_def/deformation_%04d.txt" % aux
			#np.savetxt(name_q, zeros, fmt='%16.8f', newline=" ")
			#np.savetxt(name_ho, zeros, fmt='%16.8f', newline=" ")
			#np.savetxt(name_resp, zeros, fmt='%16.8f', newline=" ")
			#np.savetxt(name_vol, zero, fmt='%16.8f', newline=" ")
			#np.savetxt(name_def, zero, fmt='%16.8f', newline=" ")

	# end of loop in samples

	f_data.close()

	# salva os dados
	evals = np.array(evals)
	np.savetxt("output/all_evals.txt", np.transpose(evals))
	np.savetxt("output/all_samples.txt", samples)

	#f_samp_q.close()
	#f_samp_ho.close()
	#f_outputs.close()

	# salva os dados
	#evals = np.array(evals)
	#evals = np.array(evals[0,:,:])	
	#np.savetxt("output/all_samples", samples)
	#np.savetxt("output/all_evals", evals)