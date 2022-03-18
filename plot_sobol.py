import sys
import numpy as np
import matplotlib.pyplot as plt 
import chaospy as cp

plt.style.use(['science','no-latex'])

# -----------------------------------------------------------------------------

def sobol(sens_m, sens_t, in_names):

	nout = np.shape(sens_m)[0] # number of outputs
	nin = np.shape(sens_m)[1]  # number of inputs
	
	out_names = [r'$\alpha_1$',r'$\beta_1$',r'$\alpha_2$',r'$\beta_2$','EDV','fiber stretch']
	
	for i in range(nout):
		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])
		ax.bar(in_names,sens_t[i,:],label='total')
		ax.bar(in_names,sens_m[i,:],label='main')
		plt.ylim([0,1])
		plt.ylabel('sensitivity')
		plt.title(out_names[i])
		plt.legend(loc='best')
		plt.tight_layout()
		plt.savefig('fig_sobol_%d.png' % i)
		plt.savefig('fig_sobol_%d.pdf' % i)

# -----------------------------------------------------------------------------

if __name__ == "__main__":

	basedir = sys.argv[1]

	# files for sobol indices
	arq_m = basedir + 'sobol_main.txt'
	arq_t = basedir + 'sobol_total.txt'

	# read files
	sens_m = np.loadtxt(arq_m, comments='#')
	sens_t = np.loadtxt(arq_t, comments='#')
	print(np.shape(sens_m))
	print(np.shape(sens_t))

	# labels for inputs
	in_names_tiso_rpar = [r'$q_1$', r'$q_2$', r'$q_3$']
	in_names_tiso_orig = [r'$a$', r'$b$', r'$a_f$', r'$b_f$']
	in_names_full_rpar = [r'$q_1$', r'$q_2$', r'$q_3$', r'$q_4$']
	in_names_full_orig = [r'$a$', r'$b$', r'$a_f$', r'$b_f$', r'$a_s$', r'$b_s$', r'$a_{f}$', r'$b_{fs}$']
	
	in_names = None
	if(('tiso' in basedir) and ('rpar' in basedir)):
		in_names = in_names_tiso_rpar
	elif(('tiso' in basedir) and ('orig' in basedir)):
		in_names = in_names_tiso_orig
	elif(('full' in basedir) and ('rpar' in basedir)):
		in_names = in_names_full_rpar
	elif(('full' in basedir) and ('orig' in basedir)):
		in_names = in_names_full_orig	
	
	# plot sobol indices
	sobol(sens_m, sens_t, in_names)