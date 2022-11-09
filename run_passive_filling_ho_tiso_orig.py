import os, sys, shutil
import numpy as np
import dolfin
import ldrb
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt 
import chaospy as cp
from math import factorial
from numpy import linalg as la
import lmfit
import time
from klotz import *

"""
# Original HO Material parameters	
alpha = 0.1 					# scaling parameter to get reasonable magnitude of deformation
a = 2280.0		*alpha 			# Pa*beta [N/m^3]
a_f = 1168.5	*alpha 		 	# Pa*beta [N/m^3]
b = 9.726		*0.8		 	# dimensionless
b_f = 15.779	*0.75 		 	# dimensionless
"""
"""
# Krister Material parameters
a = 228.0	 			# Pa
b = 7.780				# dimensionless
a_f = 116.85	 		# Pa
b_f = 11.83425			# dimensionless
"""

# -----------------------------------------------------------------------------

# Compiler options
#set_log_level(50)
flags = ["-O3", "-ffast-math", "-march=native"]
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# ----------------- Import and mark mesh ----------------- #

a1 = 0.02 + 0.013 	; c1 = 0.06 + 0.01	# m (principal axes for big)
a2 = 0.02  			; c2 = 0.06 		# m (principal axes for small)

# Marking boundaries 
class Epicardium(SubDomain): 			
	def inside(self, x, on_boundry):
		return on_boundry

class Endocardium(SubDomain): 			
	def inside(self, x, on_boundry):
		return (x[0]*x[0]+x[2]*x[2] < (a2*a2)*(1.0-(x[1]*x[1])/(c2*c2)) + 1e-7) and on_boundry

class Base(SubDomain): 		
	def inside(self, x, on_boundry):
		return (x[1] > -1e-7) and on_boundry

# -----------------------------------------------------------------------------

def Make_SubDomains(mesh):
	mf = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	mf.set_all(0)
	Epicardium().mark(mf, 40)  		# Outside
	Endocardium().mark(mf, 30)		# Inside
	Base().mark(mf,10) 				# Base
	#plot(mf,interactive=True)
	for facet in facets(mesh):
		mesh.domains().set_marker((facet.index(), mf[facet]), 2)
	return mf

#------------------------------------------------------------------------------

def CreateMesh():
	# Mesh with dimensions
	mesh = Mesh('LV_mesh.xml')
	base = Base()
	epi = Epicardium()
	endo = Endocardium()

	# Set up mesh boundaries and normal
	#dx = Measure("dx", domain=mesh, subdomain_data=mf) 
	boundaries = Make_SubDomains(mesh)
	ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
	n_mesh = FacetNormal(mesh)

	return mesh, base, endo, epi, ds, n_mesh

# -----------------------------------------------------------------------------

def CreateFiberfield(mesh):

    # New create fiber field with ldrb
	markers = None
	ffun = None

    # Decide on the angles you want to use
	angles = dict(
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_epi_lv=-60,  # Fiber angle on the epicardium
        beta_endo_lv=0,  # Sheet angle on the endocardium
        beta_epi_lv=-0,
    )  # Sheet angle on the epicardium

    # Choose space for the fiber fields
    # This is a string on the form {family}_{degree}
	#fiber_space = "Lagrange_1"	
	fiber_space = "DG_0"

    # Compte the microstructure
	fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(mesh=mesh, fiber_space=fiber_space, ffun=ffun, markers=markers, **angles)

	fiber.rename("f_0","f_0")
	sheet.rename("s_0","s_0")
	sheet_normal.rename("n_0","n_0")

	return fiber, sheet, sheet_normal

# -----------------------------------------------------------------------------

def Solve_System(mysolver):

	#problem = NonlinearVariationalProblem(eq, w, bcs, J=Jac)
	#solver  = NonlinearVariationalSolver(problem)
	
	prm = mysolver.parameters
	prm['newton_solver']['absolute_tolerance'] = 1E-8
	prm['newton_solver']['relative_tolerance'] = 1E-8
	prm['newton_solver']['maximum_iterations'] = 25
	prm['newton_solver']['linear_solver'] = 'mumps'
    
	try:
		ret = mysolver.solve()
	except:
		return False
	
	return ret[1]

# -----------------------------------------------------------------------------

def Compute_Volume(u,J,F,mesh,n_mesh,ds):
	"""
	Compute LV cavity volume
	"""
	X = SpatialCoordinate(mesh)
	vol = 1e6*abs(assemble((-1.0/3.0)*dot(X + u, J*inv(F).T*n_mesh)*ds(30)))
	return vol

# -----------------------------------------------------------------------------

def sigma(u,piolak1):
	"""
	Stress tensor
	"""
	d = len(u)
	I = Identity(d)
	F = I + grad(u)
	F = variable(F)
	E = 0.5 * ((F.T*F) - I)
	J = det(F)
	pk1 = piolak1		
	sigma = (1.0/J)*pk1*F.T
	return F, sigma, E

#------------------------------------------------------------------------------

def LVPassiveFilling_tiso_orig(mesh, base, endo, epi, ds, n_mesh, f_0, s_0, n_0, 
								sampleid, a, b, a_f, b_f):
	"""
	sampleid: sample id
	a, b, a_f, b_f: Holzapfel-Ogden parameters
	"""

	# Mesh with dimensions
	#mesh = Mesh('LV_mesh.xml')
	#base = Base()
	#epi = Epicardium()
	#endo = Endocardium()

	# Set up mesh boundaries and normal
	#dx = Measure("dx", domain=mesh, subdomain_data=mf) 
	#boundaries = Make_SubDomains(mesh)
	#ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
	#n_mesh = FacetNormal(mesh)

	outname = "output/lvpassive_%03d.xdmf" % sampleid
	fileOutput = XDMFFile(mesh.mpi_comm(), outname)
	fileOutput.parameters['rewrite_function_mesh'] = False
	fileOutput.parameters["functions_share_mesh"] = True
	fileOutput.parameters["flush_output"] = True

	outname_h5 = "output/fields_lvpassive_%03d.h5" % sampleid
	fileOutput_h5 = HDF5File(mesh.mpi_comm(), outname_h5, "w")

	# Set up function spaces and check size of system
	P2 = FiniteElement("CG", mesh.ufl_cell(), 2)
	P2v = VectorElement(P2)
	P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
	W = FunctionSpace(mesh,MixedElement([P2v,P1]))
	print('Number of DOFs: %d' % W.dim())

	# Define functions
	w = Function(W)
	(u, p) = split(w)
	(du, dp) = TestFunctions(W)

	# Kinematics
	d = u.geometric_dimension()
	I = Identity(d)             # Identity tensor
	F = I + grad(u)             # Deformation gradient
	F = variable(F)             
	C = F.T*F                   # Right Cauchy-Green tensor
	B = F*F.T                   # Left Cauchy-Green tensor

	# Invariants of deformation tensors
	I_1 = tr(C)
	J = det(F)

	# HO Material parameters
	eta = 0.1
	print("HO parameters")
	print(f"a: {a}")
	print(f"b: {b}")
	print(f"af: {a_f}")
	print(f"bf: {b_f}")

	#f_0,s_0,n_0 = CreateFiberfield(mesh)	
	#f_0.rename("f_0","f_0")
	#fileOutput.write(f_0,0)
	#s_0.rename("s_0","s_0")
	#fileOutput.write(s_0,0)
	#n_0.rename("n_0","n_0")
	#fileOutput.write(n_0,0)

	fileOutput_h5.write(f_0, "/fiber")
	fileOutput_h5.write(s_0, "/sheet")
	fileOutput_h5.write(n_0, "/normal")

	f = F*f_0
	s = F*s_0
	n = F*n_0

	# Invariants
	I_4f = inner(C*f_0,f_0)
	I_4s = inner(C*s_0,s_0)
	I_4n = inner(C*n_0,n_0)

	# Initial conditions and boundary conditions
	zero_displacement = Expression(("0.0", "0.0", "0.0"), degree=1)
	bcr = DirichletBC(W.sub(0), zero_displacement, base) #boundaries, 10)
	bcs = [bcr]
	p0 = Constant(0)
	T_a = Constant(0)

	# Stress relations

	# Passive part
	passive_cauchy_stress = a*exp(b*(I_1 - 3))*B + 2*a_f*(I_4f-1)*exp(b_f*pow(I_4f - 1, 2))*outer(f,f) - p*I
	P_p = J*passive_cauchy_stress*inv(F).T

	# Active part
	active_cauchy_stress = T_a*(outer(f,f)+eta*outer(s,s)+eta*outer(n,n))	
	P_a = J*active_cauchy_stress*inv(F.T)

	P =  P_p + P_a

	eq = inner(P,grad(du))*dx + inner(J-1,dp)*dx + dot(J*inv(F).T*n_mesh*p0, du)*ds(30) 
	Jac = derivative(eq, w)

	# define o problema e o solver
	problem = NonlinearVariationalProblem(eq, w, bcs, J=Jac)
	solver  = NonlinearVariationalSolver(problem)

	# stepping and arrays
	nsteps = 60
	t = np.linspace(0,0.9,nsteps)
	dt = Constant(t[1])

	# endocardium pressure
	p_endo = 2700 # Pa	
	
	active_tension = np.zeros(nsteps)
	pressure = np.linspace(0, p_endo, nsteps)	
	volume = np.zeros(nsteps)

	f_d_matrix = np.zeros((nsteps, mesh.num_cells()))

	for i in range(len(t)):
		print( 'Now solving time-step: %d/%d' % (i,nsteps))

		# Update time-dependent parameters
		T_a.assign(active_tension[i])
		p0.assign(pressure[i])

		#Solve_System(eq, w, bcs, Jac, active_tension[i], pressure[i])
		converged = Solve_System(solver)
		
		# se der erro, a funcao LVPassiveFilling retorna None
		if not converged:
			return None
			
		# Escreve em arquivo
		uaux,paux = w.split()
		uaux.rename("u","u")
		paux.rename("p","p")

		Fnew, sig, Enew = sigma(u, P)
		
		Cnew = dot(Fnew.T, Fnew)

		sig2 = project(sig, TensorFunctionSpace(mesh, "DG", 0))
		sig2.rename("sigma", "cauchy stress tensor")
		Fnew2 = project(Fnew, TensorFunctionSpace(mesh, "DG", 0))
		Fnew2.rename("F", "deformation gradient")
		Enew2 = project(Enew, TensorFunctionSpace(mesh, "DG", 0))
		Enew2.rename("E", "lagrange strain")
		
		# compute stress and strain in fiber direction
		fiber_stress = inner(sig*f, f)
		fiber_deformation = sqrt(inner(Cnew*f_0, f_0))

		f_s = project(fiber_stress, FunctionSpace(mesh, "DG", 0))
		f_s.rename("f_s", "fiber stress")
		
		f_d = project(fiber_deformation, FunctionSpace(mesh, "DG", 0))
		f_d.rename("f_d", "fiber deformation")
		f_d_array = f_d.vector().get_local()
		f_d_matrix[i,:] = f_d_array

		# write data to XDMF output files
		fileOutput.write(uaux,i)
		fileOutput.write(paux,i)
		
		# write data to HDF5 output files
		fileOutput_h5.write(uaux, "/u", i)
		fileOutput_h5.write(paux, "/p", i)
		fileOutput_h5.write(f_s, "/fiberstress", i)
		fileOutput_h5.write(f_d, "/fiberstrain", i)
		
		volume[i] = Compute_Volume(u,J,F,mesh,n_mesh,ds)		
		print('Volume: %f' % volume[i])
		print('Pressure: %f' % pressure[i])
	
	# fim do loop
	
	fileOutput.close()
	fileOutput_h5.close()

	outvol = volume
	outpress = pressure
	outdef = f_d_matrix

	#fvol = "output/outvol_%03d.txt" % sampleid
	#fprs = "output/outpress_%03d.txt" % sampleid
	#ffdf = "output/fiber_deformation_%03d.txt" % sampleid
	#np.savetxt(fvol, outvol)
	#np.savetxt(fprs, outpress)
	#np.savetxt(ffdf, f_d_matrix, fmt = '%.10f')	

	outpress = outpress*0.001

	fvp = "output/out_press_vol_%03d.txt" % sampleid
	np.savetxt(fvp, np.transpose([outpress, outvol]))

	# random nodes
	#np.random.seed(2)
	#nelem = np.shape(outdef)[1]
	#p_rand = np.random.randint(0, nelem, size = 20)

	# fixed nodes
	p_rand = np.array([  46,  192, 1177, 1433, 1555, 1623, 2106, 2356, 2876, 3031, 
	                   3422, 3424, 3691, 4102, 4150, 4546, 4781, 4891, 5017, 5911])
	outdef_rand = outdef[:,p_rand]
	outdef_avg = np.average(outdef_rand, axis = 1)
	print("nos para medir fiber_stretch: ", p_rand)
	
	#plt.plot(outvol, outpress, 'o')
	#plt.show()
	#plt.plot(outdef_avg, outpress*0.001, 'o')
	#plt.show()
	
	# normalizacao
	vn = np.zeros(len(outvol))
	vn[:] = (outvol[:] - outvol[0])/outvol[0]

	# respostas
	print("fitting vol")
	print(vn)
	result_vol = fitmodel(vn, outpress, 1.0, 4.0, plot_fit=True, plot_fid=sampleid, plot_label='vol')

	print("fitting def")
	print(outdef_avg)
	print(outpress)
	result_def = fitmodel(outdef_avg, outpress, 0.1, 40.0, plot_fit=True, plot_fid=sampleid, plot_label='def')

	# resultados = [alfa1, beta1, alfa2, beta2, vol, def]
	res = np.zeros((6))
	res[0:2] = result_vol[:]
	res[2:4] = result_def[:]
	res[4] = volume[-1]
	res[5] = outdef_avg[-1]
	return res

# -----------------------------------------------------------------------------

if __name__ == "__main__":
	
	os.makedirs("output/", exist_ok=True)

	# Holzapfel-Ogden reference values
	a0 = 150 #228.0	 		# Pa
	b0 = 6.0 #7.780			# dimensionless
	af0 = 116.85	 		# Pa
	bf0 = 11.83425			# dimensionless

	# a0 = 150, b0 = 6.0 -> EF = 56

	a  = cp.Uniform(0.7*a0,   1.3*a0)
	b  = cp.Uniform(0.7*b0,   1.3*b0)
	af = cp.Uniform(0.7*af0, 1.3*af0)
	bf = cp.Uniform(0.7*bf0, 1.3*bf0)
	distribution = cp.J(a, b, af, bf)

	# Code for creating gPC expansion
	npar = 4        # number of input parameter  
	p = 5			# polynomial degree	
	m = 2			# multiplicative factor
	Np = int(factorial(npar+p)/(factorial(npar)*factorial(p))) # min number of terms for PCE
	Ns = m*Np		#number of samples
	Ns = 110
	
	print('inputs=', npar)
	print('grau=', p)
	print('mult=',m)
	print('Np=', Np)
	print('Ns=', Ns)

	# take samples from joint distribution
	samples = distribution.sample(Ns)

	# prepare outputs and files
	evals = []

	fd = "output/testData.txt"
	#fd = "output/trainData.txt"

	shutil.rmtree("output/") 
	os.mkdir("output/")
	f_data = open(fd,'w')
	f_data.write('# a, b, af, bf, alfa1, beta1, alfa2, beta2, vol, fibstretch \n')

	# create mesh and fiber field only once
	print('Creating mesh and fiber fields (f0,s0,n0)')
	mesh, base, endo, epi, ds, nmesh = CreateMesh()
	f0, s0, n0 = CreateFiberfield(mesh)

	# solve model for each sample
	for i in range(len(samples.T)):
		# samples - input parameters
		Z = samples.T[i,:]
		a,b,af,bf = float(Z[0]), float(Z[1]), float(Z[2]), float(Z[3])
		qin = np.array([a,b,af,bf])

		# solve model
		print("\nRodando caso %d: (%f,%f,%f,%f)" % (i,a, b, af, bf))
		out = LVPassiveFilling_tiso_orig(mesh, base, endo, epi, ds, nmesh, f0,s0, n0, i, a, b, af, bf)
		
		# senao deu erro, salva os dados
		if out is not None:
			evals.append(out)
			np.savetxt(f_data, np.concatenate((qin,out)).reshape((1,10)), fmt='%16.8e', delimiter=',')
		else:
			print("\n\nERRO na execucao da amostra\n\n")
	# end of loop in samples

	f_data.close()

	# salva os dados
	evals = np.array(evals)
	np.savetxt("output/all_evals.txt", np.transpose(evals))
	np.savetxt("output/all_samples.txt", samples)

# fim