
import os, sys, shutil
import numpy as np
import lmfit
import matplotlib.pyplot as plt 

# -----------------------------------------------------------------------------

def klotz(x, a, b):
    """
    Klotz function
	"""
    return a*pow(x,b)

# -----------------------------------------------------------------------------

def fitmodel(x, p, a0, b0, plot_fit=False, plot_fid=0, plot_label=''):
    """
    Fitting klotz
    """
    mod = lmfit.Model(klotz)
    size = len(x)

    # parametros (e algumas aproximacoes iniciais)
    mod.set_param_hint("a", value=a0, vary=True, min=0.001, max=10000)
    mod.set_param_hint("b", value=b0, vary=True, min=0.001, max=10000)

    params = mod.make_params()
    result = mod.fit(p, params, method="leastsq", x=x)  # fitting

    if(plot_fit):
        plt.figure(figsize=(8,4))
        result.plot_fit(datafmt="o")
        plt.tight_layout()
        plt.savefig('output/fitting_%s_%03d.png' % (plot_label,plot_fid))
        #plt.show()

    mse = result.chisqr/size
    print("parameters:", result.values)
    print("chisqr:", result.chisqr)
    print("mse:", mse)    

    output = np.zeros(2)
    output[0] = result.values["a"]
    output[1] = result.values["b"]
    return output

if __name__ == "__main__":

    # Teste fit def na direcao da fibra
    pres = np.linspace(0,2.7,50)
    fstr = np.array([1.00797024e-02,  1.00797024e-02,  1.80840214e-02, 2.38932486e-02,
                    2.82293531e-02,  3.16143175e-02,  3.43598509e-02,  3.66544618e-02,
                    3.86174829e-02,  4.03280644e-02,  4.18409079e-02,  4.31951680e-02,
                    4.44197166e-02,  4.55363882e-02,  4.65620572e-02,  4.75100107e-02,
                    4.83908833e-02,  4.92133089e-02,  4.99843863e-02,  5.07100180e-02,
                    5.13951614e-02,  5.20440177e-02,  5.26601770e-02,  5.32467291e-02,
                    5.38063522e-02,  5.43413815e-02,  5.48538650e-02,  5.53456082e-02,
                    5.58182104e-02,  5.62730950e-02,  5.67115336e-02,  5.71346673e-02,
                    5.75435231e-02,  5.79390290e-02,  5.83220261e-02,  5.86932787e-02,
                    5.90534839e-02,  5.94032787e-02,  5.97432468e-02,  6.00739243e-02,
                    6.03958048e-02,  6.07093438e-02,  6.10149621e-02,  6.13130497e-02,
                    6.16039683e-02,  6.18880541e-02,  6.21656201e-02,  6.24369583e-02,
                    6.27023411e-02,  6.29620233e-02])
    out = fitmodel(fstr,pres,1,1,plot_fit=True)
    print(out)

    # Teste fit vol
    vol = np.array([0.       ,  0.0509995,  0.09323232, 0.12492338, 0.14899903, 0.16795868,
                    0.1834022,  0.19633347, 0.20740205, 0.21704485, 0.22556674, 0.23318739,
                    0.24006959, 0.24633687, 0.252085,   0.25738952, 0.26231101, 0.2668987,
                    0.27119313, 0.27522802, 0.27903172, 0.28262825, 0.28603815, 0.28927913,
                    0.29236651, 0.29531366, 0.29813232, 0.30083282, 0.30342434, 0.30591502,
                    0.30831214, 0.31062226, 0.31285124, 0.3150044,  0.31708658, 0.31910213,
                    0.32105505, 0.322949,   0.32478731, 0.32657306, 0.32830907, 0.32999795,
                    0.33164211, 0.33324378, 0.33480503, 0.33632779, 0.33781385, 0.33926489,
                    0.34068245, 0.34206802])
    out = fitmodel(vol,pres,1,1,plot_fit=True)
    print(out)