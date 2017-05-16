import numpy as np
from histogram import histogram
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
data_template = np.load('/data/datasets/CTA/pulses/templates_input.npz')

def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1,k+1,npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i,k,:])
        plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)

pes = np.copy(data_template['pes'])
samples = np.copy(data_template['samples'])
templates = np.copy(data_template['templates'])

templates = templates -np.mean(templates[:,0:219:1],axis=1)[:,None]

templates_by_sample = np.copy(templates)
templates_by_sample = np.swapaxes(templates_by_sample,0,1)
hist_per_pe = histogram(data=templates, bin_centers=samples)
hist_per_sample = histogram(data=templates_by_sample, bin_centers=pes)
plt.ion()

spl = []
# get the knots from the most difficult one
spl_0 = UnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[10][220:511:1], s=0,k=3)

'''
spl_0 = []
for i,pe in enumerate(pes):
    spl_0.append(UnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[i][220:511:1], s=0,k=3))
    print(pe,len(spl_0[-1].get_knots()))
'''
#plt.figure(0)
#plt.step(hist_per_pe.bin_centers[220:511:1],hist_per_pe.data[-1][220:511:1])
#plt.plot(xs, spl_0(xs), 'k-', lw=1)

knt0 = spl_0.get_knots()
wei0 = spl_0.get_coeffs()
print(knt0)

#knt0 = hist_per_pe.bin_centers[221:510:1]
#print(knt0)
#print(5./0)
list_coef = []
for i,pe in enumerate(pes):
    if i>99:continue
    #t = hist_per_pe.bin_centers[236:509:2]
    spl.append( LSQUnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[i][220:511:1], knt0[1:-1]))
    list_coef.append(list(spl[-1].get_coeffs()))


# Get the knots

xs1 = np.linspace(0, 17000, 10000)
fig1,ax1 = plt.subplots(10,10)
list_coef = np.swapaxes(list_coef,0,1)
spl_1 = []
for i in range(len(list_coef)):
    spl_1.append( UnivariateSpline(pes[:-1],list_coef[i] , s=0, k=3))
    if i>99:continue
    plt.subplot(10, 10,i+1)
    plt.step(pes[:-1],list_coef[i])
    plt.plot(xs1, spl_1[-1](xs1), 'k-', lw=3)


xs = np.linspace(hist_per_pe.bin_centers[220:400:1][0], hist_per_pe.bin_centers[220:400:1][-1], 1000)
fig,ax = plt.subplots(1,1,figsize=(8,6))
for i,pe in enumerate(pes):
    if i!=99:continue
    plt.subplot(1, 1 ,1)
    ax.xlabel('ADC')
    ax.ylabel('A.U.')
    plt.ylim(-400.,4000.)
    plt.xlim(900.,1600.)
    ax.errorbar(hist_per_pe.bin_centers[220:400:1],hist_per_pe.data[i][220:400:1],fmt='ok',label='Measured Shape ($N_{\gamma}=16366$)')
    #plt.plot(xs, spl[i](xs), 'g-', lw=2,label='1D Spline')
    #generate list of coefficient
    coeffs = []
    for coef in range(len(list_coef)):
        coeffs.append(spl_1[coef](float(pe)))
    y_eval = bspleval(xs, knt0, np.array(coeffs), 3, debug=False)
    y_eval[0:-17]=y_eval[16:-1]
    plt.plot(xs, y_eval, 'r--', lw=2,label='$f(N_{\gamma})$')
    plt.legend()


plt.show()


def myspline(pe):
    xs = np.linspace(hist_per_pe.bin_centers[220:400:1][0], hist_per_pe.bin_centers[220:400:1][-1], 1000)
    coeffs = []
    for coef in range(len(list_coef)):
        coeffs.append(spl_1[coef](float(pe)))
    return bspleval(xs, knt0, np.array(coeffs), 3, debug=False)