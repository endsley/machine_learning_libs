import autograd.numpy as np
import copy
from sklearn.cluster import KMeans
import csv


def load_cg_merged(filename):
    csvfile = open(filename,"rb")
    csvreader = csv.reader(csvfile)
    lines = [line for line in csvreader]
    csvfile.close()
    
    # number of samples
    n_instances = len(lines) - 1

    # number of features
    n_features = len(lines[0]) - 1

    # extract feature names
    feature_name = lines[0][1:]

    # extract case ids
    case_id = []

    # extract dataset
    data = np.empty((n_instances,n_features),dtype=list)
    for i in range(n_instances):
        case_id.append(lines[i+1][0])
        data[i,:] = lines[i+1][1:]
    
    return (feature_name,case_id,data)

def unpackParam(param, N, D, G, M, K):
    """ This function unpack the vector-shaped parameter to separate variables,
    including those described in objective.py
    
    1) tau_a1: len(M), first parameter of q(alpha_m)
    2) tau_a2: len(M), second parameter of q(alpha_m)
    3) tau_b1: len(M), first parameter of q(beta_m)
    4) tau_b2: len(M), second parameter of q(beta_m)
    5) phi: shape(M, G), phi[m,:] is the paramter vector of q(c_m)
    6) tau_v1: len(G), first parameter of q(nu_g)
    7) tau_v2: len(G), second parameter of q(nu_g)
    8) mu_w: shape(G, D, K), mu_w[g,d,k] is the mean parameter of 
        q(W^g_{dk})
    9) sigma_w: shape(G, D, K), sigma_w[g,d,k] is the std parameter of 
        q(W^g_{dk})
    10) mu_b: shape(G, K), mu_b[g,k] is the mean parameter of q(b^g_k)
    11) sigma_b: shape(G, K), sigma_b[g,k] is the std parameter of q(b^g_k)
    """
    
    tp_1 = [0, M, 2*M, 3*M, 4*M, 4*M+M*G, 4*M+M*G+G, 
            4*M+M*G+2*G, 4*M+M*G+2*G+G*D*K, 4*M+M*G+2*G+G*(2*D)*K,
            4*M+M*G+2*G+G*(2*D+1)*K, 4*M+M*G+2*G+G*(2*D+2)*K]
    tp_2 = []
    for i in np.arange(len(tp_1)-1):
        tp_2.append(param[tp_1[i] : tp_1[i+1]])
    [tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, sigma_w,\
            mu_b, sigma_b] = tp_2
    phi = np.reshape(phi, (M,G))
    mu_w = np.reshape(mu_w, (G,D,K))
    sigma_w = np.reshape(sigma_w, (G,D,K))
    mu_b = np.reshape(mu_b, (G,K))
    sigma_b = np.reshape(sigma_b, (G,K))

    return(tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, \
            sigma_w, mu_b, sigma_b)

def projectSimplex_vec(v):
    """ project vector v onto the probability simplex
    Parameter
    ---------
    v: shape(nVars,)
        input vector

    Returns
    -------
    w: shape(nVars,)
        projection of v onto the probability simplex
    """

    nVars = v.shape[0]
    mu = np.sort(v,kind='quicksort')[::-1]
    sm_hist = np.cumsum(mu)
    flag = (mu - 1./np.arange(1,nVars+1)*(sm_hist-1) > 0)
    
    lastTrue = len(flag) - 1 - flag[::-1].argmax()
    sm_row = sm_hist[lastTrue]
     
    theta = 1./(lastTrue+1) * (sm_row - 1)
    w = np.maximum(v-theta, 0.)
    return w

def projectSimplex(mat):
    """ project each row vector to the simplex
    """
    nPoints, nVars = mat.shape
    mu = np.fliplr(np.sort(mat, axis=1))
    sum_hist = np.cumsum(mu, axis=1)
    flag = (mu - 1./np.tile(np.arange(1,nVars+1),(nPoints,1))*(sum_hist-1) > 0)
    
    f_flag = lambda flagPoint: len(flagPoint) - 1 - \
            flagPoint[::-1].argmax()
    lastTrue = map(f_flag, flag)
    
    sm_row = sum_hist[np.arange(nPoints), lastTrue]
    
    theta = (sm_row - 1)*1./(np.array(lastTrue)+1.)
    
    w = np.maximum(mat - np.tile(theta, (nVars,1)).T, 0.)
    
    return w

def projectLB(v, lb):
    """ project vector v onto constraint v >= lb, used for nonnegative
    constraint
    Parameter
    ---------
    v: shape(nVars,)
        input vector
    lb: float
        lower bound

    Return
    ------
    w: shape(nVars,)
        projection of v to constraint v >= lb
    """
    return np.maximum(v, lb)

def projectParam(param, N, D, G, M, K, lb=1e-6):
    """ project variational parameter vector onto the constraint set, including
    positive constraints for parameters of Beta distributions, simplex
    constraints for parameters of Categorical distributions
    
    Parameters
    ----------
    param: length (2M + 2M + MG + 2G + GDK + GDK + GK + GK) 
        variational parameters, including:
        1) tau_a1: len(M), first parameter of q(alpha_m)
        2) tau_a2: len(M), second parameter of q(alpha_m)
        3) tau_b1: len(M), first parameter of q(beta_m)
        4) tau_b2: len(M), second parameter of q(beta_m)
        5) phi: shape(M, G), phi[m,:] is the paramter vector of q(c_m)
        6) tau_v1: len(G), first parameter of q(nu_g)
        7) tau_v2: len(G), second parameter of q(nu_g)
        8) mu_w: shape(G, D, K), mu_w[g,d,k] is the mean parameter of 
            q(W^g_{dk})
        9) sigma_w: shape(G, D, K), sigma_w[g,d,k] is the std parameter of 
            q(W^g_{dk})
        10) mu_b: shape(G, K), mu_b[g,k] is the mean parameter of q(b^g_k)
        11) sigma_b: shape(G, K), sigma_b[g,k] is the std parameter of q(b^g_k)
    N,D,G,M,K: number of samples (N), features(D), groups(G), experts(M),
        clusters(K)
    lb: float, lower bound of positive constraints
     
    Returns
    -------
    w: length (2M + 2M + MG + 2G + GNK + GDK + GDK + GK + GK) 
    """
    # unpack the input parameter vector
    tp_1 = [0, M, 2*M, 3*M, 4*M, 4*M+M*G, 4*M+M*G+G, 
            4*M+M*G+2*G, 4*M+M*G+2*G+G*D*K, 4*M+M*G+2*G+G*(2*D)*K,
            4*M+M*G+2*G+G*(2*D+1)*K, 4*M+M*G+2*G+G*(2*D+2)*K]
    tp_2 = []
    for i in np.arange(len(tp_1)-1):
        tp_2.append(param[tp_1[i] : tp_1[i+1]])
    [tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, sigma_w,\
            mu_b, sigma_b] = tp_2
    phi = np.reshape(phi, (M,G))
     
    # apply projections
    w_tau_ab = projectLB(np.concatenate((tau_a1,tau_a2,tau_b1,tau_b2)), lb)
    
    w_phi_vec = np.reshape(projectSimplex(phi), M*G)

    w_tau_v = projectLB(np.concatenate((tau_v1,tau_v2)), lb)
    
    w = np.concatenate((w_tau_ab, w_phi_vec, w_tau_v, \
            mu_w, projectLB(sigma_w,lb), mu_b, projectLB(sigma_b,lb)))
    return w

def projectParam_vec(param, N, D, G, M, K, lb=1e-6):
    # unpack the input parameter vector
    tp_1 = [0, M, 2*M, 3*M, 4*M, 4*M+M*G, 4*M+M*G+G, 4*M+M*G+2*G, 
            4*M+M*G+2*G+G*N*K, 4*M+M*G+2*G+G*(N+D)*K, 4*M+M*G+2*G+G*(N+2*D)*K,
            4*M+M*G+2*G+G*(N+2*D+1)*K, 4*M+M*G+2*G+G*(N+2*D+2)*K]
    tp_2 = []
    for i in np.arange(len(tp_1)-1):
        tp_2.append(param[tp_1[i] : tp_1[i+1]])
    [tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, eta, mu_w, sigma_w,\
            mu_b, sigma_b] = tp_2
    phi = np.reshape(phi, (M,G))
    eta = np.reshape(eta, (G,N,K))
    
    # apply projections
    w_tau_ab = projectLB(np.concatenate((tau_a1,tau_a2,tau_b1,tau_b2)), lb)
    w_phi = np.zeros((M,G))
    for m in np.arange(M):
        w_phi[m] = projectSimplex_vec(phi[m])
    w_tau_v = projectLB(np.concatenate((tau_v1,tau_v2)), lb)

    w_eta = np.zeros((G,N,K))
    for g in np.arange(G):
        for n in np.arange(N):
            w_eta[g,n] = projectSimplex_vec(eta[g,n])

    w = np.concatenate((w_tau_ab, w_phi.reshape(M*G), w_tau_v, \
            w_eta.reshape(G*N*K), mu_w, projectLB(sigma_w,lb), mu_b, \
            projectLB(sigma_b,lb)))
    return w

def isLegal(v):
    return np.sum(np.any(np.imag(v)))==0 and np.sum(np.isnan(v))==0 and \
            np.sum(np.isinf(v))==0

def lbfgsUpdate(y, s, corrections, debug, old_dirs, old_stps, Hdiag):
    """ This function implements the update formula of L-BFGS
    """
    ys = np.dot(y,s)
    if ys > 1e-10:
        numCorrections = old_dirs.shape[1]
        if numCorrections < corrections:
            # full update
            new_dirs = np.hstack((old_dirs, s.reshape(s.size,1)))
            new_stps = np.hstack((old_stps, y.reshape(y.size,1)))
        else:
            # limited-momory update
            new_dirs = np.hstack((old_dirs[:,1:corrections], \
                    s.reshape(s.size,1)))
            new_stps = np.hstack((old_stps[:,1:corrections], \
                    y.reshape(y.size,1)))
        new_Hdiag = ys/np.dot(y,y)
    else:
        if debug == True:
            print "Skipping update"
        (new_dirs, new_stps, new_Hdiag) = (old_dirs, old_stps, Hdiag)

    return (new_dirs, new_stps, new_Hdiag)

def polyinterp(points, doPlot=None, xminBound=None, xmaxBound=None):
    """ polynomial interpolation
    Parameters
    ----------
    points: shape(pointNum, 3), three columns represents x, f, g
    doPolot: set to 1 to plot, default 0
    xmin: min value that brackets minimum (default: min of points)
    xmax: max value that brackets maximum (default: max of points)
    
    set f or g to sqrt(-1)=1j if they are not known
    the order of the polynomial is the number of known f and g values minus 1

    Returns
    -------
    minPos:
    fmin:
    """
    
    if doPlot == None:
        doPlot = 0

    nPoints = points.shape[0]
    order = np.sum(np.imag(points[:, 1:3]) == 0) -1
    
    # code for most common case: cubic interpolation of 2 points
    if nPoints == 2 and order == 3 and doPlot == 0:
        [minVal, minPos] = [np.min(points[:,0]), np.argmin(points[:,0])]
        notMinPos = 1 - minPos
        d1 = points[minPos,2] + points[notMinPos,2] - 3*(points[minPos,1]-\
                points[notMinPos,1])/(points[minPos,0]-points[notMinPos,0])

        t_d2 =  d1**2 - points[minPos,2]*points[notMinPos,2]
        if t_d2 > 0:
            d2 = np.sqrt(t_d2)
        else:
            d2 = np.sqrt(-t_d2) * np.complex(0,1)
        if np.isreal(d2):
            t = points[notMinPos,0] - (points[notMinPos,0]-points[minPos,0])*\
                    ((points[notMinPos,2]+d2-d1)/(points[notMinPos,2]-\
                    points[minPos,2]+2*d2))
            minPos = np.min([np.max([t,points[minPos,0]]), points[notMinPos,0]])
        else:
            minPos = np.mean(points[:,0])
        fmin = minVal
        return (minPos, fmin)
    
    xmin = np.min(points[:,0])
    xmax = np.max(points[:,0])

    # compute bounds of interpolation area
    if xminBound == None:
        xminBound = xmin
    if xmaxBound == None:
        xmaxBound = xmax

    # constraints based on available function values
    A = np.zeros((0, order+1))
    b = np.zeros((0, 1))
    for i in range(nPoints):
        if np.imag(points[i,1]) == 0:
            constraint = np.zeros(order+1)
            for j in np.arange(order,-1,-1):
                constraint[order-j] = points[i,0]**j
            A = np.vstack((A, constraint))
            b = np.append(b, points[i,1])
    
    # constraints based on availabe derivatives
    for i in range(nPoints):
        if np.isreal(points[i,2]):
            constraint = np.zeros(order+1)
            for j in range(1,order+1):
                constraint[j-1] = (order-j+1)* points[i,0]**(order-j)
            A = np.vstack((A, constraint))
            b = np.append(b,points[i,2])
    
    # find interpolating polynomial
    params = np.linalg.solve(A, b)

    # compute critical points
    dParams = np.zeros(order)
    for i in range(params.size-1):
        dParams[i] = params[i] * (order-i)
    
    if np.any(np.isinf(dParams)):
        cp = np.concatenate((np.array([xminBound, xmaxBound]), points[:,0]))
    else:
        cp = np.concatenate((np.array([xminBound, xmaxBound]), points[:,0], \
                np.roots(dParams)))
    
    # test critical points
    fmin = np.infty;
    minPos = (xminBound + xmaxBound)/2.
    for xCP in cp:
        if np.imag(xCP) == 0 and xCP >= xminBound and xCP <= xmaxBound:
            fCP = np.polyval(params, xCP)
            if np.imag(fCP) == 0 and fCP < fmin:
                minPos = np.double(np.real(xCP))
                fmin = np.double(np.real(fCP))
    
    # plot situation (omit this part for now since we are not going to use it
    # anyway)

    return (minPos, fmin)

def subHv(p, x, g, HvFunc):
    d = p - x
    Hd = HvFunc(d)
    f = np.dot(g,d) + 0.5*np.dot(d,Hd)
    g = g+ Hd
    return (f,g)

def setDefaultOptions(options_input, options_default):
    if options_input == None:
        options_output = options_default
    else:
        options_output = copy.deepcopy(options_input)
        for item in options_default.keys():
            if item not in options_input.keys():
                options_output[item] = options_default[item]
    return options_output

def minConf_SPG(funObj, x, funProj, options=None):
    """ This function implements Mark Schmidt's MATLAB implementation of
    spectral projected gradient (SPG) to solve for projected quasi-Newton
    direction
                min funObj(x) s.t. x in C
    Parameters
    ----------
    funObj: function that returns objective function value and the gradient
    x: initial parameter value
    funProj: fcuntion that returns projection of x onto C
    options:
        verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:
            debug)
        optTol: tolerance used to check for optimality (default: 1e-5)
        progTol: tolerance used to check for lack of progress (default: 1e-9)
        maxIter: maximum number of calls to funObj (default: 500)
        numDiff: compute derivatives numerically (0: use user-supplied
            derivatives (default), 1: use finite differences, 2: use complex
            differentials)
        suffDec: sufficient decrease parameter in Armijo condition (default
            : 1e-4)
        interp: type of interpolation (0: step-size halving, 1: quadratic,
            2: cubic)
        memory: number of steps to look back in non-monotone Armijo
            condition
        useSpectral: use spectral scaling of gradient direction (default:
            1)
        curvilinear: backtrack along projection Arc (default: 0)
        testOpt: test optimality condition (default: 1)
        feasibleInit: if 1, then the initial point is assumed to be
            feasible
        bbType: type of Barzilai Borwein step (default: 1)
 
    Notes: 
        - if the projection is expensive to compute, you can reduce the
            number of projections by setting testOpt to 0
    """
    
    nVars = x.shape[0]
    options_default = {'verbose':2, 'numDiff':0, 'optTol':1e-5, 'progTol':1e-9,\
                'maxIter':500, 'suffDec':1e-4, 'interp':2, 'memory':10,\
                'useSpectral':1,'curvilinear':0,'feasibleInit':0,'testOpt':1,\
                'bbType':1}
    options = setDefaultOptions(options, options_default)

    if options['verbose'] >= 2:
        if options['testOpt'] == 1:
            print '{:10s}'.format('Iteration') + \
                    '{:10s}'.format('FunEvals') + \
                    '{:10s}'.format('Projections') + \
                    '{:15s}'.format('StepLength') + \
                    '{:15s}'.format('FunctionVal') + \
                    '{:15s}'.format('OptCond')
        else:
            print '{:10s}'.format('Iteration') + \
                    '{:10s}'.format('FunEvals') + \
                    '{:10s}'.format('Projections') + \
                    '{:15s}'.format('StepLength') + \
                    '{:15s}'.format('FunctionVal')
    
    funEvalMultiplier = 1

    # evaluate initial point
    if options['feasibleInit'] == 0:
        x = funProj(x)
    [f, g] = funObj(x)
    projects = 1
    funEvals = 1

    # optionally check optimality
    if options['testOpt'] == 1:
        projects = projects + 1
        if np.max(np.abs(funProj(x-g)-x)) < options['optTol']:
            if options['verbose'] >= 1:
                print "First-order optimality conditions below optTol at initial point"
            return (x, f, funEvals, projects)
    
    i = 1
    while funEvals <= options['maxIter']:
        # compute step direction
        if i == 1 or options['useSpectral'] == 0:
            alpha = 1.
        else:
            y = g - g_old
            s = x - x_old
            if options['bbType'] == 1:
                alpha = np.dot(s,s)/np.dot(s,y)
            else:
                alpha = np.dot(s,y)/np.dot(y,y)
            if alpha <= 1e-10 or alpha >= 1e10:
                alpha = 1.
        
        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        # compute projected step
        if options['curvilinear'] == 0:
            d = funProj(x+d) - x
            projects = projects + 1

        # check that progress can be made along the direction
        gtd = np.dot(g, d)
        if gtd > -options['progTol']:
            if options['verbose'] >= 1:
                print "Directional derivtive below progTol"
            break

        # select initial guess to step length
        if i == 1:
            t = np.minimum(1., 1./np.sum(np.abs(g)))
        else:
            t = 1.

        # compute reference function for non-monotone condition
        if options['memory'] == 1:
            funRef = f
        else:
            if i == 1:
                old_fvals = np.ones(options['memory'])*(-1)*np.infty
            
            if i <= options['memory']:
                old_fvals[i-1] = f
            else:
                old_fvals = np.append(old_fvals[1:], f)
            funRef = np.max(old_fvals)
        
        # evaluate the objective and gradient at the initial step
        if options['curvilinear'] == 1:
            x_new = funProj(x + t*d)
            projects = projects + 1
        else:
            x_new = x + t*d
        [f_new, g_new] = funObj(x_new)
        funEvals = funEvals + 1

        # Backtracking line search
        lineSearchIters = 1
        while f_new > funRef + options['suffDec']*np.dot(g,x_new-x) or \
                isLegal(f_new) == False:
            temp = t
            if options['interp'] == 0 or isLegal(f_new) == False:
                if options['verbose'] == 3:
                    print 'Halving step size'
                t = t/2.
            elif options['interp'] == 2 and isLegal(g_new):
                if options['verbose'] == 3:
                    print "Cubic Backtracking"
                t = polyinterp(np.array([[0,f,gtd],\
                        [t,f_new,np.dot(g_new,d)]]))[0]
            elif lineSearchIters < 2 or isLegal(f_prev):
                if options['verbose'] == 3:
                    print "Quadratic Backtracking"
                t = polyinterp(np.array([[0, f, gtd],\
                        [t, f_new, np.complex(0,1)]]))[0]
            else:
                if options['verbose'] == 3:
                    print "Cubic Backtracking on Function Values"
                t = polyinterp(np.array([[0., f, gtd],\
                                         [t,f_new,np.complex(0,1)],\
                                         [t_prev,f_prev,np.complex(0,1)]]))[0]
            # adjust if change is too small
            if t < temp*1e-3:
                if options['verbose'] == 3:
                    print "Interpolated value too small, Adjusting"
                t = temp * 1e-3
            elif t > temp * 0.6:
                if options['verbose'] == 3:
                    print "Interpolated value too large, Adjusting"
                t = temp * 0.6

            # check whether step has become too small
            if np.max(np.abs(t*d)) < options['progTol'] or t == 0:
                if options['verbose'] == 3:
                    print "Line Search failed"
                t = 0.
                f_new = f
                g_new = g
                break
            
            # evaluate new point
            f_prev = f_new
            t_prev = temp
            if options['curvilinear'] == True:
                x_new = funProj(x + t*d)
                projects = projects + 1
            else:
                x_new = x + t*d
            [f_new, g_new] = funObj(x_new)
            funEvals = funEvals + 1
            lineSearchIters = lineSearchIters + 1
        
        # done with line search

        # take step
        x = x_new
        f = f_new
        g = g_new

        if options['testOpt'] == True:
            optCond = np.max(np.abs(funProj(x-g)-x))
            projects = projects + 1

        # output log
        if options['verbose'] >= 2:
            if options['testOpt'] == True:
                print '{:10d}'.format(i) + \
                      '{:10d}'.format(funEvals*funEvalMultiplier) + \
                      '{:10d}'.format(projects) + \
                      '{:15.5e}'.format(t) + \
                      '{:15.5e}'.format(f) + \
                      '{:15.5e}'.format(optCond)
            else:
                print '{:10d}'.format(i) + \
                      '{:10d}'.format(funEvals*funEvalMultiplier) + \
                      '{:10d}'.format(projects) + \
                      '{:15.5e}'.format(t) + \
                      '{:15.5e}'.format(f)        
        # check optimality
        if options['testOpt'] == True:
            if optCond < options['optTol']:
                if options['verbose'] >= 1:
                    print "First-order optimality conditions below optTol"
                break

        if np.max(np.abs(t*d)) < options['progTol']:
            if options['verbose'] >= 1:
                print "Step size below progTol"
            break
        
        if np.abs(f-f_old) < options['progTol']:
            if options['verbose'] >= 1:
                print "Function value changing by less than progTol"
            break

        if funEvals*funEvalMultiplier > options['maxIter']:
            if options['verbose'] >= 1:
                print "Function evaluation exceeds maxIter"
            break

        i = i + 1

    return (x, f, funEvals, projects)

def solveSubProblem(x, g, HvFunc, funProj, optTol, progTol, maxIter, testOpt,\
        feasibleInit, x_init):
    """
    use SPG (spectral projected gradient) method to solve for projected
    quasi-Newton direction
    """

    funObj = lambda p: subHv(p, x, g, HvFunc)
    options = {'verbose':0, 'optTol':optTol, 'progTol':progTol, \
            'maxIter':maxIter, 'testOpt':testOpt, 'feasibleInit':feasibleInit}
    [p,f,funEvals,subProjects] = minConf_SPG(funObj, x_init, funProj, options)
    return (p, subProjects)

def genConstraints(prng, label, alpha, beta, num_ML, num_CL, start_expert = 0, \
        flag_same=False):
    """ This function generates pairwise constraints (ML/CL) using groud-truth
    cluster label and noise parameters
    Parameters
    ----------
    label: shape(n_sample, )
        cluster label of all the samples
    alpha: shape(n_expert, )
        sensitivity parameters of experts
    beta: shape(n_expert, )
        specificity parameters of experts
    num_ML: int
    num_CL: int
    flag_same: True if different experts provide constraints for the same set
    of sample pairs, False if different experts provide constraints for
    different set of sample pairs
    
    Returns
    -------
    S: shape(n_con, 4)
        The first column -> expert id
        The second and third column -> (row, column) indices of two samples
        The fourth column -> constraint values (1 for ML and 0 for CL)
    """
    n_sample = len(label)
    tp = np.tile(label, (n_sample,1))
    label_mat = (tp == tp.T).astype(int)
    
    ML_set = []
    CL_set = []
    # get indices of upper-triangle matrix
    [row, col] = np.triu_indices(n_sample, k=1)
    # n_sample * (n_sample-1)/2
    for idx in range(len(row)):
        if label_mat[row[idx],col[idx]] == 1:
            ML_set.append([row[idx], col[idx]])
        elif label_mat[row[idx],col[idx]] == 0:
            CL_set.append([row[idx], col[idx]])
        else:
            print "Invalid matrix entry values"

    ML_set = np.array(ML_set)
    CL_set = np.array(CL_set)

    assert num_ML < ML_set.shape[0]
    assert num_CL < CL_set.shape[0]
    
    # generate noisy constraints for each expert
    assert len(alpha) == len(beta)
    n_expert = len(alpha)
    
    # initialize the constraint matrix
    S = np.zeros((0, 4))
    
    # different experts provide constraint for the same set of sample pairs
    if flag_same == True:
        idx_ML = prng.choice(ML_set.shape[0], num_ML, replace=False)
        idx_CL = prng.choice(CL_set.shape[0], num_CL, replace=False)
        ML = ML_set[idx_ML, :]
        CL = CL_set[idx_CL, :]
        for m in range(n_expert):
            val_ML = prng.binomial(1, alpha[m], num_ML)
            val_CL = prng.binomial(1, 1-beta[m], num_CL)
            Sm_ML = np.hstack((np.ones((num_ML,1))*(m+start_expert), ML, \
                    val_ML.reshape(val_ML.size,1) ))
            Sm_CL = np.hstack((np.ones((num_CL,1))*(m+start_expert), CL, \
                    val_CL.reshape(val_CL.size,1) ))
            S = np.vstack((S, Sm_ML, Sm_CL)).astype(int)
    # different experts provide constraints for different sets of sample pairs
    else:
        for m in range(n_expert):
            prng = np.random.RandomState(1000 + m)
            idx_ML = prng.choice(ML_set.shape[0], num_ML, replace=False)
            idx_CL = prng.choice(CL_set.shape[0], num_CL, replace=False)
            ML = ML_set[idx_ML, :]
            CL = CL_set[idx_CL, :]
            val_ML = prng.binomial(1, alpha[m], num_ML)
            val_CL = prng.binomial(1, 1-beta[m], num_CL)
            Sm_ML = np.hstack((np.ones((num_ML,1))*(m+start_expert), ML, \
                    val_ML.reshape(val_ML.size,1) ))
            Sm_CL = np.hstack((np.ones((num_CL,1))*(m+start_expert), CL, \
                    val_CL.reshape(val_CL.size,1) ))
            S = np.vstack((S, Sm_ML, Sm_CL)).astype(int)

    return S

def initParam(prior, X, N, D, G, M, K, dir_param, prng):
    """ initialize variational parameters with prior parameters
    """
    
    [tpM, tpG, lb, ub] = [np.ones(M), np.ones(G), 10., 10.]
    tpR = prng.rand(2*M)
    [tau_a1, tau_a2, tau_b1, tau_b2, tau_v1, tau_v2] = \
            [lb+(ub-lb)*tpR[0 : M], tpM,\
             lb+(ub-lb)*tpR[M : 2*M], tpM, \
             tpG, tpG]

    mu_w = prng.randn(G,D,K)/np.sqrt(D)
    sigma_w = np.ones(G*D*K) * 1e-3
    mu_b = prng.randn(G, K)/np.sqrt(D)
    sigma_b = np.ones(G*K) * 1e-3

    phi = np.reshape(prng.dirichlet(np.ones(G)*dir_param, M), M*G)
    
    mu_w = np.reshape(mu_w, G*D*K)
    mu_b = np.reshape(mu_b, G*K)

    param_init = np.concatenate((tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1,\
        tau_v2, mu_w, sigma_w, mu_b, sigma_b))
    
    return param_init

def my_spectral_clustering(sim_mat, n_clusters=2):
    
    N = sim_mat.shape[0]
    sim_mat = sim_mat - np.diag(np.diag(sim_mat))
    t1 = 1./np.sqrt(np.sum(sim_mat, axis=1))
    t2 = np.dot(t1.reshape(N,1), t1.reshape(1,N))

    lap_mat = np.eye(N) - sim_mat * t2
    eig_val, eig_vec = np.linalg.eig(lap_mat)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = np.real(eig_vec[:, idx])
    
    t3 = np.diag(np.sqrt(1./np.sum(eig_vec[:,0:n_clusters]**2, axis=1)))
    embd = np.dot(t3, eig_vec[:, 0:n_clusters])

    clf = KMeans(n_clusters = n_clusters, n_jobs=-1)
    label_pred = clf.fit_predict(embd)
    
    return label_pred

if __name__ == "__main__":
    from scipy.io import savemat
    import time
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import scale
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.metrics import normalized_mutual_info_score as nmi

    flag_test = 10
    if flag_test == 1:
        V = np.random.randn(2, 100)
        v0 = V[0]
        w0 = projectLB(v0, 0) 
        
        w = projectSimplex(v0) 
        W = projectSimplex_mat(V)
        
        print np.linalg.norm(W[0] - w)

        #np.savetxt("v0.csv",v0,delimiter=',')
        #np.savetxt("w1.csv",w1,delimiter=',')
        #np.savetxt("w2.csv",w2,delimiter=',')
    elif flag_test == 2:
        #[N,D,G,M,K] = [10,2,3,10,2]
        [N,D,G,M,K] = [5000, 50, 20, 100, 10]
        param = np.random.randn( M*4 + G*(M+2+K*2+N) + G*K*(N+D*2) )
        funProj_vec = lambda param: projectParam_vec(param,N,D,G,M,K,lb=1e-6)
        funProj_mat = lambda param: projectParam(param,N,D,G,M,K,lb=1e-6)
        t1 = time.time()
        w_loop = funProj_vec(param)
        t2 = time.time()
        print "vector + loop: ", t2-t1
        w_map = funProj_mat(param)
        t3 = time.time()
        print "mat : ", t3 - t2
        print "diff: ", np.linalg.norm(w_loop - w_map)

    elif flag_test == 3:
        #v0 = np.random.randn(100)
        v0 = np.complex(0,1)
        print isLegal(v0)
    elif flag_test == 4:
        # test polyinterp
        points = np.random.randn(2, 3)
        print polyinterp(points)
        np.savetxt(\
        "./Mark_Schmidt/minConf/minFunc/test_data/polyinterp_input_1.csv",\
                points)
    elif flag_test == 5:
        # test lbfgsUpdate
        [p, m, corrections, debug, Hdiag] = [2, 5, 2, 0, 1e-3]
        y = np.random.randn(p)
        s = y + 0.1 * np.random.randn(p)
        old_dirs = np.random.randn(p, m)
        old_stps = np.random.randn(p, m)
        data_mat = {'p':p, 'm':m, 'corrections':corrections, 'debug':debug, \
                'Hdiag':Hdiag, 'y':y, 's':s, 'old_dirs':old_dirs, \
                'old_stps':old_stps}
        savemat("./Mark_Schmidt/minConf/minFunc/test_data/lbfgsUpdate_input_1.csv",\
                data_mat)
        [new_dirs, new_stps, new_Hdiag] = lbfgsUpdate(y, s, corrections, \
                debug, old_dirs, old_stps, Hdiag)
        print new_dirs 
        print new_stps
        print new_Hdiag
    elif flag_test == 6:
        # test subHv
        dim = 5
        m = 2
        p = np.random.randn(dim)
        x = np.random.randn(dim)
        g = np.random.randn(dim)
        Hdiag = 1e-3
        N = np.random.randn(dim, m)
        M = np.random.randn(m, m)
        data_mat = {'dim':dim, 'm':2, 'p':p, 'x':x, 'g':g, 'Hdiag':Hdiag, \
                'N':N, 'M':M}
        savemat("./Mark_Schmidt/minConf/minFunc/test_data/subHv_input_1.csv",\
                    data_mat)
        HvFunc = lambda v: v/Hdiag - np.dot(N,np.linalg.solve(M,np.dot(N.T,v)))
        [f, g] = subHv(p, x, g, HvFunc)
        print f
        print g
    elif flag_test == 7:
        # test function minConf_SPG, use examples in example_minConf.m
        nInst = 1000
        nVars = 100
        A = np.random.randn(nInst, nVars)
        x = np.random.rand(nVars) * (np.random.rand(nVars)>0.5)
        b = np.dot(A,x) + np.random.randn(1)
         
        def SquaredError(w, X, y):
            res =  np.dot(X,w) - y
            f = np.sum(res**2)
            g = 2. * np.dot(X.T, res)
            return (f, g)
        
        funObj = lambda x: SquaredError(x, A, b)
        funProj = lambda v: projectLB(v, 0)
        x_init = np.zeros(nVars)
        options = {'verbose':3}
        
        data_mat = {'nInst':nInst, 'nVars':nVars, 'A':A, 'x':x, 'b':b, \
                'x_init':x_init}
        savemat("./Mark_Schmidt/minConf/minConf_SPG_input.mat",data_mat)
        (x, f, funEvals, projects) = minConf_SPG(funObj, x_init, funProj, options)
    elif flag_test == 8:
        options_default = {'verbose':2, 'numDiff':0, 'optTol':1e-5, 'progTol':1e-9,\
                'maxIter':500, 'suffDec':1e-4, 'interp':2, 'memory':10,\
                'useSpectral':1,'curvilinear':0,'feasibleInit':0,'testOpt':1,\
                'bbType':1}
        options = {'verbose':100, 'interp':10}
        options = setDefaultOptions(options, options_default)
    elif flag_test == 9:
        label = np.random.randint(0,10,100)
        alpha = np.array([.95, .85])
        beta = np.array([.7, .55])
        [num_ML, num_CL] = [100, 100]
        S = genConstraints(label, alpha, beta, num_ML, num_CL)

    elif flag_test == 10:
        tp = load_iris()
        [X, Y] = [scale(tp['data']), tp['target']]
        sim_mat = rbf_kernel(X)
        Y_pred = my_spectral_clustering(sim_mat, n_clusters=3)
        print nmi(Y_pred, Y)
    else:
        pass

