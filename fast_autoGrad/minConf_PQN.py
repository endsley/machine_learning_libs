""" This function translates Mark Schmidt's MATLAB implementation of 
<AISTATS_2009_Optimizing Costly Functions with Simple Constraints: A
Limited-Memory Projected Quasi-Newton Algorithm> to Python.
"""

import autograd.numpy as np
from utils import projectLB
from utils import projectSimplex_vec
from utils import projectParam
from utils import isLegal
from utils import lbfgsUpdate
from utils import polyinterp
from utils import solveSubProblem
from utils import setDefaultOptions

def minConf_PQN(funObj, x, funProj, options=None):
    """
    The problems are of the form
                min funObj(x) s.t. x in C
    The projected quasi-Newton sub-problems are solved using the spectral
    projected gradient algorithm

    Parameters
    ----------
    funObj: function to minimize, return objective value as the first argument
            and gradient as the second argument
    funProj: function that returns projection of x onto C
    options:
        1) verbose: level of verbosity (0: no output, 1: final, 2: iter
        (default), 3: debug)
        2) optTol: tolerance used to check for optimality (default: 1e-5)
        3) progTol: tolerance used to check for progress (default: 1e-9)
        4) maxIter: maximum number of calls to funObj (default: 500)
        5) maxProject: maximum number of calls to funProj (default: 100000)
        6) numDiff: compute derivatives numerically (0: use user-supplied
            derivatives (default), 1: use finite differences, 2: use complex
            differentials)
        7) suffDec: sufficient decrease parameter in Armijo condition (default:
            1e-4)
        8) corrections: number of lbfgs corrections to store (default: 10)
        9) adjustStep: use quadratic initialization of line search (default: 0)
        10) bbInit: initialize sub-problem with Barzilai-Borwein step (default:
            0)
        11) SPGoptTol: optimality tolerance for SPG direction finding (default:
            1e-6)
        12) SPGiters: maximum number of iterations for SPG direction finding
            (default: 10)

    Returns
    -------
    x: optimal parameter values
    f: optimal objective value
    funEvals: number of function evaluations 
    """
    
    # number of variables/parameters
    nVars = len(x)
    
    # set default optimization settings
    options_default = {'verbose':2, 'numDiff':0, 'optTol':1e-5, 'progTol':1e-9, \
                'maxIter':500, 'maxProject':100000, 'suffDec':1e-4, \
                'corrections':10, 'adjustStep':0, 'bbInit':0, 'SPGoptTol':1e-6,\
                'SPGprogTol':1e-10, 'SPGiters':10, 'SPGtestOpt':0}
    options = setDefaultOptions(options, options_default)
    
    if options['verbose'] == 3:
        print 'Running PQN...'
        print 'Number of L-BFGS Corrections to store: ' + \
                str(options['corrections'])
        print 'Spectral initialization of SPG: ' + str(options['bbInit'])
        print 'Maximum number of SPG iterations: ' + str(options['SPGiters'])
        print 'SPG optimality tolerance: ' + str(options['SPGoptTol'])
        print 'SPG progress tolerance: ' + str(options['SPGprogTol'])
        print 'PQN optimality tolerance: ' + str(options['optTol'])
        print 'PQN progress tolerance: ' + str(options['progTol'])
        print 'Quadratic initialization of line search: ' + \
                str(options['adjustStep'])
        print 'Maximum number of function evaluations: ' + \
                str(options['maxIter'])
        print 'Maximum number of projections: ' + str(options['maxProject'])

    if options['verbose'] >= 2:
        print '{:10s}'.format('Iteration') + \
                '{:10s}'.format('FunEvals') + \
                '{:10s}'.format('Projections') + \
                '{:15s}'.format('StepLength') + \
                '{:15s}'.format('FunctionVal') + \
                '{:15s}'.format('OptCond')
    
    funEvalMultiplier = 1
    # project initial parameter vector
    # translate this function (Done!)
    x = funProj(x)
    projects = 1

    # evaluate initial parameters
    # translate this function (Done!)
    [f, g] = funObj(x)
    funEvals = 1

    # check optimality of initial point
    projects = projects + 1
    if np.max(np.abs(funProj(x-g)-x)) < options['optTol']:
        if options['verbose'] >= 1:
            print "First-Order Optimality Conditions Below optTol at Initial Point"
            return (x, f, funEvals)
    
    i = 1
    while funEvals <= options['maxIter']:
        # compute step direction
        # this is for initialization
        if i == 1:
            p = funProj(x-g)
            projects = projects + 1
            S = np.zeros((nVars, 0))
            Y = np.zeros((nVars, 0))
            Hdiag = 1
        else:
            y = g - g_old
            s = x - x_old

            # translate this function (Done!)
            [S, Y, Hdiag] = lbfgsUpdate(y, s, options['corrections'], \
                    options['verbose']==3, S, Y, Hdiag)

            # make compact representation
            k = Y.shape[1]
            L = np.zeros((k,k))
            for j in range(k):
                L[j+1:,j] = np.dot(np.transpose(S[:,j+1:]), Y[:,j])
            N = np.hstack((S/Hdiag, Y.reshape(Y.shape[0], Y.size/Y.shape[0])))
            M1 = np.hstack((np.dot(S.T,S)/Hdiag, L))
            M2 = np.hstack((L.T, -np.diag(np.diag(np.dot(S.T,Y)))))
            M = np.vstack((M1, M2))
            
            # translate this function (Done!)
            HvFunc = lambda v: v/Hdiag - np.dot(N,np.linalg.solve(M,np.dot(N.T,v)))
            
            if options['bbInit'] == True:
                # use Barzilai-Borwein step to initialize sub-problem
                alpha = np.dot(s,s)/np.dot(s,y)
                if alpha <= 1e-10 or alpha > 1e10:
                    alpha = min(1., 1./np.sum(np.abs(g)))
                # solve sub-problem
                xSubInit = x - alpha*g
                feasibleInit = 0
            else:
                xSubInit = x
                feasibleInit = 1

            # solve Sub-problem
            # translate this function (Done!)
            [p, subProjects] = solveSubProblem(x, g, HvFunc, funProj, \
                    options['SPGoptTol'], options['SPGprogTol'], \
                    options['SPGiters'], options['SPGtestOpt'], feasibleInit,\
                    xSubInit)
            projects = projects + subProjects

        d = p - x
        g_old = g
        x_old = x

        # check the progress can be made along the direction
        gtd = np.dot(g,d)
        if gtd > -options['progTol']:
            if options['verbose'] >= 1:
                print "Directional Derivative below progTol"
            break
        
        # select initial guess to step length
        if i == 1 or options['adjustStep'] == 0:
            t = 1.
        else:
            t = min(1., 2.*(f-f_old)/gtd)
        
        # bound step length on first iteration
        if i == 1:
            t = min(1., 1./np.sum(np.abs(g)))

        # evluate the objective and gradient at the initial step
        if t == 1:
            x_new = p
        else:
            x_new = x + t*d
        [f_new, g_new] = funObj(x_new)
        funEvals = funEvals + 1

        # backtracking line search
        f_old = f
        # translate isLegal (Done!)
        while f_new > f + options['suffDec']*np.dot(g,x_new-x) or \
                not isLegal(f_new):
            temp = t
            # backtrack to next trial value
            if not isLegal(f_new) or not isLegal(g_new):
                if options['verbose'] == 3:
                    print "Halving step size"
                t = t/2.
            else:
                if options['verbose'] == 3:
                    print "Cubic backtracking"
                # translate polyinterp (Done!)
                t = polyinterp(np.array([[0.,f,gtd],\
                                        [t,f_new,np.dot(g_new,d)]]))[0]

            # adjust if change is too small/large
            if t < temp*1e-3:
                if options['verbose'] == 3:
                    print "Interpolated value too small, Adjusting"
                t = temp*1e-3
            elif t > temp*0.6:
                if options['verbose'] == 3:
                    print "Interpolated value too large, Adjusting"
                t = temp*0.6

            # check whether step has become too small
            if np.sum(np.abs(t*d)) < options['progTol'] or t == 0:
                if options['verbose'] == 3:
                    print "Line search failed"
                t = 0
                f_new = f
                g_new = g
                break

            # evaluate new point
            f_prev = f_new
            t_prev = temp
            x_new = x + t*d
            [f_new, g_new] = funObj(x_new)
            funEvals = funEvals + 1

        # take step
        x = x_new
        f = f_new
        g = g_new

        optCond = np.max(np.abs(funProj(x-g)-x))
        projects = projects + 1

        # output log
        if options['verbose'] >= 2:
            print '{:10d}'.format(i) + \
                  '{:10d}'.format(funEvals*funEvalMultiplier) + \
                  '{:10d}'.format(projects) + \
                  '{:15.5e}'.format(t) + \
                  '{:15.5e}'.format(f) + \
                  '{:15.5e}'.format(optCond)

        # check optimality
        if optCond < options['optTol']:
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

        if funEvals > options['maxIter']:
            if options['verbose'] >= 1:
                print "Function evaluation exceeds maxIter"
            break

        if projects > options['maxProject']:
            if options['verbose'] >= 1:
                print "Number of projections exceeds maxProject"
            break
        i = i + 1

    return (x, f, funEvals)


if __name__ == "__main__":
    
    from scipy.io import savemat 

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
    funProj = lambda v: projectSimplex_vec(v)
    x_init = np.random.randn(nVars)
    options = {'verbose':2}
    
    data_mat = {'nInst':nInst, 'nVars':nVars, 'A':A, 'x':x, 'b':b, \
            'x_init':x_init}
    #savemat("./Mark_Schmidt/minConf/minConf_PQN_input.mat",data_mat)

    (x, f, funEvals) = minConf_PQN(funObj, x_init, funProj, options=options)

