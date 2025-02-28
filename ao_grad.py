from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AOGrad(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.y= np.zeros(shape=x0.shape)
        self.y[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=eta
        self.lam=np.zeros(shape=x0.shape)
        self.lam=self.lam+1e-6
        self.t=0.0
        self.beta=0.0
        self.l1=l1
        self.l2=l2
        self.h=np.zeros(shape=x0.shape)

    def update(self):
        self.t+=1.0
        self.beta+=self.t
        g=self.func_p(self.y)
        self.step(g)
        self.h[:]=g
        return self.y
                

    def step(self,g):
        self.update_parameters(g)
        self.md(g)

    def update_parameters(self,g):
        self.lam+=((self.t*(g-self.h))**2)
       
        
    def md(self,g):
        alpha = np.sqrt(self.lam)*self.eta
        z = self.x-(self.t*g-self.t*self.h+(self.t+1.0)*g)/alpha
        x_sgn = np.sign(z)
        x_val = np.maximum(alpha*np.abs(z)-self.l1*(self.t+1),0.0)/(alpha+self.l2*(self.t+1))
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)
        self.y=(self.t/self.beta)*self.x+((self.beta-self.t)/self.beta)*self.y
        

def fmin(func, func_p,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0 ):
    alg=AOGrad(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
    nit=maxfev
    fev=1
    y=None
    while fev <= maxfev:
        y=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    return OptimizeResult(func=func(y), x=y, nit=nit,
                          nfev=fev, success=(y is not None))

