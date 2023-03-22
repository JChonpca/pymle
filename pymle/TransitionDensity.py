from abc import ABC, abstractmethod
import numpy as np
from pymle.Model import Model1D
from typing import Union


class TransitionDensity(ABC):
    def __init__(self, model: Model1D):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

        :param model: the SDE model, referenced during calls to the transition density
        """
        self._model = model

    @property
    def model(self) -> Model1D:
        """ Access to the underlying model """
        return self._model

    @abstractmethod
    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError


class ExactDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the exact transition density for a model (when available)
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        return self._model.exact_density(x0=x0, xt=xt, t0=t0, dt=dt)


class EulerDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig2t = (self._model.diffusion(x0, t0) ** 2) * 2 * dt
        mut = x0 + self._model.drift(x0, t0) * dt
        return np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)


class OzakiDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)
        mu_x = self._model.drift_x(x0, t0)
        temp = mu * (np.exp(mu_x * dt) - 1) / mu_x

        Mt = x0 + temp
        Kt = (2 / dt) * np.log(1 + temp / x0)
        Vt = sig * np.sqrt((np.exp(Kt * dt) - 1) / Kt)

        return np.exp(-0.5 * ((xt - Mt) / Vt) ** 2) / (np.sqrt(2 * np.pi) * Vt)


class ShojiOzakiDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Shoji-Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Shoji-Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)

        Mt = 0.5 * sig ** 2 * self._model.drift_xx(x0, t0) + self._model.drift_t(x0, t0)
        Lt = self._model.drift_x(x0, t0)
        if (Lt == 0).any():  # TODO: need to fix this
            B = sig * np.sqrt(dt)
            A = x0 + mu * dt + Mt * dt ** 2 / 2
        else:
            B = sig * np.sqrt((np.exp(2 * Lt * dt) - 1) / (2 * Lt))

            elt = np.exp(Lt * dt) - 1
            A = x0 + mu / Lt * elt + Mt / (Lt ** 2) * (elt - Lt * dt)

        return np.exp(-0.5 * ((xt - A) / B) ** 2) / (np.sqrt(2 * np.pi) * B)


class ElerianDensity(EulerDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Elerian (Milstein) approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Milstein Expansion (Elarian density).
        When d(sigma)/dx = 0, reduces to Euler
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig_x = self._model.diffusion_x(x0, t0)
        if (isinstance(x0, np.ndarray) and (sig_x == 0).any) or (not isinstance(x0, np.ndarray) and sig_x == 0):
            return super().__call__(x0=x0, xt=xt, t0=t0, dt=dt)

        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)

        A = sig * sig_x * dt * 0.5
        B = -0.5 * sig / sig_x + x0 + mu * dt - A
        z = (xt - B) / A
        C = 1. / (sig_x ** 2 * dt)

        scz = np.sqrt(C * z)
        cpz = -0.5 * (C + z)
        ch = (np.exp(scz + cpz) + np.exp(-scz + cpz))
        return np.power(z, -0.5) * ch / (2 * np.abs(A) * np.sqrt(2 * np.pi))


class KesslerDensity(EulerDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Kessler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Kessler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        sig2 = sig ** 2
        sig_x = self._model.diffusion_x(x0, t0)
        sig_xx = self._model.diffusion_xx(x0, t0)
        mu = self._model.drift(x0, t0)
        mu_x = self._model.drift_x(x0, t0)

        d = dt ** 2 / 2
        E = x0 + mu * dt + (mu * mu_x + 0.5 * sig2 * sig_xx) * d

        term = 2 * sig * sig_x
        V = x0 ** 2 + (2 * mu * x0 + sig2) * dt + (2 * mu * (mu_x * x0 + mu + sig * sig_x) +
                                                   sig2 * (sig_xx * x0 + 2 * sig_x + term + sig * sig_xx)) * d - E ** 2
        V = np.sqrt(np.abs(V))
        return np.exp(-0.5 * ((xt - E) / V) ** 2) / (np.sqrt(2 * np.pi) * V)


class AitSahalia(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Ait-Sahalia approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        return self._model.AitSahalia_density(x0=x0, xt=xt, t0=t0, dt=dt)




import numba as nb
from numba import njit, prange, jit


@nb.jit(nopython=True,cache=True)
def factorial_jit(i):
    
    result = 1
    
    for j in range(1,int(i+1)):
        
        result = result*j
        
    return result


@nb.jit(nopython=True,cache=True)
def STM_likehood_jit1(params,x0,xt,t,dt):
    
    likehood = 0
    
    for i in range(10):
        
        f1 = ((params[8]*dt)**i)/(factorial_jit(i))*(np.exp(-params[8]*dt))
        
        if i == 0:
            
            f2 = 1
            
            drift = (-params[2])*((np.sign(-t+params[0])+1)/2)*((np.sign(-t+params[1])+1)/2) + (-params[3])*((np.sign(t-params[1])+1)/2)
            
            diffusion = (params[4]*((np.sign(-t+params[0])+1)/2) + params[5]*((np.sign(t-params[0])+1)/2))*((np.sign(-t+params[1])+1)/2) + params[6]*((np.sign(t-params[1])+1)/2)
            
            sig2t = (diffusion ** 2) * 2 * dt
            
            mut = x0 + drift * dt
            
            f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
            
            likehood += f1*f2*f3
        
        else:
            
            #积分
            
            n = i
            
            S = np.linspace(-n*params[7], n*params[7], 100)
            
            S_norm = (S+n*params[7])/(2*params[7])
            
            Intel = 0
            
            for j in range(1, S.shape[0]):
                
                x = S_norm[j]
                
                f2 = 0
                
                for k in range(int(np.ceil(x))):
                    
                    f2 += ((-1)**(k))*((factorial_jit(n))/((factorial_jit(k))*(factorial_jit(n-k))))*((x-k)**(n-1))
                    
                f2 = f2/factorial_jit(n-1)
                
                f2 = f2/(2*params[7])
                
                # print(f2)
                    
            
                drift = (-params[2])*((np.sign(-t+params[0])+1)/2)*((np.sign(-t+params[1])+1)/2) + (-params[3])*((np.sign(t-params[1])+1)/2)
                
                diffusion = (params[4]*((np.sign(-t+params[0])+1)/2) + params[5]*((np.sign(t-params[0])+1)/2))*((np.sign(-t+params[1])+1)/2) + params[6]*((np.sign(t-params[1])+1)/2)
                
                sig2t = (diffusion ** 2) * 2 * dt
                
                mut = x0 + drift * dt + S[j]
                
                f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
                
                Intel += f2*f3*(S[j]-S[j-1])
                
            
            likehood += f1*Intel
        
    
    return likehood

@nb.jit(nopython=True,cache=True)
def STM_likehood_jit2(params,x0,xt,t,dt):
    
    likehood = 0
    
    for i in range(10):
        
        f1 = ((params[2]*dt)**i)/(factorial_jit(i))*(np.exp(-params[2]*dt))
        
        if i == 0:
            
            f2 = 1
            
            drift = 0
            
            diffusion = params[0]
            
            sig2t = (diffusion ** 2) * 2 * dt
            
            mut = x0 + drift * dt
            
            f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
            
            likehood += f1*f2*f3
        
        else:
            
            #积分
            
            n = i
            
            S = np.linspace(-n*params[1], n*params[1], 100)
            
            S_norm = (S+n*params[1])/(2*params[1])
            
            Intel = 0
            
            for j in range(1, S.shape[0]):
                
                x = S_norm[j]
                
                f2 = 0
                
                for k in range(int(np.ceil(x))):
                    
                    f2 += ((-1)**(k))*((factorial_jit(n))/((factorial_jit(k))*(factorial_jit(n-k))))*((x-k)**(n-1))
                    
                f2 = f2/factorial_jit(n-1)
                
                f2 = f2/(2*params[1])
                
                # print(f2)
                    
            
                drift = 0
                
                diffusion = params[0]
                
                sig2t = (diffusion ** 2) * 2 * dt
                
                mut = x0 + drift * dt + S[j]
                
                f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
                
                Intel += f2*f3*(S[j]-S[j-1])
                
            
            likehood += f1*Intel
        
    
    return likehood


@nb.njit(parallel=True,cache=True)
def goal1(x,x0,xt,t0,dt):
    
    bias = []
    
    for i in prange(len(x0)):
        
        bias.append(STM_likehood_jit1(x, x0[i], xt[i], t0[i], dt[i]))
    
        
    bias = np.array(bias)
    
    
    return bias



@nb.njit(parallel=True,cache=True)
def goal2(x,x0,xt,t0,dt):
    
    bias = []
    
    for i in prange(len(x0)):
        
        bias.append(STM_likehood_jit2(x, x0[i], xt[i], t0[i], dt[i]))
    
        
    bias = np.array(bias)
    
    
    return bias




import taichi as ti
import taichi.math as tm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

ti.init(arch=ti.cuda, default_fp=ti.f64 ,debug=True)


@ti.func
def factorial_taichi(i: int) -> int:
    
    result = 1
    
    ti.loop_config(serialize=True)
    
    for j in range(1,i+1):
        
        result *= j
        
    return result


@ti.func
def I1(params: ti.types.ndarray(), x0: float, xt: float, t: float, dt: float, i : int) -> float:
    
    n = i
        
    Intel = 0.0
        
    for j in range(1, 100):
        
        x = n/(99)*j
        
        f2 = IHpdf1(params, x, n)
        
        drift = (-params[2])*((tm.sign(-t+params[0])+1)/2)*((tm.sign(-t+params[1])+1)/2) + (-params[3])*((tm.sign(t-params[1])+1)/2)
        
        diffusion = (params[4]*((tm.sign(-t+params[0])+1)/2) + params[5]*((tm.sign(t-params[0])+1)/2))*((tm.sign(-t+params[1])+1)/2) + params[6]*((tm.sign(t-params[1])+1)/2)
        
        sig2t = (diffusion ** 2) * 2 * dt
        
        mut = x0 + drift * dt + (2*n*params[7])/99*j
        
        f3 = tm.exp(-(xt - mut) ** 2 / sig2t) / tm.sqrt(tm.pi * sig2t)
        
        Intel += f2*f3*(2*n*params[7])/99
    
    return Intel


@ti.func
def IHpdf1(params: ti.types.ndarray(), x: float , n : int)  -> float:
    
    f2 = 0.0
    
    for k in range(int(tm.ceil(x))):
        
        f2 += ((-1)**(k))*((factorial_taichi(n))/((factorial_taichi(k))*(factorial_taichi(n-k))))*((x-k)**(n-1))
        
    f2 = f2/factorial_taichi(n-1)
    
    f2 = f2/(2*params[7])
    
    return f2


@ti.func
def STM_likehood_tiachi1(params: ti.types.ndarray(),x0: float, xt: float, t: float, dt: float) -> float:
    
    likehood = 0.0
    
    for i in range(10):
        
        f1 = ((params[8]*dt)**i)/(factorial_taichi(i))*(tm.exp(-params[8]*dt))
        
        if i == 0:
            
            f2 = 1
            
            drift = (-params[2])*((tm.sign(-t+params[0])+1)/2)*((tm.sign(-t+params[1])+1)/2) + (-params[3])*((tm.sign(t-params[1])+1)/2)
            
            diffusion = (params[4]*((tm.sign(-t+params[0])+1)/2) + params[5]*((tm.sign(t-params[0])+1)/2))*((tm.sign(-t+params[1])+1)/2) + params[6]*((tm.sign(t-params[1])+1)/2)
            
            sig2t = (diffusion ** 2) * 2 * dt
            
            mut = x0 + drift * dt
            
            f3 = tm.exp(-(xt - mut) ** 2 / sig2t) / tm.sqrt(tm.pi * sig2t)
            
            likehood += f1*f2*f3
        
        else:
            
            Intel = I1(params, x0, xt, t, dt, i)
            
            likehood += f1*Intel
        
    
    return likehood




@ti.func
def I2(params: ti.types.ndarray(), x0: float, xt: float, t: float, dt: float, i : int) -> float:
    
    n = i
        
    Intel = 0.0
        
    for j in range(1, 100):
        
        x = n/(99)*j
        
        f2 = IHpdf2(params, x, n)
        
        drift = 0
        
        diffusion = params[0]
        
        sig2t = (diffusion ** 2) * 2 * dt
        
        mut = x0 + drift * dt + (2*n*params[1])/99*j
        
        f3 = tm.exp(-(xt - mut) ** 2 / sig2t) / tm.sqrt(tm.pi * sig2t)
        
        Intel += f2*f3*(2*n*params[1])/99
    
    return Intel


@ti.func
def IHpdf2(params: ti.types.ndarray(), x: float , n : int)  -> float:
    
    f2 = 0.0
    
    for k in range(int(tm.ceil(x))):
        
        f2 += ((-1)**(k))*((factorial_taichi(n))/((factorial_taichi(k))*(factorial_taichi(n-k))))*((x-k)**(n-1))
        
    f2 = f2/factorial_taichi(n-1)
    
    f2 = f2/(2*params[1])
    
    return f2


@ti.func
def STM_likehood_tiachi2(params: ti.types.ndarray(),x0: float, xt: float, t: float, dt: float) -> float:
    
    likehood = 0.0
    
    for i in range(10):
        
        f1 = ((params[2]*dt)**i)/(factorial_taichi(i))*(tm.exp(-params[2]*dt))
        
        if i == 0:
            
            f2 = 1
            
            drift = 0
            
            diffusion = params[0]
            
            sig2t = (diffusion ** 2) * 2 * dt
            
            mut = x0 + drift * dt
            
            f3 = tm.exp(-(xt - mut) ** 2 / sig2t) / tm.sqrt(tm.pi * sig2t)
            
            likehood += f1*f2*f3
        
        else:
            
            Intel = I2(params, x0, xt, t, dt, i)
            
            likehood += f1*Intel
        
    
    return likehood








# 内存预先占用

all_likehood = ti.field(ti.f64, shape=20000)


        
@ti.kernel
def goal3(params: ti.types.ndarray(),
          x0: ti.types.ndarray(), 
          xt: ti.types.ndarray(), 
          t: ti.types.ndarray(), 
          dt: ti.types.ndarray()):
    
    
    for i in x0:
        
        all_likehood[i] = STM_likehood_tiachi1(params,x0[i], xt[i], t[i], dt[i])


@ti.kernel
def goal4(params: ti.types.ndarray(),
          x0: ti.types.ndarray(), 
          xt: ti.types.ndarray(), 
          t: ti.types.ndarray(), 
          dt: ti.types.ndarray()):
    
    
    for i in x0:
        
        all_likehood[i] = STM_likehood_tiachi2(params,x0[i], xt[i], t[i], dt[i])




class JumpEulerDensity1(TransitionDensity):
    
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        
        all_likehood = goal1(self._model.params,x0,xt,t0,dt)   
        
        return all_likehood





class JumpEulerDensity2(TransitionDensity):
    
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        
        all_likehood = goal2(self._model.params,x0,xt,t0,dt)   
        
        return all_likehood



class JumpEulerDensity3(TransitionDensity):
    
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        
        
        goal3(self._model.params,x0,xt,t0,dt)   
        
        all_likehood_return = all_likehood.to_numpy()
        
        all_likehood_return = all_likehood_return[0:t0.shape[0]]
        
        return all_likehood_return





class JumpEulerDensity4(TransitionDensity):
    
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        
        
        goal4(self._model.params,x0,xt,t0,dt)
        
        all_likehood_return = all_likehood.to_numpy()
        
        all_likehood_return = all_likehood_return[0:t0.shape[0]]
        
        return all_likehood_return


# class JumpEulerDensity(TransitionDensity):
    
#     def __init__(self, model: Model1D):
#         """
#         Class which represents the Euler approximation transition density for a model
#         :param model: the SDE model, referenced during calls to the transition density
#         """
#         super().__init__(model=model)

#     def __call__(self,
#                  x0: Union[float, np.ndarray],
#                  xt: Union[float, np.ndarray],
#                  t0: Union[float, np.ndarray],
#                  dt: float) -> Union[float, np.ndarray]:
        
#         """
#         The transition density obtained via Euler expansion
#         :param x0: float or array, the current value
#         :param xt: float or array, the value to transition to  (must be same dimension as x0)
#         :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
#         :param dt: float, the time step between x0 and xt
#         :return: probability (same dimension as x0 and xt)
#         """
        
#         # self._model.diffusion(x0, t0)
        
#         # self._model.drift(x0, t0)
        
#         # self._modle.jump(x0, t0)
        
#         # sig2t = (self._model.diffusion(x0, t0) ** 2) * 2 * dt
        
#         # mut = x0 + self._model.drift(x0, t0) * dt
        
#         # np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
        
#         # Tuable
        
#         x0_ = x0.copy()
        
#         xt_ = xt.copy()
        
#         t0_ = t0.copy()
        
#         dt_ = dt.copy()
        
#         all_likehood = np.zeros_like(x0_)
        
#         for num in range(x0_.shape[0]):
            
#             print(num)
            
#             x0 = x0_[num]
            
#             xt = xt_[num]
            
#             t0 = t0_[num]
            
#             dt = dt_[num]
        
#             likehood = 0
            
#             for i in range(40):
                
#                 f1 = ((self._model._params[8]*dt)**i)/(np.math.factorial_jit(i))*(np.e**(-self._model._params[8]*dt))
                
#                 if i == 0:
                    
#                     f2 = 1
                    
#                     sig2t = (self._model.diffusion(x0, t0) ** 2) * 2 * dt
                    
#                     mut = x0 + self._model.drift(x0, t0) * dt + 0
                    
#                     f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
                    
#                     likehood += f1*f2*f3
                
#                 else:
                    
#                     n = i
                    
#                     S = np.linspace(-n*self._model._params[8], n*self._model._params[8], 100)
                    
#                     S_norm = (S+n*self._model._params[8])/(2*self._model._params[8])
                    
#                     Intel = 0
                    
#                     for j in range(S_norm.shape[0]):
                        
#                         x = S_norm[j]
                        
#                         f2 = 0
                        
#                         for k in range(int(x)):
                            
#                             f2 += ((-1)**(k))*((np.math.factorial_jit(n))/((np.math.factorial_jit(k))*(np.math.factorial_jit(n-k))))*((x-k)**(n-1))
                            
#                         f2 = f2/np.math.factorial_jit(n-1)
                            
                    
#                         sig2t = (self._model.diffusion(x0, t0) ** 2) * 2 * dt
                        
#                         mut = x0 + self._model.drift(x0, t0) * dt + S[j]
                        
#                         f3 = np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)
                        
#                         Intel += f2*f3*(S[j]-S[j-1])
                        
                    
#                     likehood += f1*Intel
                
                
#             all_likehood[num] = likehood
            
            

#         return all_likehood
