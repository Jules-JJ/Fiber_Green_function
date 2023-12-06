#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:25:09 2023

Atom coupled with fiber simulation.

@author: juliette
"""

# import libraries
from math import *
import cmath
import matplotlib.pyplot as plt
import numpy as np
import scipy

#%% Green's tensor in vacuum

def _G0(r_vect, r_s):
    
    """
    w = pulsation
    r_vect = position to calculate the field vector in cartesian coordinates
    r_s = position vector of the source in cartesian coordinates
    
    """
    k = k0 #c = 1
    
    #Cartesian coordinates
    (x, y, z) = r_vect
    (xs, ys, zs) = r_s
    
    Delta_R = r_vect - r_s #distance
    R = sqrt((x-xs)**2 + (y-ys)**2 +(z-zs)**2) #norm of r_dist
    #print(R)
    
    if R == 0:
        
        #print("Divergence") 
        G0 = k/(6*pi)*np.identity(3) #Self term
        
    else:
        
        RR = 1/R**2 * np.array([[(x-xs)**2, (x-xs)*(y-ys), (x-xs)*(z-zs)], 
                               [(y-ys)*(x-xs), (y-ys)**2, (y-ys)*(z-zs)], 
                               [(z-zs)*(x-xs), (y-ys)*(z-zs), (z-zs)**2]])
        
        #print("No divergence")  
        G0 = cmath.exp(1j*k*R)/(4*pi*R)*((1 + (1j*k*R)/(k*R)**2) * np.identity(3) \
                                    + ((3-3*1j*k*R-(k*R)**2)/(k*R)**2) * RR)
    
    return G0

#%% Nanofiber Green's tensor

# Integrand

def IGA(nu, beta, k1, k2, r_vect, r_s, a=1):
    
    Eta1 = a*np.emath.sqrt(k1**2 - beta**2)
    Eta2 = a*np.emath.sqrt(k2**2 - beta**2) #complex for guided modes
    k1 = k1*a
    k2 = k2*a
    
    r_vect = r_vect/a
    r_obs = r_s/a
    
    [AR, BR, CR] = scalar_coeffs(nu, a*beta, Eta1, Eta2, k1, k2)
    
    [M1, N1, M_conj1, N_conj1] = Vector_field_coeffs(r_vect, r_s, nu, beta, Eta2)
    
    term1 = (AR*M1 + BR*N1)
    term2 = (CR*N1 + BR*M1)
    
    Int_GA = (1j/(8*pi*Eta2**2)) * (np.outer(term1, M_conj1)
                                  + np.outer(term2, N_conj1))
        
    return Int_GA

#print(IGA(0, 1.5, 2, 1, np.array([1, 0, 1]), np.array([1, 0, 1])))

# Branch integrand

def I_branch(nu, beta, k1, k2, r_vect, r_s, a=1):
    
    Eta1 = a*np.emath.sqrt(k1**2 - beta**2)
    Eta2 = a*np.emath.sqrt(k2**2 - beta**2) #complex for guided modes
    k1 = k1*a
    k2 = k2*a
    
    r_vect = r_vect/a
    r_obs = r_s/a
    
    dr, dphi, dz = r_vect - r_obs #cylindrical
    
    [AR, BR, CR] = scalar_coeffs(nu, a*beta, Eta1, Eta2, k1, k2)
    
    [M1, N1, M_conj1, N_conj1] = Vector_field_coeffs(r_vect, r_s, nu, beta, Eta2)
    
    term1 = (AR*M1 + BR*N1)
    term2 = (CR*N1 + BR*M1)
    
    g = (1j/(Eta2**2)) * (np.outer(term1, M_conj1)
                                  + np.outer(term2, N_conj1))*cmath.exp(-1j*nu*dphi)
    
    eps = beta.imag
    
    Int_B = g.imag * cmath.exp(-1j*nu*dphi)
    
    return Int_B

# Vector field coefficients

def Vector_field_coeffs(r_vect, r_s, nu, beta, eta, a=1): # -> 4 vectors of size (3, 1)
    
    """
    nu = mode
    beta = integration point in the real line (physically: propagation direction)
    r_vect = position to calculate the field vector in cartesian coordinates
    r_s = position vector of the source in cartesian coordinates
    
    """
    
    #observation vector
    r, phi, z = r_vect #cylindrical
    
    er = np.array([cos(phi), sin(phi), 0]) #radial unity vector
    ephi = np.array([-sin(phi), cos(phi), 0]) #angular unity vector
    ez = np.array([0, 0, 1]) #axis unity vector
    
    Hn = scipy.special.hankel1(nu, eta*r)
    Hn_prime = scipy.special.h1vp(nu, eta*r)
    
    #Source vector
    rs, phi_s, zs = r_s
    
    er_s = np.array([cos(phi_s), sin(phi_s), 0]) #radial unity vector
    ephi_s = np.array([-sin(phi_s), cos(phi_s), 0]) #angular unity vector
    ez_s = np.array([0, 0, 1]) #axis unity vector
    
    Hn_s = scipy.special.hankel1(nu, eta*rs)
    Hn_prime_s = scipy.special.h1vp(nu, eta*rs)
    
    M = a*((1j*nu/r)*Hn*er - eta*Hn_prime*ephi) * cmath.exp(1j*(nu*phi + beta*z))
    
    N = (1/a)*(1/cmath.sqrt(eta**2 + beta**2)) * (1j*eta*beta*Hn_prime*er - (nu*beta/r)*Hn*ephi + eta**2*Hn*ez) * cmath.exp(1j*(nu*phi + beta*z))
    
    M_conj = a*((-1j*nu/rs)*Hn_s*er_s - eta*Hn_prime_s*ephi_s) * cmath.exp(-1j*(nu*phi_s + beta*zs))
    
    N_conj = (1/a)*(1/cmath.sqrt(eta**2 + beta**2)) * (-1j*eta*beta*Hn_prime_s*er_s - (nu*beta/rs)*Hn_s*ephi_s + eta**2*Hn_s*ez_s) * cmath.exp(-1j*(nu*phi_s + beta*zs))
    
    return [M, N, M_conj, N_conj]

# Scalar field coefficients

def scalar_coeffs(nu, beta, eta1_norm, eta2_norm, k1_norm, k2_norm, a=1):
    
    """
    nu = mode
    beta = integration point (dimentionless)
    eta i_norm: eta_i*a 
    k i_norm: k_i*a  
    
    """
    
    #Bessel fucntions
    J1 = scipy.special.jv(nu, eta1_norm)
    J2 = scipy.special.jv(nu, eta2_norm)
    
    #H2 = Hankel_n(nu, eta2*a)
    H2 = scipy.special.hankel1(nu, eta2_norm) #complex
    
    #Derivatives of the bessel functions
    JD1 = scipy.special.jvp(nu, eta1_norm)
    JD2 = scipy.special.jvp(nu, eta2_norm)
    
    #HD2 = Hankel_n_prime(nu, eta2*a)
    HD2 = scipy.special.h1vp(nu, eta2_norm) #complex
    
    W = WR(nu, beta, eta1_norm, eta2_norm, k1_norm, k2_norm)
    
    A = (1/W)*(J2/H2) * ((nu*beta*a)**2 * (1/(eta2_norm**2) - 1/(eta1_norm**2))**2 \
          - (JD1/(eta1_norm*J1) - JD2/(eta2_norm*J2)) * ((k1_norm**2*JD1)/(eta1_norm*J1) - (k2_norm**2*HD2)/(eta2_norm*H2)))
    
    B = (1/W)*(J2/H2) * (k2_norm*nu*beta*a/(eta2_norm))*(1/(eta2_norm**2) - 1/(eta1_norm**2))*(JD2/J2 - HD2/H2)

    C = (1/W)*(J2/H2) * ((nu*beta*a)**2 * (1/(eta2_norm**2) - 1/(eta1_norm**2))**2 \
        - (JD1/(eta1_norm*J1) - HD2/(eta2_norm*H2))*((k1_norm**2*JD1)/(eta1_norm*J1) - (k2_norm**2*JD2)/(eta2_norm*J2)))
        
    return [A, B, C] #D=B

def WR(nu, beta, eta1_norm, eta2_norm, k1_norm, k2_norm, a=1): 
    
    """
    nu = mode
    beta = integration point (dimentionless)
    eta i_norm: eta_i*a 
    k i_norm: k_i*a  
    
    """
    
    #Bessel fucntions
    J1 = scipy.special.jv(nu, eta1_norm)
    H2 = scipy.special.hankel1(nu, eta2_norm)
    
    #Derivatives of the bessel functions
    JD1 = scipy.special.jvp(nu, eta1_norm)
    HD2 = scipy.special.h1vp(nu, eta2_norm)

    W = -(nu*beta)**2 * (1/(eta2_norm**2) - 1/(eta1_norm**2))**2 \
       + (JD1/(eta1_norm*J1) - HD2/(eta2_norm*H2))*(k1_norm**2*JD1/(eta1_norm*J1) - k2_norm**2*HD2/(eta2_norm*H2))

    return W

#%% Nanofiber Green's tensor

def _Gng(r_vect, r_s, verbose = "False"):
        
    print("Calculating radiative mode contribution... \n")
    
    Rc = 10 # Rc >> 2.25
    delta = 1e-3
    step = 1e-3

    C9 = 1j*np.arange(delta, Rc+step, step) - k0
    C8 = np.arange(k0, -k0, -step) +1j*delta
    
    nu = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] #terme dans la somme (mode)
    
    I_ng = 0
    
    for nu_value in nu:
        
        if verbose == "True":
            print("mode: ", nu_value, "\n")
        
        #Branch -> contributes to frecuency shift
        I9 = 0
        
        #if r_obs = r_source -> Ignore branch integral
        
        if r_vect.all() != r_s.all():
            
            for i in range(len(C9)-1):
    
                I9 += (step/2) * (I_branch(nu_value, C9[i], n_f*k0, k0, r_vect, r_s) + I_branch(nu_value, C9[i+1], n_f*k0, k0, r_vect, r_s))
        
        if verbose == "True":
            print("Integral on contour 8... \n")
        
        #On top of the axis -> contributes to decay rate and frecuency shift
        I8 = 0
    
        for i in range(len(C8)-1):
    
            I8 += (step/2) * (IGA(nu_value, C8[i], n_f*k0, k0, r_vect, r_s) + (IGA(nu_value, C8[i+1], n_f*k0, k0, r_vect, r_s)))
    
        I_ng = I_ng + (I8 - 2*I9)
    
    G_non_guided = _G0(r_vect, r_s) +  1/(k0**2) * I_ng
    
    print(5*"--------")
    
    return G_non_guided

def _Gg(r_vect, r_s, verbose = "False"):
    
    print("Calculating guided mode contribution... \n")
    
    #Cartesian coordinates
    (x, y, z) = r_vect
    (xs, ys, zs) = r_s
    
    nu = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] #terme dans la somme (mode)
    step = 1e-3
    Line_beta = np.arange(k0 + step, n_f * k0 - step, step)
    
    beta_pole = []
    pole_index = []
    
    Res_guided_poles = 0
    Res_beta_0 = 0
    
    for nu_value in nu:
        

        #print("mode: ", nu_value)
    
        beta_pole.append([])
        pole_index.append([])
    
        result = []
    
        number_of_poles = 0
    
        #Pole determination
        for ind, beta in enumerate(Line_beta): 
            Eta1 = cmath.sqrt(k1**2 - beta**2)
            Eta2 = cmath.sqrt(k2**2 - beta**2)
    
            result.append(abs(WR(nu_value, beta, Eta1, Eta2, k1, k2)))
            #Poles when WR = 0, however to refine the pole search we can use log(WR)
            #log(WR -> 0) -> -oo se we just have to grab the values for beta that make log(WR) < 0
        
        result_l = np.log(result) 
        result_dl = np.gradient(result_l)
    
        for index in range(len(result_dl)-1): 
            if result_l[index] < 0 and (result_dl[index] < 0 and result_dl[index+1] > 0): 
            
                beta_pole[-1].append(Line_beta[index])
                pole_index[-1].append(index)
        
        number_of_poles = len(beta_pole[-1])
        
        if verbose == "True":
            print("Number of poles for mode nu = {}: ".format(nu_value), number_of_poles, "\n")
    
        #Residue calculations
        if verbose == "True":
            print("Residue calculations... \n")
        
    
        print("Residue calculations... \n")
    

        for k, pole in enumerate(beta_pole[-1]): 
            #print("pole: ", pole)
            #contributions from poles in [k0 + step, n_f * k0 - step]
            Res_guided_poles += 2j*np.pi * step * IGA(nu_value, pole, n_f*k0, k0, r_vect, r_s) 
    
            #contributions from poles in [-n_f * k0 + step, -k0 - step,]
            Res_guided_poles = 2*Res_guided_poles
    

        Res_beta_0 += 1j*np.pi * step * IGA(nu_value, n_f * k0 - step, n_f*k0, k0, r_vect, r_s)
        Res_beta_0 += 1j*np.pi * step * IGA(nu_value, -n_f * k0 + step, n_f*k0, k0, r_vect, r_s)
        
    Gg = cmath.exp(1j*k0*(z - zs))/(2*np.pi*k0**2) * (Res_guided_poles + Res_beta_0)
    
    print(5*"--------")
    
    return Gg  

def G_fibre(r_vect, r_s):
        
    return _Gg(r_vect, r_s) + _Gng(r_vect, r_s) 

#%% Nanofiber and environment parameter definition

n_f = 1.45 #index of refraction nanofiber
n_env = 1 #index of refraction environnement
# eps_f = n_f**2
# eps_env = 1

# rA = np.array([2, 0, 0]) #coordinate of the evaluation point (nm)
# rB = np.array([2, 0, 10]) #coordinate of the source point (nm)

# We are goin to find the poles with k0 and nu
a = 1 # nanofiber radius 
k0 = 1.2*a
k1 = n_f*k0
k2 = k0

rA = 1.5*a

#%% Simulation on one point
import time
start = time.time()

#position and dipole are row vectors

#points = [np.array([rA, 0, 0]), np.array([2*rA, 0, 0]), np.array([5*rA, 0, 0]), np.array([10*rA, 0, 0])]
#points = [np.array([rA, 0, 0]), np.array([rA, np.pi/2, 0]), np.array([rA, np.pi, 0]), np.array([rA, 2*np.pi, 0])]
points = np.array([np.array([0.25*rA, 0, 0]), np.array([0.5*rA, 0, 0]),
                    np.array([0.75*rA, 0, 0]), np.array([rA, 0, 0]),
                    np.array([1.25*rA, 0, 0]), np.array([1.5*rA, 0, 0]),
                    np.array([1.75*rA, 0, 0]), np.array([2*rA, 0, 0]),
                    np.array([2.25*rA, 0, 0]), np.array([2.5*rA, 0, 0]), 
                    np.array([2.75*rA, 0, 0]), np.array([3*rA, 0, 0])])


p = np.array([0, 0, 1]) #in cylindrical coordinates

GAMMA_PLOT = []

for r_obs in points:
    
    print(r_obs)
    
    Gf_g =  _Gg(r_obs, r_obs)
    #print("Guided mode tensor: \n", Gf_g, "\n")

    Gf_ng =  _Gng(r_obs, r_obs)
    #print("Radiative mode tensor: \n", Gf_ng, "\n")

    #print("Total tensor: \n", Gf_ng + Gf_g, "\n")

    #Gamma_g = 2*k0**2 * Gf_ng.imag[2][2] # Pi transition -> dipole along pz
    
    Gamma_g = 2*k0**2 *np.dot(p, np.dot((Gf_ng + Gf_g).imag, p))
    print("Gamma_guided: ", Gamma_g, "\n")
    
    GAMMA_PLOT.append(Gamma_g)

    #Gfng_rr = Gf_ng[0][0]

    #J_prime = -Gfng_rr.real
    #Gamma_prime = -Gfng_rr.imag

    #print("Gamma' : ", Gamma_prime, "\n")
    #print("J' : ", J_prime, "\n")

print('Final time', time.time()-start, 'seconds. \n')

GAMMA_PLOT =  np.array(GAMMA_PLOT)

fig = plt.figure()
plt.plot(points.T[0], GAMMA_PLOT)