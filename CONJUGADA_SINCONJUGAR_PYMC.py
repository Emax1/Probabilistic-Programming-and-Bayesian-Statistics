# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:53:05 2025

@author: UTM
"""
import matplotlib.pyplot as plt
import scipy.stats as stats

# Prior: Beta(2, 2)
alpha_prior, beta_prior = 2, 2

# Datos: 7 caras en 10 lanzamientos
caras, total = 7, 10

# Posterior: Beta(2+7, 2+3)
alpha_post = alpha_prior + caras
beta_post = beta_prior + (total - caras)

posterior = stats.beta(alpha_post, beta_post)

# Media posterior
media = posterior.mean()  # (2+7)/(2+7+2+3) = 9/14 ≈ 0.64



import pymc as pm

with pm.Model():
    p = pm.Uniform('p', 0, 1)  # Prior no conjugado
    obs = pm.Binomial('obs', n=10, p=p, observed=7)
    trace = pm.sample(1000)  # Requiere MCMC
    
# Visualizar resultados
pm.plot_trace(trace)
plt.show()
# Resumen estadístico
print(pm.summary(trace))
