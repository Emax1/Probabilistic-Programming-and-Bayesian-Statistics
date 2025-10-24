##Probabilistic Programming and Bayesian Statistics
%matplotlib inline
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)

import scipy.stats as stats

dist = stats.beta
n_trials = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials) // 2, 2, k + 1)  # Cambio aquí: / → //
    plt.xlabel("$p$, probability of heads") \
        if k in [0, len(n_trials) - 1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)

    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)

plt.suptitle("Bayesian updating of posterior probabilities",
             y=1.02,
             fontsize=14)
plt.tight_layout()


###########
figsize(12.5, 4)
p = np.linspace(0, 1, 50)
plt.plot(p, 2 * p / (1 + p), color="#348ABD", lw=3)
# plt.fill_between(p, 2*p/(1+p), alpha=.5, facecolor=["#A60628"])
plt.scatter(0.2, 2 * (0.2) / 1.2, s=140, c="#348ABD")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Prior, $P(A) = p$")
plt.ylabel("Posterior, $P(A|X)$, with $P(A) = p$")
plt.title("Is my code bug-free?")


########
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

figsize(12.5, 4)
colours = ["#348ABD", "#A60628"]

prior = [0.20, 0.80]
posterior = [1. / 3, 2. / 3]

plt.bar([0, .7], prior, alpha=0.70, width=0.25,
        color=colours[0], label="prior distribution",
        lw=3, edgecolor=colours[0])  # Cambiado: lw="3" → lw=3

plt.bar([0 + 0.25, .7 + 0.25], posterior, alpha=0.7,
        width=0.25, color=colours[1],
        label="posterior distribution",
        lw=3, edgecolor=colours[1])  # Cambiado: lw="3" → lw=3

plt.ylim(0,1)
plt.xticks([0.20, .95], ["Bugs Absent", "Bugs Present"])
plt.title("Prior and Posterior probability of bugs present")
plt.ylabel("Probability")
plt.legend(loc="upper left")
plt.show()


#####
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

figsize(12.5, 4)

a = np.arange(16)
poi = stats.poisson
lambda_ = [1.5, 4.25]
colours = ["#348ABD", "#A60628"]

plt.bar(a, poi.pmf(a, lambda_[0]), color=colours[0],
        label="$\lambda = %.1f$" % lambda_[0], alpha=0.60,
        edgecolor=colours[0], lw=3)  # Cambiado: lw="3" → lw=3

plt.bar(a, poi.pmf(a, lambda_[1]), color=colours[1],
        label="$\lambda = %.1f$" % lambda_[1], alpha=0.60,
        edgecolor=colours[1], lw=3)  # Cambiado: lw="3" → lw=3

plt.xticks(a + 0.4, a)
plt.legend()
plt.ylabel("probability of $k$")
plt.xlabel("$k$")
plt.show()


####
a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.5, 1]

for l, c in zip(lambda_, colours):
    plt.plot(a, expo.pdf(a, scale=1. / l), lw=3,
             color=c, label="$\lambda = %.1f$" % l)
    plt.fill_between(a, expo.pdf(a, scale=1. / l), color=c, alpha=.33)

plt.legend()
plt.ylabel("PDF at $z$")
plt.xlabel("$z$")
plt.ylim(0, 1.2)
plt.title("Probability density function of an Exponential random variable;\
 differing $\lambda$");
 
 
 #####
figsize(12.5, 3.5)
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);

########
import pymc as pm
import numpy as np

# Asegúrate de que count_data está definido
# count_data = ... (tus datos aquí)

# Calcula n_count_data si no está definido
n_count_data = len(count_data)

# Define el modelo dentro de un contexto
with pm.Model() as model:
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
    
    # Definir la likelihood
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
    observation = pm.Poisson("obs", lambda_, observed=count_data)


with model:
    print("Random output:", tau.eval(), tau.eval(), tau.eval())


##########
import pymc as pm
import numpy as np

with pm.Model() as model:
    # Priors
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=len(count_data))
    
    # Lambda determinístico
    idx = np.arange(len(count_data))
    lambda_ = pm.Deterministic("lambda", 
                pm.math.switch(tau > idx, lambda_1, lambda_2))
    
    # Likelihood
    observation = pm.Poisson("obs", lambda_, observed=count_data)
    
    # Inference
    trace = pm.sample(1000, tune=1000)
    
####

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo (reemplaza con tus datos reales)
# count_data = np.loadtxt("data.txt")  # o como cargues tus datos
count_data = np.array([10, 15, 12, 8, 20, 25, 18, 22, 19, 16])  # ejemplo

with pm.Model() as model:
    # Priors
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=len(count_data))
    
    # Lambda determinístico
    idx = np.arange(len(count_data))
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
    
    # Likelihood
    observation = pm.Poisson("obs", lambda_, observed=count_data)
    
    # Muestreo
    trace = pm.sample(2000, tune=1000, target_accept=0.95)

# Visualizar resultados
pm.plot_trace(trace)
plt.show()

# Resumen estadístico
print(pm.summary(trace))


#######
# Versión más moderna con InferenceData
# Asegúrate de que esto esté AL PRINCIPIO de tu notebook/código
import pymc as pm
import numpy as np
import arviz as az

# Reinicia el kernel si estás en Jupyter si has tenido errores previos

# Definir n_count_data correctamente
n_count_data = len(count_data)

print(f"Datos: {count_data}")
print(f"Longitud: {n_count_data}")

with pm.Model() as model:
    # Priors
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
    
    # Lambda usando where en lugar de switch (más robusto)
    idx = np.arange(n_count_data)
    lambda_ = pm.Deterministic("lambda", 
                pm.math.where(idx < tau, lambda_1, lambda_2))
    
    # Likelihood
    obs = pm.Poisson("obs", lambda_, observed=count_data)
    
    # Sample
    trace = pm.sample(
        2000, 
        tune=1000, 
        chains=2,
        target_accept=0.9,
        return_inferencedata=True
    )

print("¡Muestreo completado exitosamente!")

####
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np

# Asegúrate de tener las muestras del trace
lambda_1_samples = trace.posterior['lambda_1'].values.flatten()
lambda_2_samples = trace.posterior['lambda_2'].values.flatten()
tau_samples = trace.posterior['tau'].values.flatten()

# Configurar el tamaño de la figura
figsize(12.5, 10)

# Gráfica 1: Distribución posterior de lambda_1
ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")

# Ajustar los límites según tus datos - puedes modificar estos valores
plt.xlim([lambda_1_samples.min() * 0.9, lambda_1_samples.max() * 1.1])
plt.xlabel("$\lambda_1$ value")

# Gráfica 2: Distribución posterior de lambda_2
ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([lambda_2_samples.min() * 0.9, lambda_2_samples.max() * 1.1])
plt.xlabel("$\lambda_2$ value")

# Gráfica 3: Distribución posterior de tau
plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=len(count_data), alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(len(count_data)))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([0, len(count_data)])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")

plt.tight_layout()
plt.show()

#########
import arviz as az

# Gráficas de traza y distribuciones posteriores
az.plot_trace(trace)
plt.tight_layout()
plt.show()

# Gráficas de distribución posterior más detalladas
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# lambda_1
axes[0,0].hist(lambda_1_samples, bins=30, alpha=0.7, color="#A60628", density=True)
axes[0,0].set_title("Posterior of $\lambda_1$")
axes[0,0].set_xlabel("$\lambda_1$")

# lambda_2
axes[0,1].hist(lambda_2_samples, bins=30, alpha=0.7, color="#7A68A6", density=True)
axes[0,1].set_title("Posterior of $\lambda_2$")
axes[0,1].set_xlabel("$\lambda_2$")

# tau
axes[1,0].hist(tau_samples, bins=len(count_data), alpha=0.7, color="#467821", density=True)
axes[1,0].set_title("Posterior of $\\tau$")
axes[1,0].set_xlabel("$\\tau$")

# Distribución conjunta lambda_1 vs lambda_2
axes[1,1].hexbin(lambda_1_samples, lambda_2_samples, gridsize=30, cmap='Blues')
axes[1,1].set_xlabel("$\lambda_1$")
axes[1,1].set_ylabel("$\lambda_2$")
axes[1,1].set_title("Joint distribution")

plt.tight_layout()
plt.show()

#######
# Gráfica del punto de cambio tau
figsize(10, 6)
plt.hist(tau_samples, bins=len(count_data), alpha=0.8, 
         color="#467821", density=True, edgecolor='black')
plt.axvline(tau_samples.mean(), color='red', linestyle='--', 
            label=f'Mean tau = {tau_samples.mean():.1f}')
plt.xlabel("Day of change ($\\tau$)")
plt.ylabel("Probability")
plt.title("Posterior distribution of change point $\\tau$")
plt.legend()
plt.show()

######
# Gráfica de las tasas a lo largo del tiempo
figsize(12, 6)

# Calcular promedios posteriores
lambda_1_mean = lambda_1_samples.mean()
lambda_2_mean = lambda_2_samples.mean()
tau_mean = int(tau_samples.mean())

# Crear vector de lambda a lo largo del tiempo
lambda_over_time = np.zeros(len(count_data))
lambda_over_time[:tau_mean] = lambda_1_mean
lambda_over_time[tau_mean:] = lambda_2_mean

plt.plot(np.arange(len(count_data)), lambda_over_time, 'r-', 
         label='Posterior mean of $\lambda$', lw=2)
plt.bar(np.arange(len(count_data)), count_data, alpha=0.3, 
        color='#348ABD', label='Observed data')
plt.axvline(tau_mean, color='k', linestyle='--', 
            label=f'Change point (day {tau_mean})')
plt.xlabel("Day")
plt.ylabel("Count")
plt.title("Observed data and posterior mean rate")
plt.legend()
plt.show()


########
figsize(12.5, 5)
# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left");