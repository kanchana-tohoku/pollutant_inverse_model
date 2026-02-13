import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Analytical solution of 1-D advection–diffusion eq.
# Based on Zhang & Xin (2017), Eq. (3)
# --------------------------------------------------

def pollutant_concentration(
    x, t,
    source_positions,
    source_strengths,
    l=1000.0,        # river length (m)
    u=0.1,           # flow velocity (m/s)
    Ex=5.0,          # diffusion coefficient (m^2/s)
    K=1e-5,          # decay coefficient (1/s)
    n_max=1000       # number of Fourier terms
):
    """
    Compute pollutant concentration C(x,t) using analytical solution.

    Parameters
    ----------
    x : array_like
        Spatial positions (m)
    t : float
        Time (s)
    source_positions : array_like
        Positions of pollution sources xi (m)
    source_strengths : array_like
        Source strengths Mi (g/s)
    l, u, Ex, K : float
        Physical parameters
    n_max : int
        Truncation of infinite series

    Returns
    -------
    C : ndarray
        Pollutant concentration at x and t
    """

    x = np.asarray(x)
    C = np.zeros_like(x, dtype=float)

    source_positions = np.asarray(source_positions)
    source_strengths = np.asarray(source_strengths)

    for n in range(1, n_max + 1):
        lambda_n = (n * np.pi / l)**2
        denom = u**2 / (4 * Ex) + lambda_n + K

        # Sum over pollution sources
        source_sum = 0.0
        for Mi, xi in zip(source_strengths, source_positions):
            source_sum += Mi * np.exp(-u * xi / (2 * Ex)) * np.sin(n * np.pi * xi / l)

        time_factor = np.exp(u**2 * t / (4 * Ex)) - np.exp(-denom * t)
        spatial_factor = np.sin(n * np.pi * x / l)

        C += (2 / l) * (source_sum / denom) * time_factor * spatial_factor

    # Exponential correction term
    C *= np.exp(u * x / (2 * Ex) - u**2 * t / (4 * Ex))

    return C

#==================================================
# Generate “observed” data (synthetic monitoring)
#==================================================

# Monitoring locations
#l = 1000
#x_obs = np.linspace(0.1*l, 0.9*l, 9)
T_obs = 4000

# TRUE (unknown) source — what GA must recover
#true_x = [500]
#true_M = [3.0]

#C_obs = pollutant_concentration(
 #   x_obs, T_obs,
  #  true_x, true_M
#)

x_obs = [0,100,200,300,400,500,600,700,800]
C_obs = [0,0,0,5,10,30,60,40,0]

#==================================================
# Fitness function
#==================================================

def fitness(individual, x_obs, C_obs, T_obs):
    x_i = [individual[0]]
    M_i = [individual[1]]

    C_model = pollutant_concentration(
        x_obs, T_obs,
        x_i, M_i
    )

    error = np.sum((C_model - C_obs)**2)
    return 1.0 / (error + 1e-12)   # GA maximizes fitness


#==================================================
#Genetic Algorithm implementation
#==================================================

#=======GA parameters==========
POP_SIZE = 10
N_GEN = 200
PC = 0.8     # crossover probability
PM = 0.05    # mutation probability

X_MIN, X_MAX = 0, 1000
M_MIN, M_MAX = 0.5, 6.0

#===========Initialize population===========

def init_population():
    pop = []
    for _ in range(POP_SIZE):
        x = np.random.uniform(X_MIN, X_MAX)
        M = np.random.uniform(M_MIN, M_MAX)
        pop.append([x, M])
    return np.array(pop)


#=====Selection (roulette wheel)

def select(pop, fitness_vals):
    probs = fitness_vals / fitness_vals.sum()
    idx = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[idx]

#=======Crossover + mutation===

def crossover(parent1, parent2):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    return parent1, parent2


def mutate(ind):
    if np.random.rand() < PM:
        ind[0] += np.random.normal(0, 20)   # position mutation
        ind[1] += np.random.normal(0, 0.3)  # strength mutation

    ind[0] = np.clip(ind[0], X_MIN, X_MAX)
    ind[1] = np.clip(ind[1], M_MIN, M_MAX)
    return ind

#====Run GA inversion

pop = init_population()
best_history = []

for gen in range(N_GEN):

    fitness_vals = np.array([
        fitness(ind, x_obs, C_obs, T_obs) for ind in pop
    ])

    best_idx = np.argmax(fitness_vals)
    best_history.append(pop[best_idx])

    # Selection
    pop = select(pop, fitness_vals)

    # Crossover
    new_pop = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = pop[i], pop[i+1]
        c1, c2 = crossover(p1, p2)
        new_pop.extend([c1, c2])

    # Mutation
    pop = np.array([mutate(ind) for ind in new_pop])

# Final result
best_solution = best_history[-1]
print(f"Recovered source position x = {best_solution[0]:.2f} m")
print(f"Recovered source strength M = {best_solution[1]:.2f} g/s")





