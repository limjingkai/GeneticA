import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_onemax(dim: int) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return float(80 - abs(40 - ones))  # peak at 40 ones, max fitness = 80

    return GAProblem(
        name=f"Bit Pattern Optimization ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int,
):

    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):
        best_fit = np.max(fit)
        avg_fit = np.mean(fit)
        worst_fit = np.min(fit)

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Elitism
        elite_idx = np.argsort(fit)[-elitism:]
        elites = pop[elite_idx]

        next_pop = []

        while len(next_pop) < pop_size - elitism:
            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - elitism:
                next_pop.append(c2)

        pop = np.vstack([next_pop, elites])
        fit = evaluate(pop, problem)

    best_idx = np.argmax(fit)
    best = pop[best_idx]

    history = pd.DataFrame({
        "Best": history_best,
        "Average": history_avg,
        "Worst": history_worst
    })

    return best, fit[best_idx], history


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm Bit Pattern", layout="wide")
st.title("Genetic Algorithm – Bit Pattern Generator")
st.caption("Fixed-parameter Genetic Algorithm web application")

# Fixed parameters display
st.sidebar.header("Fixed Parameters")
st.sidebar.markdown("**Population Size:** 300")
st.sidebar.markdown("**Chromosome Length:** 80")
st.sidebar.markdown("**Generations:** 50")
st.sidebar.markdown("**Fitness Peak:** 40 ones")
st.sidebar.markdown("**Maximum Fitness:** 80")

# GA fixed values
POP_SIZE = 300
CHROM_LENGTH = 80
GENERATIONS = 50
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
ELITISM = 2
SEED = 42

problem = make_onemax(CHROM_LENGTH)

if st.button("Run Genetic Algorithm", type="primary"):
    best, best_fit, history = run_ga(
        problem=problem,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        tournament_k=TOURNAMENT_K,
        elitism=ELITISM,
        seed=SEED,
    )

    st.subheader("Fitness Convergence")
    st.line_chart(history)

    st.subheader("Best Bit Pattern")
    bitstring = "".join(map(str, best.astype(int)))
    st.code(bitstring, language="text")

    st.write(f"**Number of Ones:** {int(np.sum(best))} / {CHROM_LENGTH}")
    st.write(f"**Best Fitness Achieved:** {best_fit:.2f}")
