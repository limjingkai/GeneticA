import streamlit as st
import numpy as np

POPULATION = 300
CHROM_LENGTH = 80
GENERATIONS = 50
FITNESS_PEAK = 40

def fitness(individual):
    ones = np.sum(individual)
    return max(0, CHROM_LENGTH - abs(ones - FITNESS_PEAK))

population = np.random.randint(2, size=(POPULATION, CHROM_LENGTH))

best_fitness = []

for gen in range(GENERATIONS):
    scores = np.array([fitness(ind) for ind in population])
    best_fitness.append(scores.max())

    selected = population[np.argsort(scores)[-POPULATION//2:]]

    children = []
    while len(children) < POPULATION:
        p1, p2 = selected[np.random.randint(len(selected), size=2)]
        cp = np.random.randint(1, CHROM_LENGTH)
        child = np.concatenate((p1[:cp], p2[cp:]))

        if np.random.rand() < 0.01:
            idx = np.random.randint(CHROM_LENGTH)
            child[idx] = 1 - child[idx]

        children.append(child)

    population = np.array(children)

st.title("Genetic Algorithm Bit Pattern Generator")
st.line_chart(best_fitness)
st.write("Final Best Fitness:", best_fitness[-1])
