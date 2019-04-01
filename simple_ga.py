import scipy as sp
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# https://gist.github.com/josephmisiti/940cee03c97f031188ba7eac74d03a4f
class Simple_GA:
    #
    # Global variables
    # Setup optimal string and GA input variables.
    #
    def __init__(self, pop_size, n_generations, matrix, classes):
        # self.start_indiv     = start_indiv
        self.individual_size    = np.shape(matrix)[1]
        self.pop_size    = pop_size
        self.n_generations = n_generations
        self.matrix = matrix
        self.classes = classes

    #
    # Main driver
    # Generate a population and simulate GENERATIONS generations.
    #
    def executeGA(self):
        # Generate initial population. This will create a list of POP_SIZE strings,
        # each initialized to a sequence of random characters.
        population = self.random_population()

        # Simulate all of the generations.
        for generation in range(self.n_generations):
            print("Generation "+str(generation))
            
            weighted_population = []

            # Add individuals and their respective fitness levels to the weighted
            # population list. This will be used to pull out individuals via certain
            # probabilities during the selection phase. Then, reset the population list
            # so we can repopulate it after selection.
            for individual in population:
                fitness_val = self.fitness(individual)

                # Generate the (individual,fitness) pair, taking in account whether or
                # not we will accidently divide by zero.
                if fitness_val == 0:
                    pair = (individual, 0.0001)
                else:
                    pair = (individual, fitness_val)

                weighted_population.append(pair)

            # # Select two random individuals, based on their fitness probabilites, cross
            # # their genes over at a random point, mutate them, and add them back to the
            # # population for the next iteration.
            # for _ in range(int(self.pop_size/2)):
            
            # Selection
            ind1, id_1 = self.weighted_choice(weighted_population)
            ind2, id_2 = self.weighted_choice(weighted_population)

            # Crossover
            ind1, ind2 = self.crossover(ind1, ind2)

            # Mutate and add back into the population.
            ind1 = self.mutate(ind1)
            ind2 = self.mutate(ind2)

            if self.fitness(ind1) > weighted_population[id_1][1]:
                population[id_1] = ind1

            if self.fitness(ind2) > weighted_population[id_2][1]:
                population[id_2] = ind2

        # Display the highest-ranked string after all generations have been iterated
        # over. This will be the closest string to the OPTIMAL string, meaning it
        # will have the smallest fitness value. Finally, exit the program.
        fittest_weights = population[0]
        minimum_fitness = self.fitness(population[0])
        
        for individual in population:
            ind_fitness = self.fitness(individual)
            if ind_fitness < minimum_fitness:
                fittest_weights = individual
                minimum_fitness = ind_fitness

        print("Coefficient: "+str(self.fitness(fittest_weights)))
        return fittest_weights
    #
    # Helper functions
    # These are used as support, but aren't direct GA-specific functions.
    #
    
    def weighted_choice(self, items):
        """
        Chooses a random element from items, where items is a list of tuples in
        the form (item, weight). weight determines the probability of choosing its
        respective item. Note: this function is borrowed from ActiveState Recipes.
        """
        idx1 = np.random.randint(0,(self.pop_size-1))
        n1 = items[idx1]
        idx2 = np.random.randint(0,(self.pop_size-1))
        n2 = items[idx2]
        if n1[1] < n2[1]:
            return n1[0], idx1
        else:
            return n2[0], idx2

    def random_population(self):
        """
        Return a list of POP_SIZE individuals, each randomly generated via iterating
        DNA_SIZE times to generate a string of random characters with random_char().
        """
        pop = []
        #pop.append(self.start_indiv.tolist()[0])
        for i in range(self.pop_size):
            # pop.append(np.random.randint(2, size=self.individual_size).tolist())
            pop.append(np.random.random(size=self.individual_size).tolist())
        return pop

    #
    # GA functions
    # These make up the bulk of the actual GA algorithm.
    #

    def fitness(self, dna):
        """
        For each gene in the DNA, this function calculates the difference between
        it and the character in the same position in the OPTIMAL string. These values
        are summed and then returned.
        """
        #xInd = yInd = range(len(dna))
        #newIndividual = sp.sparse.csr_matrix((dna, (xInd, yInd)))
        mean_dist = []
        for c in np.unique(self.classes):            
            cid = [cid_ for cid_, c_ in enumerate(self.classes) if c_ == c]
            newFeatures = np.multiply(self.matrix[cid,:], dna)
            distance_mtx = pairwise_distances(newFeatures, metric="cosine", n_jobs=None)
            mean_dist.append(np.mean(distance_mtx))
        return 1 - np.mean(mean_dist)

    def mutate(self, dna):
        """
        For each gene in the DNA, there is a 1/mutation_chance chance that it will be
        switched out with a random character. This ensures diversity in the
        population, and ensures that is difficult to get stuck in local minima.
        """
        mutation_chance = 1
        rand_pos = np.random.randint(self.individual_size-1, size=int((self.individual_size-1)/100))
        for pos in rand_pos:
            if int(np.random.random()*mutation_chance) == 1:
                dna[pos] = np.random.random()
        return dna

    def crossover(self, dna1, dna2):
        """
        Slices both dna1 and dna2 into two parts at a random index within their
        length and merges them. Both keep their initial sublist up to the crossover
        index, but their ends are swapped.
        """
        pos = int(np.random.random()*(self.individual_size))
        arr1 = dna1[:pos]+dna2[pos:]
        arr2 =  dna2[:pos]+dna1[pos:]
        return (arr1, arr2)
