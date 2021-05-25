#!/usr/bin/env python
# coding: utf-8
from random import choices, randint, randrange, random, sample 
from collections import namedtuple
from typing import List, Optional, Callable, Tuple
#Choices(, , # of draws from the population (2 , a pair)) is a function from the random module of python

###########################################################################
#========================> 0.intialisation #==============================>
##########################################################################
#1. Initilize genomes, ie, candidate solutions placeholder list and specify type
Genome = List[int] #list of 1 and 0 ?? ; need to update !!

#2.Initiliase Population, ie multiple candidate solution placeholder list with genome type   
Population = List[Genome] # list of candidate solutions generation=pop={solu1, ..., soln}={geno1, ..., genon} 

###########################################################################
#========================> 1.Callable functions #==============================>
###########################################################################
#callable fn Use functions as parameters for abstracting the problem from the algo
PopulateFunc = Callable[[], Population] # a population fn that takes nothing and returns new solutiobns #=Callable[[Input], Output]
FitnessFunc = Callable[[Genome], int] # a fitness function that takes a genome and returns a fitness value to make the correct choice
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]  #takes a population and a fitness fn to select 2 solutions to be the parents of our next generaation solution
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]] #takes 2 genomes and returns 2 new genomes 
MutationFunc = Callable[[Genome], Genome] #takes 1 genome and sometimes returns a modified one
PrinterFunc = Callable[[Population, int, FitnessFunc], None]



###########################################################################
#========================> 1.Encoding and Generating Genoms #==============================>
###########################################################################
#######How to encode vessel speed and number of vessels #############
def generate_genome(length: int) -> Genome:
    '''A function to encode and generate genomes for one solution of vessel speed and number of vessels
        k is length of the genome'''
    return choices([0,1], k=length)


###########################################################################
#========================> 2.Generating Populations #==============================>
###########################################################################
def generate_population(size: int, genome_length: int) -> Population: 
    '''a population is a list of genomes
    a function to generate a population by calling generate genome multiple times 
    until our population has the desired size'''
    return [generate_genome(genome_length) for _ in range(size)]

###########################################################################
#======================>3.Single point Crossover function #=========================>
###########################################################################
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    '''generate a new solution for the next generation'''
    '''takes 2 genomes as parameters'''
    '''returns 2 genomes as output'''
    # ====> Crosseover function onlly makes senses if 
    # Gneomes need to be the same length for this to work
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")
    
    # lengths of genome is at least 2 otherwise there would be no point to cut the genome in half
    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1) #randomly choose an index to cut the genomes in half it 
    return a[0:p] + b[p:], b[0:p] + a[p:] # the first new solution =  the first 1/2 of the 1st genome a and the rest of the second genome b
#2nd new solution: 1st part of genome b and 2nd part of genome a

###########################################################################
#==============================> 4.Mutation function #==============================>
###########################################################################
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    '''takes a genome and w/ a certain proba changes 1 to 0 and 0 to 1 @ random positions'''
    ''''''
    for _ in range(num):
        index = randrange(len(genome)) # generate a random index
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1) #if random returns a value>> prob, we leave it alone otherwise it 
        #falls within the mutation proba and we need to change it into the | current value - 1|
    return genome 


###########################################################################
#==============================> 5.Population Fitness function #==============================>
###########################################################################
def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    '''the fitness fn determm how good a candidate given solution (genome) is'''
    '''insert the subroutine constraint'''
    return sum([fitness_func(genome) for genome in population])
#update this selection fn with a fitness fn that takes a genome and returns a fitness value to ake the correct choice
#keep in mind genome: Genome, things: [Thing], weight_limit: int are problem specific parameter for the knapsack


###########################################################################
#==============================> 6.Selection function #==============================>
###########################################################################
#fitness_func(gene) parameter 
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    '''select the solution to generate the next generation '''
    '''select a pair of solutions which will be the parent of 2 new solutions of the next generation 
    Solution with higher fitness should be more likely to be chosen for reproduction'''
    #choices fn from python random module allow us to assign the fitness as weights for each element it can choose from
    #fitness of a genome = the genome's weights
    # k=2, we draw 2 from our population
    return choices(population=population,
                   weights=[fitness_func(gene) for gene in population],
                   k=2
                  )

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return sample(
        population=generate_weighted_distribution(population, fitness_func),
        k=2
    )


def generate_weighted_distribution(population: Population, fitness_func: FitnessFunc) -> Population:
    result = []

    for gene in population:
        result += [gene] * int(fitness_func(gene)+1)

    return result

###########################################################################
#==============================> Results printing #==============================>
###########################################################################
def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


###########################################################################
#==============================> Run algorithm #==============================>
###########################################################################

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    i = 0
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i

#########################8.Evolution function ###########################################
def run_evolution(populate_func: PopulateFunc,
                  fitness_func: FitnessFunc,
                  fitness_limit: int,
                  selection_func: SelectionFunc = selection_pair,
                  crossover_func: CrossoverFunc = single_point_crossover,
                  mutation_func: MutationFunc = mutation,
                  generation_limit: int = 100,
                  printer: Optional[PrinterFunc] = None) \
               -> Tuple[Population, int]:
    '''parameters:
        generation_limit : max # of generation our evolution runs for if it s not reaching the fitness limit before then ''' 
    
####=======>1. generate the 1st ever generation by calling the populate function
    population = populate_func()
    
####=======>2. loop for generation limit time 
    for i in range(generation_limit):
        population = sorted(population, 
                            key=lambda genome: fitness_func(genome), 
                            reverse=True) #Sort population by fitness so that top solutions are in the first indices of our list of genomes 
    
        if printer is not None:
            printer(population, i, fitness_func)
            
        if fitness_func(population[0]) >= fitness_limit: #check if we have already reached the firtness limit and return early from our loop or if we want to implement elitism  
            break # return early 
            
        next_generation = population[0:2] #or Implement elitism, ie keep our top 2 solutions for our next generation

        #-----------------------Step 2:  generate all other new solutions for our next generations-------
        #pick 2 parents and get 2 new solutions everytime
        # ==> loop for 1/2 the length of a genration to get as many solutions in our next generation as
        #as before but b/c we already copied the 2 top solutions fron our last generation we can 
        #save one more loop 
        for j in range(int(len(population) / 2) - 1):
        #'''in each loop, we call the selection fn to get our parents'''
            parents = selection_func(population, fitness_func)
        #'''putting the parents in the crossover fn to get 2 child solutions for our next generation '''
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
        #'''To generate our next generation, we apply the mutuation fn for each offspring to expand the variety of the solutions we generate, '''
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
#Replace the current population with our next generation and start into the next round of the algo
#by sorting the population and checking if we reached our fitness limit 
        population = next_generation
    
    #Step 3: Sort population one lase time in case we run out of generation  
    population = sorted(population, 
                        key=lambda genome: fitness_func(genome), 
                        reverse=True) 

    return population, i #the i is to distinguish wether the algorithm exausted the generation limit or actually found the solultion that meets our fitness criteria 