import random
import numpy as np

# Function to calculate the Euclidean distance between two points
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Fitness Function: Calculate the total distance of the route
def total_distance(route, cities):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(cities[route[i]], cities[route[i + 1]])
    dist += distance(cities[route[-1]], cities[route[0]])  # Return to the start city
    return dist

# Roulette Wheel Selection
def roulette_wheel_selection(population, cities):
    total_fitness = sum(1 / total_distance(route, cities) for route in population)
    probabilities = [(1 / total_distance(route, cities)) / total_fitness for route in population]
    cumulative_probabilities = np.cumsum(probabilities)
    rand = random.random()
    
    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if rand <= cumulative_prob:
            return population[i]

# Order Crossover (OX) method
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted([random.randint(0, size-1), random.randint(0, size-1)])
    
    # Create offspring by preserving order of genes from parent1
    offspring = [-1] * size
    offspring[start:end+1] = parent1[start:end+1]
    
    # Fill remaining positions from parent2
    current_pos = 0
    for i in range(size):
        if offspring[i] == -1:
            while parent2[current_pos] in offspring:
                current_pos += 1
            offspring[i] = parent2[current_pos]
    return offspring

# Swap Mutation
def swap_mutation(route):
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Main function to run the Genetic Algorithm
def genetic_algorithm(cities, population_size=100, generations=500, mutation_rate=0.05):
    # Generate initial population (random routes)
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(population_size)]
    
    best_route = None
    best_distance = float('inf')

    # Start generations loop
    for generation in range(generations):
        new_population = []
        
        # Evaluate fitness and select parents using roulette wheel selection
        for _ in range(population_size // 2):
            parent1 = roulette_wheel_selection(population, cities)
            parent2 = roulette_wheel_selection(population, cities)
            
            # Crossover to produce offspring
            offspring1 = order_crossover(parent1, parent2)
            offspring2 = order_crossover(parent2, parent1)
            
            # Mutation
            if random.random() < mutation_rate:
                offspring1 = swap_mutation(offspring1)
            if random.random() < mutation_rate:
                offspring2 = swap_mutation(offspring2)
                
            new_population.extend([offspring1, offspring2])
        
        # Replace population with new generation
        population = new_population
        
        # Find the best solution in the current population
        for route in population:
            route_distance = total_distance(route, cities)
            if route_distance < best_distance:
                best_distance = route_distance
                best_route = route
        
        # Print progress (best solution in current generation)
        if generation % 100 == 0:
            print(f"Generation {generation}: Best Distance = {best_distance}")
            

    return best_route, best_distance

#  cities (coordinates for cities)
cities = [
    (0, 0),   # City 1
    (1, 8),   # City 2
    (2, 6),   # City 3
    (1, 4),   # City 4
    (3, 1)    # City 5
]

# Run Genetic Algorithm
best_route, best_distance = genetic_algorithm(cities)

# Output the best route and distance found
print("\nBest Route:", best_route)
print("Best Distance:", best_distance)
