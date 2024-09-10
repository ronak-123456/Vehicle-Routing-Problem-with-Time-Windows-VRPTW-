import random
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Customer:
    def __init__(self, id: int, x: float, y: float, demand: int, ready_time: int, due_time: int, service_time: int):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

class Vehicle:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.route: List[Customer] = []
        self.load = 0
        self.distance = 0
        self.time = 0

class VRPTWOptimizer:
    def __init__(self, customers: List[Customer], vehicles: List[Vehicle], depot: Customer):
        self.customers = customers
        self.vehicles = vehicles
        self.depot = depot
        self.best_solution: List[Vehicle] = []
        self.best_fitness = float('inf')

    def distance(self, c1: Customer, c2: Customer) -> float:
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

    def is_feasible(self, vehicle: Vehicle, customer: Customer) -> bool:
        if vehicle.load + customer.demand > vehicle.capacity:
            return False

        arrival_time = vehicle.time + self.distance(vehicle.route[-1], customer)
        if arrival_time > customer.due_time:
            return False

        return True

    def calculate_fitness(self, solution: List[Vehicle]) -> float:
        total_distance = sum(v.distance for v in solution)
        total_vehicles = len([v for v in solution if v.route])
        return total_distance + 1000 * total_vehicles  # Penalize using more vehicles

    def generate_initial_solution(self) -> List[Vehicle]:
        solution = [Vehicle(v.capacity) for v in self.vehicles]
        unassigned = self.customers.copy()
        random.shuffle(unassigned)

        for customer in unassigned:
            assigned = False
            for vehicle in solution:
                if not vehicle.route:
                    vehicle.route = [self.depot]

                if self.is_feasible(vehicle, customer):
                    vehicle.route.append(customer)
                    vehicle.load += customer.demand
                    vehicle.distance += self.distance(vehicle.route[-2], customer)
                    vehicle.time = max(customer.ready_time, vehicle.time + self.distance(vehicle.route[-2], customer)) + customer.service_time
                    assigned = True
                    break

            if not assigned:
                # If no vehicle can accommodate the customer, create a new vehicle
                new_vehicle = Vehicle(solution[0].capacity)
                new_vehicle.route = [self.depot, customer]
                new_vehicle.load = customer.demand
                new_vehicle.distance = self.distance(self.depot, customer)
                new_vehicle.time = max(customer.ready_time, self.distance(self.depot, customer)) + customer.service_time
                solution.append(new_vehicle)

        return solution

    def crossover(self, parent1: List[Vehicle], parent2: List[Vehicle]) -> List[Vehicle]:
        child = [Vehicle(v.capacity) for v in self.vehicles]
        all_customers = set(customer for vehicle in parent1 for customer in vehicle.route if customer != self.depot)

        # Inherit routes from parent1
        crossover_point = random.randint(1, len(parent1) - 1)
        for i in range(crossover_point):
            child[i].route = parent1[i].route.copy()
            for customer in child[i].route:
                if customer != self.depot:
                    all_customers.remove(customer)

        # Fill remaining customers from parent2
        for vehicle in parent2:
            for customer in vehicle.route:
                if customer in all_customers:
                    for child_vehicle in child:
                        if self.is_feasible(child_vehicle, customer):
                            child_vehicle.route.append(customer)
                            all_customers.remove(customer)
                            break

        # Assign any remaining customers
        for customer in all_customers:
            for vehicle in child:
                if self.is_feasible(vehicle, customer):
                    vehicle.route.append(customer)
                    break

        return child

    def mutate(self, solution: List[Vehicle]) -> List[Vehicle]:
        if random.random() < 0.5:
            # Swap two random customers
            v1, v2 = random.sample(solution, 2)
            if len(v1.route) > 1 and len(v2.route) > 1:
                i, j = random.randint(1, len(v1.route) - 1), random.randint(1, len(v2.route) - 1)
                v1.route[i], v2.route[j] = v2.route[j], v1.route[i]
        else:
            # Move a random customer to another vehicle
            v1 = random.choice(solution)
            if len(v1.route) > 1:
                i = random.randint(1, len(v1.route) - 1)
                customer = v1.route.pop(i)
                v2 = random.choice(solution)
                j = random.randint(1, len(v2.route))
                v2.route.insert(j, customer)

        return solution

    def local_search(self, solution: List[Vehicle]) -> List[Vehicle]:
        improved = True
        while improved:
            improved = False
            for i, vehicle in enumerate(solution):
                for j in range(1, len(vehicle.route) - 1):
                    for k in range(i, len(solution)):
                        for l in range(1 if i == k else 0, len(solution[k].route)):
                            if i == k and abs(j - l) <= 1:
                                continue
                            new_solution = self.swap_customers(solution, i, j, k, l)
                            new_fitness = self.calculate_fitness(new_solution)
                            if new_fitness < self.calculate_fitness(solution):
                                solution = new_solution
                                improved = True
        return solution

    def swap_customers(self, solution: List[Vehicle], i: int, j: int, k: int, l: int) -> List[Vehicle]:
        new_solution = [Vehicle(v.capacity) for v in solution]
        for m, vehicle in enumerate(solution):
            new_solution[m].route = vehicle.route.copy()

        new_solution[i].route[j], new_solution[k].route[l] = new_solution[k].route[l], new_solution[i].route[j]
        return new_solution

    def optimize(self, generations: int, population_size: int) -> List[Vehicle]:
        population = [self.generate_initial_solution() for _ in range(population_size)]

        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = [self.calculate_fitness(solution) for solution in population]

            # Select parents
            parents = random.choices(population, weights=[1/score for score in fitness_scores], k=population_size)

            # Create new population
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                child1, child2 = self.mutate(child1), self.mutate(child2)
                new_population.extend([child1, child2])

            # Local search
            new_population = [self.local_search(solution) for solution in new_population]

            # Elitism: keep the best solution from the previous generation
            best_solution = min(population, key=self.calculate_fitness)
            new_population[0] = best_solution

            population = new_population

            # Update best solution
            current_best = min(population, key=self.calculate_fitness)
            current_best_fitness = self.calculate_fitness(current_best)
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best

        return self.best_solution

    def visualize_solution(self, solution: List[Vehicle]):
        plt.figure(figsize=(12, 8))
        plt.scatter([c.x for c in self.customers], [c.y for c in self.customers], c='blue', label='Customers')
        plt.scatter([self.depot.x], [self.depot.y], c='red', marker='s', s=200, label='Depot')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
        for vehicle, color in zip(solution, colors):
            if vehicle.route:
                route_x = [c.x for c in vehicle.route]
                route_y = [c.y for c in vehicle.route]
                plt.plot(route_x, route_y, c=color)

        plt.title("VRPTW Solution Visualization")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
import numpy as np

def generate_random_customers(num_customers: int) -> List[Customer]:
    customers = []
    for i in range(num_customers):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        demand = random.randint(1, 20)
        ready_time = random.randint(0, 100)
        due_time = ready_time + random.randint(50, 100)
        service_time = random.randint(10, 30)
        customers.append(Customer(i+1, x, y, demand, ready_time, due_time, service_time))
    return customers

def main():
    num_customers = 50
    num_vehicles = 10
    vehicle_capacity = 100

    depot = Customer(0, 50, 50, 0, 0, 1000, 0)
    customers = generate_random_customers(num_customers)
    vehicles = [Vehicle(vehicle_capacity) for _ in range(num_vehicles)]

    optimizer = VRPTWOptimizer(customers, vehicles, depot)
    best_solution = optimizer.optimize(generations=100, population_size=50)

    print(f"Best solution fitness: {optimizer.best_fitness}")
    for i, vehicle in enumerate(best_solution):
        if vehicle.route:
            print(f"Vehicle {i+1} route: {' -> '.join(str(c.id) for c in vehicle.route)}")

    optimizer.visualize_solution(best_solution)

if __name__ == "__main__":
    main()
