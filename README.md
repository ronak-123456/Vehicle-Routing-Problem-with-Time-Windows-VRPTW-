# Vehicle Routing Problem with Time Windows (VRPTW) Optimizer

This project implements a sophisticated optimizer for the Vehicle Routing Problem with Time Windows (VRPTW) using a hybrid genetic algorithm with local search. The VRPTW is a complex optimization problem in logistics and supply chain management, where the goal is to find optimal routes for a fleet of vehicles to serve a set of customers within specified time windows while minimizing the total distance traveled and the number of vehicles used.

## Features

- Hybrid algorithm combining genetic algorithm and local search
- Handles multiple vehicles with capacity constraints
- Considers customer time windows and service times
- Visualizes the optimal solution using matplotlib

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ronak-123456/vrptw-optimizer.git
   cd vrptw-optimizer
   ```

2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```

## Usage

1. Import the necessary classes and functions from the `vrptw_optimizer.py` file.

2. Create a list of `Customer` objects, each with the following attributes:
   - id: unique identifier
   - x, y: coordinates
   - demand: required capacity
   - ready_time: earliest time the customer can be served
   - due_time: latest time the customer can be served
   - service_time: time required to serve the customer

3. Create a list of `Vehicle` objects with a specified capacity.

4. Create a `VRPTWOptimizer` object with the customers, vehicles, and depot.

5. Call the `optimize` method with the desired number of generations and population size.

6. Visualize the solution using the `visualize_solution` method.

Example usage:

```python
from vrptw_optimizer import Customer, Vehicle, VRPTWOptimizer

# Create customers, vehicles, and depot
customers = [Customer(1, 10, 20, 5, 50, 100, 10), ...]
vehicles = [Vehicle(100) for _ in range(10)]
depot = Customer(0, 0, 0, 0, 0, 1000, 0)

# Initialize optimizer
optimizer = VRPTWOptimizer(customers, vehicles, depot)

# Run optimization
best_solution = optimizer.optimize(generations=100, population_size=50)

# Visualize solution
optimizer.visualize_solution(best_solution)
```

## Algorithm Details

The optimizer uses a hybrid approach combining a genetic algorithm with local search:

1. Initial population generation
2. Fitness evaluation based on total distance and number of vehicles
3. Parent selection using fitness-proportionate selection
4. Crossover to create new solutions
5. Mutation to introduce diversity
6. Local search to improve solutions
7. Elitism to preserve the best solution

The algorithm continues for a specified number of generations, attempting to improve the solution in each iteration.

## Customization

You can customize the algorithm by modifying the following parameters:

- Number of generations
- Population size
- Mutation rate
- Crossover method
- Local search intensity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
