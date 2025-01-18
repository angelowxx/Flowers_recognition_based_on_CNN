import random

# Define the integers and their corresponding probabilities
integers = [1, 2, 3, 4]
probabilities = [0.1, 0.5, 0.3, 0.1]  # Probabilities must sum to 1

for i in range(20):
    # Generate a single random integer
    r = result = random.choices(integers, weights=probabilities, k=1)[0]
    print(f'{r}, {result}')  # Outputs one of the integers
