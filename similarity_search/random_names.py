import random
import os

def get_random_names(filepath: str = None) -> list:
    if filepath is None:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'data', 'random_names.txt')
    
    names = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')  # assuming tab-separated
                if len(parts) >= 3:
                    names.extend(parts[1:3])  # get both names on the line
        random.shuffle(names)
        return names
    except FileNotFoundError:
        print(f"Error: Could not find random names file at {filepath}")
        return []
