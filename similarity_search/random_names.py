import random
import os

def get_random_names(filepath: str = None) -> list:
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
