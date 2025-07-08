
import numpy as np



cell_size = 80

def hash_position(pos):
    # pos: (x, y)
    return (int(pos[0] // cell_size), int(pos[1] // cell_size))

def insert_particle(grid, particle):
    cell = hash_position((particle.positionX, particle.positionY))
    if cell not in grid:
        grid[cell] = []
    grid[cell].append(particle)

def get_neighbors(grid, particle):
    cx, cy = hash_position((particle.positionX, particle.positionY))
    neighbors = []
    # Check the particle's cell and all adjacent cells (3x3 neighborhood)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cell = (cx + dx, cy + dy)
            neighbors.extend(grid.get(cell, []))
    return neighbors