import pygame
import numpy
from sys import exit
import ParticleClass
import time
import FluidMotion
import numpy as np
import cProfile
from scipy.spatial import cKDTree
import Hashgrid
import DrawArrow



WIDTH = 300
HEIGHT = 300
FRAME_RATE = 1/60
INFLUENCE_RADIUS = 80
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

pygame.display.set_caption("Fluid Simulation 2D")

all_particles = ParticleClass.createParticles(100, WIDTH, HEIGHT, 3)



def blur_surface(surface, amount=8):
    scale = 1.0 / amount
    small = pygame.transform.smoothscale(surface, (int(surface.get_width() * scale), int(surface.get_height() * scale)))
    return pygame.transform.smoothscale(small, surface.get_size())



# Main Pygame Loop
def main():
    spatial_grid = {}
    while True:
        screen.fill((0, 0, 0))
        surface.fill((0, 0, 0))
        filtered_circles = []
        filtered_positions = [] 
        all_distances = []
        density_mask = []
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        spatial_grid.clear()
    
        # Insert all particles into grid
        for p in all_particles:
            Hashgrid.insert_particle(spatial_grid, p)
        
        # Now query neighbors for each particle efficiently
        for particle in all_particles:
            neighbors = np.array(Hashgrid.get_neighbors(spatial_grid, particle))

            neighbors_pos = np.empty((len(neighbors), 2))
    
            for i, circle in enumerate(neighbors):
                neighbors_pos[i, 0] = circle.positionX
                neighbors_pos[i, 1] = circle.positionY
            
            particle_position = (particle.positionX, particle.positionY)

            dsts = np.linalg.norm(neighbors_pos - particle_position, axis=1)
            filtered_mask = (dsts <= particle.smoothing_radius) & (dsts > 0)
            filtered_density_mask = (dsts <= particle.smoothing_radius)

            density_mask.append(filtered_density_mask)
            filtered_circles.append(neighbors[filtered_mask])
            filtered_positions.append(neighbors_pos[filtered_mask, :])
            all_distances.append(dsts[filtered_mask])



        # Assigning filtered_circles as a numpy list
        idx = 0
        for particle in all_particles:
            # Mask out points that are relevant for density function
            
            filtered_circles_particle = filtered_circles[idx]
            filtered_positions_particle = filtered_positions[idx]
            position = [particle.positionX, particle.positionY]

            #print(len(filtered_circles_particle))
            if (len(filtered_circles_particle) != 0):
                particle.density = FluidMotion.CalculateDensity(particle, density_mask[idx])
            else:
                ## Apply gravity
                particle.vectorY = 0
            idx += 1



        idx = 0
        for particle in all_particles:
            # Mask out points that are relevant for density function
            filtered_circles_particle = filtered_circles[idx]
            filtered_positions_particle = filtered_positions[idx]
            position = [particle.positionX, particle.positionY]
            ## Apply relevant vectors
            FluidMotion.viscosityForce(filtered_circles_particle, filtered_positions_particle, position,  all_distances[idx], particle, mass=1)
            idx += 1

        # handling pushing and pulling forces
        mouse_pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEMOTION:
            # Check if a button is still held while moving
            buttons = pygame.mouse.get_pressed()
            if buttons[0]:
                pushForce(mouse_pos, all_particles)
            if buttons[2]:
                pullForce(mouse_pos, all_particles)

        for particle in all_particles:
            particle.MoveTick()
            
            # Making particles change colour based on their pressure, high pressure = red, low pressure = blue
            resting_pressure = 0.25
            particle_pressure = particle.density
            #print(particle_pressure)
            #Red to blue ratio

            
            proportional_pressure = particle_pressure-resting_pressure
            #print(proportional_pressure, particle_pressure)
            proportion_sigmoid = sigmoid(proportional_pressure)

            red_amount = proportion_sigmoid*255
            blue_amount = (1-proportion_sigmoid)*255

            # Draw particle
            pygame.draw.circle(screen, (red_amount,0, blue_amount), (particle.positionX, particle.positionY), 3)


            # Draw smoothing Radius
            pygame.draw.circle(surface, [red_amount,0, blue_amount, 50], (particle.positionX, particle.positionY), particle.smoothing_radius)

            particle_pos = pygame.Vector2()
            particle_pos.x = particle.positionX
            particle_pos.y = particle.positionY

            particle_dir = pygame.Vector2()
            particle_dir.x = particle.positionX + particle.vectorX
            particle_dir.y = particle.positionY + particle.vectorY


            #DrawArrow.draw_arrow(screen, particle_pos, particle_dir,[red_amount,30, blue_amount])



        screen.blit(surface, (0, 0))
        pygame.display.update()
        # END OF LOOP HERE
        elapsed_time = time.time() - current_time
        sleep_time = FRAME_RATE - elapsed_time


        # sleep the remainde if positive otherwise just dont at all. if result is negative its running slower than anticipated.
        if sleep_time > 0:
            time.sleep(sleep_time)

def sigmoid(x):
    return 1/(1+(np.e**-(10*x)))

def pushForce(click_position, particles, radius=100):
    # Filtering out relevant particles

    particles_pos = np.empty((len(particles), 2))

    for i, circle in enumerate(particles):
        particles_pos[i, 0] = circle.positionX
        particles_pos[i, 1] = circle.positionY


    dsts = np.linalg.norm(particles_pos - click_position, axis=1)
    filtered_mask = (dsts <= radius) & (dsts > 0)
    
    # Nearby Particles
    distances = dsts[filtered_mask]

    # particles closer get pushed further and further is pushed less
    push_vector = distances[:, None]/(click_position-particles_pos[filtered_mask])

    # Tuneable dampening factor
    push_dampening = 1

    total_push = -push_dampening * push_vector

    for i, particle in enumerate(np.array(particles)[filtered_mask]):
        particle.positionX += total_push[i, 0]
        particle.positionY += total_push[i, 1]


def pullForce(click_position, particles, radius=100):
    # Filtering out relevant particles

    particles_pos = np.empty((len(particles), 2))

    for i, circle in enumerate(particles):
        particles_pos[i, 0] = circle.positionX
        particles_pos[i, 1] = circle.positionY


    dsts = np.linalg.norm(particles_pos - click_position, axis=1)
    filtered_mask = (dsts <= radius) & (dsts > 0)
    
    # Nearby Particles
    distances = dsts[filtered_mask]

    # particles closer get pushed further and further is pushed less
    push_vector = distances[:, None]/(click_position-particles_pos[filtered_mask])

    # Tuneable dampening factor
    push_dampening = 1

    total_push = -push_dampening * push_vector

    for i, particle in enumerate(np.array(particles)[filtered_mask]):
        particle.positionX -= total_push[i, 0]
        particle.positionY -= total_push[i, 1]

cProfile.run('main()')
    
    


    
