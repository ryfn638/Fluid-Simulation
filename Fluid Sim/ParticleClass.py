import numpy as np
import random
# Constant Tick rate
FRAME_RATE = 60
GRAVITY_RATE = 800

# Particle class object
class Particle():
    def __init__(self, radius, vectors, position, width, height, smoothing_radius):
        # particles are cirlces so this is the radius
        self.radius = radius
        # Positional Tokens
        self.positionX = position[0]
        self.positionY = position[1]

        # Current vectors for movement, this variable is changed over time
        # Vectors are in the form newton Force Units so acceleration. . This sim will run at 60fps. so 1 tick is 0.016667 seconds
        self.vectorX = vectors[0]
        self.vectorY = vectors[1]

        self.speedX = 0 #random.uniform(-500, 500)
        self.speedY = 0

        self.mass = 1
        self.viscosity = 1e-4
        self.density = 1e-5
        self.pressure = None
        self.pressureForce = None

        self.smoothing_radius = smoothing_radius
        # maybe can update this later we'll see
        self.MINWIDTH = 0
        self.MINHEIGHT = 0


        self.MAXWIDTH = width
        self.MAXHEIGHT = height

        self.gravity_rate = 0
        self.terminal_velocity = 800
        

    def ChangeVector(self, vectorX, vectorY):
        # When we want add a new vector
        # since this is a fluid sim we want gravity to constantly affect our vectors hence why adding is done and not simply reassigning
        self.vectorX += vectorX
        self.vectorY += vectorY

    def MoveTick(self):
        # At 60fps
        
        tickrate = 1/FRAME_RATE
        # Update speed with the acceleration
        self.speedX = 10*self.vectorX * tickrate
        self.speedY = 10*self.vectorY * tickrate

        self.gravity_rate += 1*GRAVITY_RATE * tickrate

        self.gravity_rate = min(self.gravity_rate, self.terminal_velocity)

        self.positionX += tickrate * self.speedX
        self.positionY += tickrate * self.speedY

        #print(GRAVITY_RATE * tickrate * tickrate, tickrate*self.speedY)
        # Bound to the shape of the window. Kill the speed if the thing hits a wall
        self.positionX = max(min(self.positionX, self.MAXWIDTH+self.radius), self.MINWIDTH+self.radius)
        self.positionX = max(min(self.positionX, self.MAXWIDTH-self.radius), self.MINWIDTH-self.radius)

        self.collision_dampening = 0.5



        # Setting speed to 0 if the particle hits a wall
        if self.positionX == self.MAXWIDTH+self.radius or self.positionX == self.MINWIDTH+self.radius:
            self.speedX = -self.speedX*self.collision_dampening
        elif self.positionX == self.MAXWIDTH-self.radius or self.positionX == self.MINWIDTH-self.radius:
            self.speedX = -self.speedX*self.collision_dampening

        # Same thing with the Y speeds
        self.positionY = max(min(self.positionY, self.MAXHEIGHT+self.radius), self.MINHEIGHT+self.radius)
        self.positionY = max(min(self.positionY, self.MAXHEIGHT-self.radius), self.MINHEIGHT-self.radius)

        if self.positionY == self.MAXHEIGHT + self.radius or self.positionY == self.MINHEIGHT + self.radius:
            self.speedY = -self.speedY*self.collision_dampening
        elif self.positionY == self.MAXHEIGHT - self.radius or self.positionY == self.MINHEIGHT - self.radius:
            self.speedY = -self.speedY*self.collision_dampening

def ListIndexToCoordinate(width, index_num):
    y = np.floor(index_num/width)
    x = index_num - y*width
    return x,y

def gridPosition(index, width, height, total_shapes):
    # how many points fit on one thing
    grid_width = int(np.sqrt(total_shapes))
    grid_height = int(np.sqrt(total_shapes))

    x_index, y_index = ListIndexToCoordinate(grid_width, index)

    # 10% padding of the window of the way into the window
    start_point_width = 0.1* width
    end_point_width = 0.9*width

    start_point_height = 0.5 * height
    end_point_height = 0.1*height

    dist_height = start_point_height - end_point_height
    dist_width = end_point_width - start_point_width

    position_x = start_point_width + x_index*(dist_width/grid_width)
    position_y = start_point_height - y_index * (dist_height/grid_height)
    return position_x, position_y


# Create a bunch of particles in a grid pattern.
def createParticles(num_particles, width, height, radius):
    all_particles = []
    smoothing_radius = np.sqrt(width * height/num_particles)
    for particle in range(num_particles):
        #particle_pos = #gridPosition(particle, width, height, num_particles)
        particle_pos = [random.randrange(0, width), random.randrange(0, height)]
        # at first the new particles will have no vectors but gravity, we're just trying to make it fall first.
        all_particles.append(Particle(radius, [0, GRAVITY_RATE], particle_pos, width, height, smoothing_radius))

    return all_particles

def EuclideanDistance(p1, p2):
    # get the eucliden distance between two points
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

