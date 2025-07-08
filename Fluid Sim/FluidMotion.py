import numpy as np
from scipy.spatial import cKDTree

# fluids are naturally incompressible
""" def CalculateProperty(all_points, influence_radius, point, mass=1, property=None):
    position = [point.positionX, point.positionY]

    positions = np.array([[circle.positionX, circle.positionY] for circle in all_points])


    # This is to only look at particles inside the smoothing radius
    mask = (np.linalg.norm(positions - position, axis=1) <= influence_radius)

    filtered_positions = positions[mask]
    density = CalculateDensity(all_points, influence_radius, point, mass)

    distances = np.linalg.norm(position - filtered_positions, axis=1)
    influences = SmoothingFunc(influence_radius, distances) * mass * property/density

    new_property = np.sum(influences)
    return new_property """

def CalculateDensity(point, distances, mass=1):
    # Only check the points inside of the smoothing radius as an improvement
    influences = SmoothingFunc(distances, point.smoothing_radius) * mass
    density = np.sum(influences)

    return density

# Smoothing Functions
def SmoothingFunc(dst, smoothing_radius):
    volume = 12/((smoothing_radius**4) * np.pi)
    smoothed = (smoothing_radius - dst) * (smoothing_radius - dst) * volume
    return smoothed

def LapLacianSmoothingFunc(dst, smoothing_radius): 
    zero_mask = (dst == 0)
    dst[zero_mask] = 1e-5
    volume = 12/((smoothing_radius**2) * np.pi)
    smoothed = (2/volume)*(1-(smoothing_radius-dst)/dst)

    return smoothed

def kernel_gradient(dst, smoothing_radius):

    smoothing_mask = (dst > 0)
    inv_smoothing_mask = (dst < 0)
    smoothed = np.zeros_like(dst)

    volume = 12/((smoothing_radius**4) * np.pi)

    # Regular kernel gradient
    smoothed[smoothing_mask] = -2*(smoothing_radius - dst[smoothing_mask])*volume
    # inverted kernel gradient for negative numbers
    return smoothed

def kernel_gradient_alt(dst, smoothing_radius):
    volume = 12/((smoothing_radius**4) * np.pi)
    # Regular kernel gradient
    smoothed = -2*(smoothing_radius - dst)*volume
    # inverted kernel gradient for negative numbers
    return smoothed

def vector_gradients(vector_distance, distances, smoothing_radius):
    # Ensure no zero division
    distances = np.maximum(distances, 1e-6)
    
    # Normalized direction vectors: r̂_ij = r_ij / ||r_ij||
    directions = vector_distance / distances[:, None]  # shape (N, 2)

    # Use your existing kernel_gradient function
    grad_W_scalar = kernel_gradient(distances, smoothing_radius)  # shape (N,)

    # Multiply scalar gradient by direction to get vector gradient
    grad_W = directions * grad_W_scalar[:, None]  # shape (N, 2)

    return grad_W


def calculatePressure(density, k=1*10**4, rest_density = 0.1, adiabatic_const=7):

    # Pressure Multiplier goes here
    #pressure_multiplier = 100
    #pressure = max(0, (density-rest_density) * pressure_multiplier)
    pressure = k*(((density/rest_density) ** adiabatic_const) - 1)
    return pressure
'''
def PressureForceCalc(filtered_circles, filtered_positions, position, distances, point, mass=1):
    primary_density = point.density
    
    # preassigning list memory
    pressure_and_density = np.empty((len(filtered_circles), 2))

    for i, particle in enumerate(filtered_circles):
        pressure_and_density[i, 0] = calculatePressure(particle.density)
        pressure_and_density[i, 1] = particle.density

    pressures = pressure_and_density[:, 0]
    densities = pressure_and_density[:, 1]

    primary_pressure = sharedPressureValue(primary_density, densities)

    
    vector_distance = position - filtered_positions
    directions = vector_distance/distances[:, None]
    grad_W = kernel_gradient(distances, point.smoothing_radius)[:, None]

    grad_W = grad_W * directions


    coeffs = -((mass * (primary_pressure/(primary_density**2)) + (pressures/(densities**2)))[:, None] * grad_W)
    total_pressure_force = np.sum(coeffs, axis=0)


    return total_pressure_force  # Now a vector [fx, fy]
'''
def PressureForceCalc(filtered_circles, filtered_positions, position, distances, point, mass=1):
    # More theoretically correct way, this'll be a bit slower but it shouldnt make a difference if what i did previously was correct
    primary_density = point.density
    total_force = np.array([0,0], dtype=float)

    if (primary_density == 0 ):
        return 0
    
    for particle in filtered_circles:
        particle_density = particle.density
        primary_pressure = sharedPressureValue(particle_density, primary_density)
        particle_pressure = primary_pressure

        # Force coefficient term
        #pressure_force_coeff = (primary_pressure/(primary_density**2)) + (particle_pressure/(particle_density**2))
        pressure_force_coeff = (primary_pressure + particle_pressure)/particle_density
        # Distance vectors
        particle_position = np.array([particle.positionX, particle.positionY])
        distance = np.linalg.norm(particle_position-np.array(position))
        if (distance > 0): 
            # Normalised direction, normalised as further points will have higher vectors than closer which is opposite of desired effect
            direction = (position - particle_position)/distance

            # Gradient func
            W_grad = kernel_gradient_alt(distance, particle.smoothing_radius)
            
            directional_derivative = direction * W_grad
            pressure_force = -directional_derivative * pressure_force_coeff * mass

            total_force += pressure_force


        # Checking high pressure points(above the resting density)
        '''        if (point.density - rest_density > 75):
            print("High Pressure: " + str(total_force))
        else:
            print("Low Pressure" + str(total_force))'''


    return total_force


        
        
    
def sharedPressureValue(densityA, densityB):
    pressureA = calculatePressure(densityA)
    pressureB = calculatePressure(densityB)
    return (pressureA + pressureB)/2

def viscosityForce(filtered_circles, filtered_positions, position, distances, point,mass=1):
    ## Make all density calculations here
    laplacian_velocity_x = 0
    laplacian_velocity_y = 0

    # Preallocating numpy array memory
    velocity = np.empty((len(filtered_circles), 2))
    
    for i, circle in enumerate(filtered_circles):
        velocity[i, 0] = circle.speedX
        velocity[i, 1] = circle.speedY

    filtered_velocities_x = velocity[:, 0]
    filtered_velocities_y = velocity[:, 1]

    viscosity_force_x = mass * (filtered_velocities_x - point.speedX) * LapLacianSmoothingFunc(distances, point.smoothing_radius)
    viscosity_force_y = mass * (filtered_velocities_y - point.speedY) * LapLacianSmoothingFunc(distances, point.smoothing_radius)

    laplacian_velocity_x = point.viscosity * np.sum(viscosity_force_x)
    laplacian_velocity_y = point.viscosity * np.sum(viscosity_force_y)

    point_density = point.density

    pressure_vector = PressureForceCalc(filtered_circles, filtered_positions, position, distances, point)

    
    acceleration_i_x = 1*(laplacian_velocity_x + pressure_vector[0])/ point_density
    acceleration_i_y = 1*(laplacian_velocity_y + pressure_vector[1])/ point_density

    ## update the existing vectors
    point.vectorX = acceleration_i_x
    point.vectorY = acceleration_i_y + 800 * 1/60

