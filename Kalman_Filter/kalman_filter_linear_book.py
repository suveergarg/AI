# Solution to 3.8.1

import numpy as np
import matplotlib.pyplot as plt
from filterpy.stats import plot_gaussian_pdf, plot_covariance_ellipse


def process_update(A, B, mu, cov, u, R):
    '''
    mu : n*1 vector (from previous timestep)
    cov : n*n matrix (from previous timestep)
    u : k*1 vector (control input)

    A : n*n State evolution matrix
    B : n*k Control Matrix 

    R : Process noise

    Returns mu & cov after process update
    '''

    new_mu = A @ mu + B @ u
    new_cov = A @ cov @ A.T + R

    return new_mu, new_cov

def measurement_update(mu, cov, z, C, Q):
    '''
    mu : n*1 vector (output from process update)
    cov : n*n matrix (output from process update)

    z : m*1 measurement vector
    C : m*n measurement matrix
    Q : Observation noise

    Return mu and cov after measurement update
    '''
    K = cov @ C.T @ np.linalg.inv(C @ cov @ C.T + Q)
    
    new_mu = mu + K @ (z - C @ mu)
    new_cov = (np.eye(cov.shape[0]) - K @ C) @ cov
    
    return new_mu, new_cov

##################################
# Solution to 3.8.1 and 3.8.2 ####
##################################
timestep = 1
A = np.array([[1, timestep],
              [0, 1]])

B = np.array([[0.5 * timestep**2],
              [timestep]])

C = np.array([[1, 0]])

mu = np.array([[0],
               [0]])

cov = np.eye(2)

steps = 5
control_input_variance = 1
control_input_std_dev = np.sqrt(control_input_variance)

measurement_variance = 10
measurement_std_dev = np.sqrt(10)

Q = measurement_variance

# Very specific to this example : 
# https://github.com/pptacher/probabilistic_robotics/blob/master/ch3_gaussian_filters/ch3_gaussian_filters.pdf
R = control_input_variance * B @ B.T 

print(R)

position_subplot = plt.subplot(311)
velocity_subplot = plt.subplot(312)
two_d_subplot = plt.subplot(313)

for step in range(1, steps+1):
    u = np.abs(np.random.normal(0, control_input_std_dev, size = (1,1)))
    mu, cov = process_update(A, B, mu, cov, u, R)

    # Creating a measurement value
    z = C @ mu + np.random.normal(0, measurement_std_dev, size = (1,1))

    plot_gaussian_pdf(mean= mu[0, 0], variance= cov[0, 0], 
                  xlabel='Position of car', ylabel='pdf', xlim= (-2, 20), ylim= (0, 1), ax = position_subplot)
    
    plot_gaussian_pdf(mean= mu[1, 0], variance= cov[1, 1], 
                  xlabel='Velocity of car', ylabel='pdf', xlim= (-2, 20), ylim= (0, 1), ax = velocity_subplot)
    
    plot_covariance_ellipse((mu[0, 0], mu[1, 0]), cov, fc='g', alpha=0.2, std=[1])

    mu, cov = measurement_update(mu, cov, z, C, Q)


    if(step == 5):

        plot_gaussian_pdf(mean= mu[0, 0], variance= cov[0, 0], 
                    xlabel='Position of car', ylabel='pdf', xlim= (-2, 20), ylim= (0, 1), ax = position_subplot)
        
        plot_gaussian_pdf(mean= mu[1, 0], variance= cov[1, 1], 
                    xlabel='Velocity of car', ylabel='pdf', xlim= (-2, 20), ylim= (0, 1), ax = velocity_subplot)
        

        plot_covariance_ellipse((mu[0, 0], mu[1, 0]), cov, fc='b', alpha=0.2, std=[1])

    print(mu)
    print(cov)

    
plt.show()


