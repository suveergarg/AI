import matplotlib.pyplot as plt 
import numpy as np 
import sys  
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--initial_poses", dest = "initial_poses", 
                    default='init_node.txt', 
                    help='Filename that contains the original poses')

parser.add_option("--optimized_poses", dest = "optimized_poses", 
                    default="after_opt.txt", 
                    help = "Filename that contains optimized poses")

(options, args) = parser.parse_args()

# Read the original and optimized poses files.
poses_original = None
if options.initial_poses != '':
  poses_original = np.genfromtxt(options.initial_poses, usecols = (1, 2))

poses_optimized = None
if options.optimized_poses != '':
  poses_optimized = np.genfromtxt(options.optimized_poses, usecols = (1, 2))

# Plots the results for the specified poses.
plt.figure()
if poses_original is not None:
  plt.plot(poses_original[:, 0], poses_original[:, 1], '-', label="Original",
            alpha=1, color="green")


if poses_optimized is not None:
  plt.plot(poses_optimized[:, 0], poses_optimized[:, 1], '-', label="Optimized",
            alpha=1, color="blue")

plt.axis('equal')
plt.legend()
# Show the plot and wait for the user to close.
plt.show()