# Overaching idea

Simulate a graph of ADC against tm (mixing time) for given input variables
Add noise to the points in the graph
Using the points, see if you can reconstruct the graph using non-linear least squares fitting.
This is done for 1 voxel at a time. The whole process is then done for each voxel, but this would be in how the function is called, not within the function itself.

# axr_sim

Simulate a noise free sum of magnetisations.
