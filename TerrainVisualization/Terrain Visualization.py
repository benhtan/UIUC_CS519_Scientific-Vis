#!/usr/bin/env python
# coding: utf-8

# # Terrain Visualization

# In this activity, you will work with creating and manipulating 3D surface meshes using **PyVista**, a Python interface for the **Visualization Toolkit (VTK)**. VTK is a powerful open-source library for computer graphics, visualization, and image processing. You can learn more about both tools through these references:
# - https://docs.pyvista.org/
# - https://vtk.org/
# 
# We will also be using the **itkwidgets** library, which provides interactive Jupyter widgets for plotting, to visualize our meshes.
# 
# The outline of this activity will be:
# 1. Creating a 3D surface mesh
# 2. Writing code to coarsen the mesh
# 3. Writing code to visualize the error in elevation between the original mesh and the coarse mesh

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pyvista import set_plot_theme
set_plot_theme('document')


# # Part 1: Creating a 3D Surface Mesh
# We will start by using a topographic surface to create a 3D terrain-following mesh.
# 
# Terrain following meshes are common in the environmental sciences, for instance
# in hydrological modelling (see
# [Maxwell 2013](https://www.sciencedirect.com/science/article/abs/pii/S0309170812002564)
# and
# [ParFlow](https://parflow.org)).
# 
# Below, we domonstrate a simple way to make a 3D grid/mesh that
# follows a given topographic surface. In this example, it is important to note
# that the given digital elevation model (DEM) is structured (gridded and not
# triangulated): this is common for DEMs.
# 

# In[2]:


import pyvista as pv
import math
import numpy as np
import pylab as plt
from pyvista import examples


# Download a gridded topography surface (DEM) using one of the examples provided by PyVista.
# 
# 

# In[3]:


dem = examples.download_crater_topo()
dem


# Now let's subsample and extract an area of interest to make this example
# simple (also the DEM we just loaded is pretty big).
# Since the DEM we loaded is a `pyvista.UniformGrid` mesh, we can use
# the `pyvista.UniformGridFilters.extract_subset` filter to extract a 257x257-point (256x256-cell) area from the DEM:
# 
# 

# In[4]:


subset = dem.extract_subset((572, 828, 472, 728, 0, 0), (1,1,1))
subset
# print(subset)


# Let's plot the area we just extracted to see what it looks like.

# In[5]:


pv.plot_itk(subset)


# Now that we have a region of interest for our terrain following mesh, lets
# make a 3D surface of that DEM.

# In[6]:


terrain = subset.warp_by_scalar() #Warp into a 3D surface mesh (without volume)
terrain
# print(terrain.y.shape)
# print(terrain.dimensions)
# print(terrain.z.shape)
# print(terrain.z)


# We can see that our terrain is now a `pyvista.StructuredGrid` mesh. Now let's plot our terrain.

# In[7]:


pv.plot_itk(terrain)


# ## Part 2: Coarsening the Mesh (and Writing Code)
# In this section, you will write code to generate a new coarse mesh from our `terrain` mesh. Coarse meshes generally provide less accurate solutions, but are computationally faster. 
# 
# Your new mesh should be a `StructuredGrid`, just like the original mesh, but with a lower resolution. This means you will need to redefine the (x, y, z) coordinate points of your mesh. We will explain how to redefine your coordinates a little later on.
# 
# First, let's start with understanding how to generate a new mesh. You can initialize a new `StructuredGrid` object directly from the three point arrays that each contain the x, y, and z coordinates of all points in the mesh, respectively. Note: Each array is a 3D array with dimensions M x N x 1 (with the z-axis always being of length 1).
# 
# You will find the following reference helpful: https://docs.pyvista.org/core/point-grids.html#pyvista.StructuredGrid.
# 
# Let's look at the example below for initializing a new `StructuredGrid`.

# In[8]:


xrng = np.arange(-10, 10, 2)                # [-10,  -8,  -6,  -4,  -2,   0,   2,   4,   6,   8]
yrng = np.arange(-10, 10, 2)
zrng = np.arange(-10, 10, 2)
x_example, y_example, z_example = np.meshgrid(xrng, yrng, zrng)
grid_example = pv.StructuredGrid(x_example, y_example, z_example)
grid_example
# print(x_example)


# Now, let's follow the same general steps as in the above example to generate our new coarse mesh from our previously created `terrain` mesh.
# 
# We can coarsen the mesh by merging every `2f` quads/cells into one and dropping the center point, where `f` is your sampling factor aka the factor by which you want to reduce the resolution. In other words, we can produce a reduced version of the mesh by sampling one out of every `f` points along each axis of the mesh.
# 
# Write code to coarsen `terrain` by a **factor of 2**. In other words, we will be converting the mesh from a 257x257-point mesh to a 129x129-point mesh (or equivalently, a 256x256-cell mesh to a 128x128-cell mesh). 
# 
# In the code block below, define three new point arrays, `xnew`, `ynew`, and `znew` and compose them into a new `StructuredGrid` object named `coarse`.

# In[9]:


#NOTE: You do not need to round any values within your results.
# YOUR CODE HERE
# raise NotImplementedError()
# rng = np.arange(0,terrain.dimensions[0],2)
xnew = terrain.x[::2, ::2]
ynew = terrain.y[::2, ::2]
znew = terrain.z[::2, ::2]
coarse = pv.StructuredGrid(xnew, ynew, znew)
print(f"Coarsening from {terrain.dimensions[0]} to {math.ceil(terrain.dimensions[0]/2)}...")
coarse


# In[10]:


### Tests for coarsenMesh. 
### Please DO NOT hard-code the answers as we will also be using hidden test cases when grading your submission.
assert xnew.shape == (129,129,1)
assert ynew.shape == (129,129,1)
np.testing.assert_allclose(xnew[0][0][0],1818580, rtol=1e-7)
np.testing.assert_allclose(xnew[5][120][0],1818730, rtol=1e-7)
np.testing.assert_allclose(ynew[128][120][0],5650680, rtol=1e-7)
np.testing.assert_allclose(znew[12][12][0],1880.53, rtol=1e-5)


# We can plot the z-values of our new coarsened mesh by adding an additional attribute `values` to our mesh, which will contain a normalized, column-major flattened representation of the z-axis values of our grid.
# 
# See the following reference for more information on array flattening: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html

# In[11]:


#Plot the z-values using the viridis (default) color map
coarse['values'] = pv.plotting.normalize(coarse.z.flatten("F"))

pv.plot_itk(coarse, scalars='values')


# ## Part 3: Visualizing Error Values
# 
# Now that we have generated our coarse mesh, we can visualize the error in elevation between our coarse mesh and our original mesh. More specifically, we want to compute the error value for each point between the new (bilinearly interpolated) center point elevation and the original. We will then visualize the error as a scalar field on the original mesh.

# Since we will need to bilinearly interpolate the center point elevation (i.e. the z-value) of each point in our coarse mesh in order to match the dimensions of our original mesh, let's define a function to do just that.
# 
# Define the function `bilin()` to bilinearly interpolate the value at coordinates `(x,y)` within a rectilinear grid of points.
# 
# **The parameters of your function are:**
# - `x` = x-coordinate of point whose value we wish to interpolate
# - `y` = y-coordinate of point whose value we wish to interpolate
# - `points` = a list of four triplets of the form `(xc, yc, val)`, where `val` denotes the function value associated with coordinates `(xc, yc)`
# 
# This function should return a bilinearly interpolated value associated with coordinate `(x,y)` w.r.t the rectilinear grid formed by `points`.
# 
# **Hints:**
# - You may assume the four triplets within `points` form a valid rectangle
# - You may assume `x` and `y` fall within the rectangle formed by the `points` parameter
# - You should NOT assume the four triplets within `points` are in any specific order

# In[12]:


#NOTE: You do not need to round any values within your results.
def bilin(x, y, points):
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    # to keep unique x and y coordinate
    x_set = set()
    y_set = set()
    
    # to be able to get value at the coordinate quickly
    points_dict = {}
    
    for point in points:
        # add coordinates to set and dict
        x_set.add(point[0])
        y_set.add(point[1])
        points_dict[ (point[0], point[1]) ] = point[2]
    
    # turn it to list
    x_list = list(x_set)
    y_list = list(y_set)
    
    # sort the list
    x_list.sort()
    y_list.sort()
    
#     print(x_list)
#     print(y_list)
#     print(points_dict)
    
    # calculate t ratio for x
    t_x = ( x - x_list[0] ) / ( x_list[1] - x_list[0] )
#     print(t_x)

    # calculate first bilinear interpolation for x axis
    bilin_bottom = ( 1 - t_x ) * points_dict[ (x_list[0], y_list[0]) ] + ( t_x * points_dict[ (x_list[1], y_list[0]) ])
    bilin_top = ( 1 - t_x ) * points_dict[ (x_list[0], y_list[1]) ] + ( t_x * points_dict[ (x_list[1], y_list[1]) ])
    
#     print(bilin_bottom)
#     print(bilin_top)

    # calculate t ratio for y
    t_y = ( y - y_list[0] ) / ( y_list[1] - y_list[0] )
    
    # return bilinear interpolation for y axis
    return ( 1 - t_y ) * bilin_bottom + t_y * bilin_top


# In[13]:


### Tests for bilin(x, y, points) function. 
### Please DO NOT hard-code the answers as we will also be using hidden test cases when grading your submission.
testing_points = [(1,1,3), (3,1,6), (1,3,7), (3,3,9)]
result = bilin(2,2,testing_points)
np.testing.assert_allclose(result,6.25, rtol=1e-2)
result = bilin(2.5,2.5,testing_points)
np.testing.assert_allclose(result,7.6875, rtol=1e-3)
result = bilin(1.1,1.1,testing_points)
np.testing.assert_allclose(result,3.3475, rtol=1e-3)


# Now, using your `bilin()` function, create a new mesh or `StructuredGrid` object named `intmesh`, reconstructed from `coarse` using bilinear interpolation, with the same dimensions as our original mesh `terrain`. Your new mesh should contain the interpolated z-values for each point in `terrain`.
# 
# As a starting point, we have defined some of the variables that will be helpful to you for creating your new interpolated mesh. Specifically, we will be checking the values in `errz` and `intz`, as defined below:
# - `intz`: a 3D array with the same shape as `terrain.z` that will contain the bilinearly interpolated z-values from the coarsened mesh.<br/>**Note:** `intz` is a 3D M x N x 1 array where the last dimension contains the z-values. You should note this when assigning elements to `intz`. See the following Piazza post for more information: https://piazza.com/class/kd7le70c8f1y4?cid=205.
# - `errz`: a list of scalar values. This should contain the absolute error values between each z-value in the original mesh and each interpolated z-value in the new returned mesh
# 
# Just like how we added the attribute `values` to our coarse mesh in order to plot the z-values of the mesh, you should add an additional attribute `errors` to `intmesh` in order to plot the absolute error values between the z-values in the original mesh and the interpolated z-values in our new returned mesh.

# In[14]:


#NOTE: You do not need to round any values within your results.

errz   = []                    #Create a new empty list for holding absolute error values
intz   = np.zeros_like(terrain.z) #Create a new array for holding bilinearly interpolated values from coarse mesh
# print(intz[0][0][0])

xlen   = coarse.z.shape[0]-1   #Number of cells (points-1) on the x-axis of the coarse mesh
ylen   = coarse.z.shape[1]-1   #Number of cells (points-1) on the y-axis of the coarse mesh
# print((xlen)*2)
# print((ylen)*2)
# print(len(terrain.z))
scale = (terrain.z.shape[0]-1)/(coarse.z.shape[0]-1) #Reduction factor between original and coarse; should equal 2
# print(scale)

# YOUR CODE HERE
# raise NotImplementedError()
# print(terrain.dimensions)
for i in range(terrain.dimensions[0]):
    for j in range(terrain.dimensions[0]):
        # odd index, interpolate
        if j % 2 != 0 or i % 2 != 0:
            # print(f'C: {i} {j}')
            i_min = i - 1
            i_max = i + 1
            j_min = j - 1
            j_max = j + 1
            
            if i_min < 0:
                i_min = 0
            if i_max > terrain.dimensions[0] - 1:
                i_max = terrain.dimensions[0] - 1
            if j_min < 0:
                j_min = 0
            if j_max > terrain.dimensions[0] - 1:
                j_max = terrain.dimensions[0] - 1
                
            points = []
            try:
                points = [
                    (terrain.x[i_min][j_min][0], terrain.y[i_min][j_min][0], terrain.z[i_min][j_min][0]),
                    (terrain.x[i_min][j_max][0], terrain.y[i_min][j_max][0], terrain.z[i_min][j_max][0]),
                    (terrain.x[i_max][j_min][0], terrain.y[i_max][j_min][0], terrain.z[i_max][j_min][0]),
                    (terrain.x[i_max][j_max][0], terrain.y[i_max][j_max][0], terrain.z[i_max][j_max][0])
                    ]
            except:
                print(f'{i} {j}')
            
            intz[i][j][0] = bilin(terrain.x[i][j][0], terrain.y[i][j][0], points)
        
        # even index no need to interpolate
        else:
            # print(f'D: {i} {j}')
            intz[i][j][0] = terrain.z[i][j][0]
            
        # calculate error
        errz.append( abs( terrain.z[i][j][0] - intz[i][j][0] ) )
        
intmesh = pv.StructuredGrid(terrain.x, terrain.y, intz)

intmesh['errors'] = pv.plotting.normalize(errz)


# In[15]:


### Tests for vizError. 
### Please DO NOT hard-code the answers as we will also be using hidden test cases when grading your submission.
np.testing.assert_allclose(intz[130][130][0],2547.8, rtol=1e-4)
np.testing.assert_allclose(intz[247][13][0],2142.71, rtol=1e-5)
np.testing.assert_allclose(errz[89],1.89996337890625, rtol=1e-2)
np.testing.assert_allclose(errz[30678],1.18499755859375, rtol=1e-2)
np.testing.assert_allclose(errz[-10],1.0299072265625, rtol=1e-2)


# In[16]:


intmesh


# Now we can visualize the error values that we computed! We recommend adjusting the color map to better visualize the error values. You can change the color map by clicking the settings icon at the top left of the interface.

# In[17]:


print(f"Visualizing error between resolutions {terrain.dimensions[0]} and {math.ceil(terrain.dimensions[0]/2)}...")

pv.plot_itk(intmesh, scalars='errors')


# For reference, here is a sample of what your final visualization should look like (with the magma colormap applied):
# <img src='error-visualization.png' width=600/>

# In[ ]:




