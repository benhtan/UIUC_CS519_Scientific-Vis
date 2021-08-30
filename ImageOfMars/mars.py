import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


mars_gray = mpimg.imread('mars.png') # Shape (501, 1024, 3)
lum_img = mars_gray[:, :, 0]
# print(mars_gray.shape)
# print(lum_img.shape)

mars_colormap_dict =    {   'red':     [ [0.0, 0.0, 0.0], [0.2, 0.0, 0.14], [0.75, 1.0, 1.0], [1.0, 0.7, 0.7] ],
                            'green':   [ [0.0, 0.0, 0.14], [0.24, 0.14, 0.24], [0.3, 0.7, 0.7], [0.4, 0.3, 0.0], [1.0, 0.0, 0.0] ],
                            'blue':    [ [0.0, 1.0, 1.0], [0.28, 0.0, 0.0], [1.0, 0.0, 0.0] ]
                        }
mars_colormap = LinearSegmentedColormap('mars_colormap', segmentdata=mars_colormap_dict, N=256)

mars_red = mars_colormap(lum_img)

plt.imsave('mars_red.png', mars_red)
# plt.matshow(mars_gray)
# plt.matshow(mars_red)