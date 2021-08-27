import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import matplotlib.image as mpimg
import copy

mario = mpimg.imread("mario_big.png") # Shape is (320, 240, 4)
# Red pixel [1.         0.03529412 0.03921569 1.        ]
luigi = copy.deepcopy(mario)

for row_index, row in enumerate(luigi):
    # print(row[0])
    # break
    for col_index, pixel in enumerate(row):
        if pixel[0] == 1.0 and pixel[1] <= 0.1 and pixel[2] <= 0.1 and pixel[3] == 1.0:
            luigi[row_index][col_index] = [0.0, 1.0, 0.0, 1.0]

plt.imsave('luigi.png', luigi)
# print(luigi.shape)
# print(mario[0])