import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as p
import skimage.io as io
from skimage.exposure import equalize_hist
from sklearn import datasets

x = datasets.load_digits()
print (x.images[9])
print (x.images[9].reshape(-1, 64))
print (x.target[9])

def show_corners(corners, img):
	fig = p.figure()
	p.gray()
	p.imshow(img)
	y_corner, x_corner = zip(*corners)
	p.plot(x_corner, y_corner, 'or')
	p.xlim(0, img.shape[1])
	p.ylim(img.shape[0], 0)
	fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
	p.show()
	
img = io.imread('C:/Users/TCS_1549117/Desktop/Environment/machine_learning_examples/licence.png')
img = equalize_hist(rgb2gray(img))
corners = corner_peaks(corner_harris(img), min_distance=2)
show_corners(corners, img)

"""
import mahotas as mh
from mahotas.features import surf

image = mh.imread('C:/Users/TCS_1549117/Desktop/Environment/machine_learning_examples/licence.png', as_grey=True)

print ('The first SURF descriptor:\n', surf.surf(image)[0])
print ('Extracted %s SURF descriptors' % len(surf.surf(image)))
"""