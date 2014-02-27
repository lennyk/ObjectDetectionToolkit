__author__ = 'lenny'

# from tutorial: http://youtu.be/G55tr8GcaLQ

from PIL import Image
from pylab import *
import numpy
from scipy.ndimage import filters
import requests
import cv2

r = requests.get('http://i.imgur.com/Cgly4.jpg')
with open('imported_pic.jpg', 'wb') as f:
    f.write(r.content)

# read the image
im = cv2.imread('imported_pic.jpg')

# pyrdown
im_pyrdown = cv2.pyrDown(im)

# convert color image to grayscale
gray = cv2.cvtColor(im_pyrdown, cv2.COLOR_RGB2GRAY)

# detect feature points
s = cv2.SURF()
mask = numpy.uint8(ones(gray.shape))
keypoints = s.detect(gray, mask)

# image visualization
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for k in keypoints[::10]:
    cv2.circle(vis,(int(k.pt[0]), int(k.pt[1])), 2, (0, 255, 0), 1)
    cv2.circle(vis,(int(k.pt[0]), int(k.pt[1])), int(k.size), (0, 255, 0), 2)

cv2.imshow('SURF Image', vis)
cv2.waitKey()
cv2.destroyAllWindows()