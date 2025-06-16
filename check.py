import cv2
import numpy as np

original = cv2.imread('input3.jpg')
blurred = cv2.imread('blurred_serial7.jpg')

combined = np.hstack((original, blurred))  # Combine images side by side
cv2.imshow('Original vs Blurred', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
edges_original = cv2.Canny(original, 100, 100)
edges_blurred = cv2.Canny(blurred, 100, 100)

cv2.imshow('Original Edges', edges_original)
cv2.imshow('Blurred Edges', edges_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
