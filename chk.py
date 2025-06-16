import cv2
import numpy as np

# Load the original and blurred images
original = cv2.imread('input2.jpg')
blurred = cv2.imread('blurred_serial2.jpg')

# Resize both images to the same dimensions
width, height = 800, 600  # Adjust as needed
original = cv2.resize(original, (width, height))
blurred = cv2.resize(blurred, (width, height))

# Concatenate side by side
comparison = np.hstack((original, blurred))

# Make window resizable
cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
cv2.imshow("Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
