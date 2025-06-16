import cv2
import time
image = cv2.imread('input3.jpg')  
start_time = time.time()
blurred = cv2.GaussianBlur(image, (35, 35), 0)
end_time = time.time()
cv2.imwrite('blurred_serial7.jpg', blurred)
print(f"Serial Execution Time: {end_time - start_time:.4f} seconds")