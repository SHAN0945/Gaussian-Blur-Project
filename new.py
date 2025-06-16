import cv2
import time
import numpy as np

image = cv2.imread('input3.jpg')

if image is None:
    print("Error: Image not found!")
    exit()

def apply_blur_and_measure_time(blur_function, image, params):
    start_time = time.time()
    blurred_image = blur_function(image, *params)
    end_time = time.time()
    execution_time = end_time - start_time
    return blurred_image, execution_time


gaussian_blur, gaussian_time = apply_blur_and_measure_time(cv2.GaussianBlur, image, [(35, 35), 0])

mean_blur, mean_time = apply_blur_and_measure_time(cv2.blur, image, [(35, 35)])

median_blur, median_time = apply_blur_and_measure_time(cv2.medianBlur, image, [35])

cv2.imwrite("gaussian_blur.jpg", gaussian_blur)
cv2.imwrite("mean_blur.jpg", mean_blur)
cv2.imwrite("median_blur.jpg", median_blur)

print(f"Gaussian Blur Time: {gaussian_time:.4f} sec")
print(f"Mean Blur Time: {mean_time:.4f} sec")
print(f"Median Blur Time: {median_time:.4f} sec")

cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Mean Blur", mean_blur)
cv2.imshow("Median Blur", median_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()