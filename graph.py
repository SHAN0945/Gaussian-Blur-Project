import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


image = cv2.imread("input3.jpg")  
kernel_size = 35
start_time = time.time()
mean_blurred = cv2.blur(image, (kernel_size, kernel_size))
mean_time = time.time() - start_time


start_time = time.time()
median_blurred = cv2.medianBlur(image, kernel_size)
median_time = time.time() - start_time


start_time = time.time()
gaussian_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
gaussian_time = time.time() - start_time


blur_types = ["Mean Blur", "Median Blur", "Gaussian Blur"]
execution_times = [mean_time, median_time, gaussian_time]


plt.figure(figsize=(8, 5))
plt.bar(blur_types, execution_times, color=["red", "blue", "green"])
plt.xlabel("Blurring Techniques")
plt.ylabel("Execution Time (seconds)")
plt.title("Comparison of Execution Time for Blurring Techniques")
plt.show()


print(f"Mean Blur Time: {mean_time:.4f} sec")
print(f"Median Blur Time: {median_time:.4f} sec")
print(f"Gaussian Blur Time: {gaussian_time:.4f} sec")