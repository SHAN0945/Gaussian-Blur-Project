import cv2
import numpy as np
import multiprocessing
import time
import os

def apply_gaussian_blur(image_section, kernel_size):
    return cv2.GaussianBlur(image_section, (kernel_size, kernel_size), 0)

def split_image(image, num_processes):
    height = image.shape[0]
    split_height = height // num_processes
    sections = []

    for i in range(num_processes):
        start_row = i * split_height
        end_row = height if i == num_processes - 1 else (i + 1) * split_height
        sections.append(image[start_row:end_row])

    return sections

def merge_sections(sections):
    return np.vstack(sections)

def parallel_gaussian_blur(image, kernel_size=49, num_processes=None):
    if num_processes is None:
        num_processes = min(4, os.cpu_count())  # Limit to 4 or CPU max

    sections = split_image(image, num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        blurred_sections = pool.starmap(apply_gaussian_blur, [(sec, kernel_size) for sec in sections])

    return merge_sections(blurred_sections)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    image = cv2.imread("input3.jpg")
    if image is None:
        print("Image not found.")
    else:
        start_time = time.time()
        blurred_image = parallel_gaussian_blur(image, kernel_size=49)
        end_time = time.time()

        print(f"Parallel Execution Time (optimized): {end_time - start_time:.4f} seconds")
        cv2.imwrite("blurred_parallel_optimized.png", blurred_image)
