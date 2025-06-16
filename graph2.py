import cv2
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt

# Load Image
image = cv2.imread("input3.jpg")  # Change to your image path
kernel_size = 49
num_processes = 4

# ---------------- SERIAL GAUSSIAN BLUR ----------------
start_time = time.time()
serial_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
serial_time = time.time() - start_time

# ---------------- PARALLEL GAUSSIAN BLUR ----------------
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

def parallel_gaussian_blur(image, kernel_size=35, num_processes=4):
    sections = split_image(image, num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        blurred_sections = pool.starmap(apply_gaussian_blur, [(sec, kernel_size) for sec in sections])
    return merge_sections(blurred_sections)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Measure Parallel Execution Time
    start_time = time.time()
    parallel_blurred = parallel_gaussian_blur(image, kernel_size, num_processes)
    parallel_time = time.time() - start_time

    # ---------------- PLOTTING RESULTS ----------------
    blur_types = ["Serial Gaussian Blur", "Parallel Gaussian Blur"]
    execution_times = [serial_time, parallel_time]

    plt.figure(figsize=(8, 5))
    plt.bar(blur_types, execution_times, color=["blue", "orange"])
    plt.xlabel("Blurring Technique")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Serial vs Parallel Gaussian Blur Execution Time")
    plt.show()

    # Print Execution Times
    print(f"Serial Gaussian Blur Time: {serial_time:.4f} sec")
    print(f"Parallel Gaussian Blur Time: {parallel_time:.4f} sec")
