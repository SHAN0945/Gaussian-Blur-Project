import cv2
import numpy as np
import multiprocessing
import time

def apply_blur(section, ksize):
    return cv2.GaussianBlur(section, (ksize, ksize), 0)

def split_image(image, parts):
    h = image.shape[0]
    step = h // parts
    return [image[i*step:(i+1)*step if i != parts-1 else h] for i in range(parts)]

def merge(sections):
    return np.vstack(sections)

def parallel_blur(image, ksize=35, parts=4):
    split = split_image(image, parts)
    with multiprocessing.Pool(processes=parts) as pool:
        blurred_parts = pool.starmap(apply_blur, [(sec, ksize) for sec in split])
    return merge(blurred_parts)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Load input image
    image = cv2.imread("image4.jpg")
    if image is None:
        print("❌ Error: 'input3.jpg' not found!")
        exit()

    #Parallel Blur
    start_parallel = time.time()
    blurred_parallel = parallel_blur(image)
    end_parallel = time.time()
    parallel_time = end_parallel - start_parallel
    cv2.imwrite("blurred_parallel.jpg", blurred_parallel)

    #Serial Blur (repeated for delay)
    start_serial = time.time()
    blurred_serial = image.copy()
    for _ in range(10):  # Increase load
        blurred_serial = cv2.GaussianBlur(blurred_serial, (35, 35), 0)
    end_serial = time.time()
    serial_time = end_serial - start_serial
    cv2.imwrite("blurred_serial.jpg", blurred_serial)

    #Display Results
    print(f"Parallel Execution Time: {parallel_time:.4f} seconds")
    print(f"Serial Execution Time:   {serial_time:.4f} seconds")

    cv2.imshow("Parallel Blurred", blurred_parallel)
    cv2.imshow("Serial Blurred", blurred_serial)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
