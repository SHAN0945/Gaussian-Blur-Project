import cv2
import numpy as np
import multiprocessing
import time  

def apply_gaussian_blur(image_section, kernel_size):
    """Applies Gaussian blur to a section of the image."""
    return cv2.GaussianBlur(image_section, (kernel_size, kernel_size), 0)

def split_image(image, num_processes):
    """Splits the image into sections for parallel processing."""
    height = image.shape[0]
    split_height = height // num_processes
    sections = []

    for i in range(num_processes):
        start_row = i * split_height
        if i == num_processes - 1:  
            end_row = height
        else:
            end_row = (i + 1) * split_height
        sections.append(image[start_row:end_row])

    return sections

def merge_sections(sections):
    """Merges processed sections back into a single image."""
    return np.vstack(sections)

def parallel_gaussian_blur(image, kernel_size=35, num_processes=4):
    """Applies Gaussian blur in parallel using multiprocessing."""
    sections = split_image(image, num_processes)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        blurred_sections = pool.starmap(apply_gaussian_blur, [(sec, kernel_size) for sec in sections])
    
    return merge_sections(blurred_sections)

if __name__ == "__main__":
    multiprocessing.freeze_support()  

    
    image_path = "image.png"  
    image = cv2.imread("input3.jpg")

    if image is None:
        print("Error: Image not found!")
    else:
        
        start_time = time.time()  

        blurred_image = parallel_gaussian_blur(image)  

        end_time = time.time()  

        
        parallel_execution_time = end_time - start_time
        print(f"Parallel Execution Time: {parallel_execution_time:.4f} seconds")

        
        cv2.imwrite("blurred_parallel7.png", blurred_image)

        cv2.imshow("Original", image)
        cv2.imshow("Parallel Blurred", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        