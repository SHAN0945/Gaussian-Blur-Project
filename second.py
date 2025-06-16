import cv2
import numpy as np
import multiprocessing

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
        if i == num_processes - 1:  # Last section takes remaining rows
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
    multiprocessing.freeze_support()  # Required for Windows
    
    # Load Image
    image_path = "image.png"  # Change to your image path
    image = cv2.imread("input3.jpg")

    if image is None:
        print("Error: Image not found!")
    else:
        # Apply Parallel Gaussian Blur
        blurred_image = parallel_gaussian_blur(image)
        cv2.imwrite("blurred_parallel2.png", blurred_image)


        # Show Results
        cv2.imshow("Original", image)
        cv2.imshow("Parallel Blurred", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
