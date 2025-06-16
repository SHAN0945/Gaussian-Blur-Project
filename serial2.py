import cv2
import time

# Load the input image
image = cv2.imread('input3.jpg')

# Check if image is loaded
if image is None:
    print("Error: Image not found!")
else:
    # Start measuring time
    start_time = time.time()

    # Split image into color channels
    b, g, r = cv2.split(image)

    # Apply Gaussian blur to each channel multiple times
    for _ in range(3):  # 3 passes to make it slower
        b = cv2.GaussianBlur(b, (49,49),0)
        g = cv2.GaussianBlur(g, (49,49),0)
        r = cv2.GaussianBlur(r,(49,49),0)

    # Merge back the blurred channels
    blurred = cv2.merge([b, g, r])

    # Stop measuring time
    end_time = time.time()

    # Save the output image
    cv2.imwrite('blurred_serial_slow.jpg', blurred)

    # Print execution time
    print(f" Serial Execution Time: {end_time - start_time:.4f} seconds")
