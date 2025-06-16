import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    sparse_img = csr_matrix(gray)
    sparse_mask = csr_matrix(mask)
    return sparse_img.multiply(sparse_mask / 255)  # apply mask

def extract_features(sparse_img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    sparse_kernel = csr_matrix(kernel)
    image = sparse_img.toarray()
    h, w = image.shape
    output = np.zeros_like(image)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            region = image[i-1:i+2, j-1:j+2]
            output[i, j] = np.sum(region * sparse_kernel.toarray())
    
    return csr_matrix(output.flatten())

# Step 1: Load reference image (e.g. human eye)
reference_sparse = preprocess_image("reference.jpg")
reference_features = extract_features(reference_sparse)

# Step 2: Load input image
input_sparse = preprocess_image("input.jpg")
input_features = extract_features(input_sparse)

# Step 3: Compare using Cosine Similarity
cos_sim = reference_features.dot(input_features.T).toarray()[0][0] / (
    norm(reference_features) * norm(input_features)
)

print(f"\n🔍 Cosine Similarity: {cos_sim:.4f}")
if cos_sim > 0.85:
    print("✅ The input image matches the reference image.")
else:
    print("❌ The input image does NOT match the reference image.")

# Optional: show images
ref_show = reference_sparse.toarray().astype(np.uint8)
inp_show = input_sparse.toarray().astype(np.uint8)
cv2.imshow("Reference", ref_show)
cv2.imshow("Input", inp_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
