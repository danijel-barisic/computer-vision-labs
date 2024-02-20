from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


# Harrisov detektor kutova
# 1. Učitavanje slike
img = np.array(Image.open("fer_logo.jpg"))

if len(img.shape) == 3:
    height, width, channels = img.shape
else:
    height, width = img.shape
    channels = 1  # grayscale img

print("Image Height:", height)
print("Image Width:", width)

# grayscale img operation
if channels == 3:
    img = np.mean(img, axis=-1)
else:
    img = img

min_intensity = np.min(img)
max_intensity = np.max(img)
print("Min Intensity:", min_intensity)
print("Max Intensity:", max_intensity)

top_left_patch = img[:10, :10]
print("Intensities of the top-left patch:\n", top_left_patch)

print("Data type before conversion:", img.dtype)
img = img.astype(float)
print("Data type after conversion:", img.dtype)

# 2. Gaussovo zaglađivanje
sigma_values = [1, 3, 5, 7]

plt.figure(figsize=(12, 4))

for i, sigma in enumerate(sigma_values, 1):
    smoothed_img = gaussian_filter(img, sigma=sigma)

    plt.subplot(1, len(sigma_values), i)
    plt.imshow(smoothed_img, cmap='gray')
    plt.title(f'Sigma = {sigma}')
    plt.axis('off')

plt.show()

# 3. Izračun gradijenata
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gradient_x = convolve(img, sobel_x)
gradient_y = convolve(img, sobel_y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(gradient_x), cmap='gray')
plt.title('Gradient X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(gradient_y), cmap='gray')
plt.title('Gradient Y')
plt.axis('off')

plt.show()

# calculate second moments of gradients
gradient_xx = convolve(gradient_x**2, np.ones((3, 3)))
gradient_yy = convolve(gradient_y**2, np.ones((3, 3)))
gradient_xy = convolve(gradient_x * gradient_y, np.ones((3, 3)))

print("Second moments of gradients:")
print("Mxx:\n", gradient_xx)
print("Myy:\n", gradient_yy)
print("Mxy:\n", gradient_xy)

# 4. Sumiranje gradijenata u lokalnom susjedstvu
# local neighborhood kernel for summing gradients
local_neighborhood_kernel = np.ones((3, 3))

# second moments of gradients in the local neighborhood
G_xx = convolve(gradient_x**2, local_neighborhood_kernel, mode='nearest')
G_yy = convolve(gradient_y**2, local_neighborhood_kernel, mode='nearest')
G_xy = convolve(gradient_x * gradient_y, local_neighborhood_kernel, mode='nearest')

G = np.stack([[G_xx, G_xy], [G_xy, G_yy]], axis=-1)

print("Characteristic matrix G in each pixel:")
print(G)

# 5. Izračun Harrisovog odziva
# Harris response calculation
k = 0.04
det_G = G_xx * G_yy - G_xy**2
trace_G = G_xx + G_yy
harris_response = det_G - k * trace_G**2

plt.imshow(harris_response) 
plt.title("Harris Response")
plt.colorbar()
plt.show()

# 6. Potiskivanje nemaksimalnih odziva
threshold = 1e10
neighborhood_size = 14

# thresholding #TODO uhh, not using summing window size? But still works?
harris_response[harris_response < threshold] = 0

# non-maximum suppression
local_maxima = np.zeros_like(harris_response)
for i in range(neighborhood_size // 2, harris_response.shape[0] - neighborhood_size // 2):
    for j in range(neighborhood_size // 2, harris_response.shape[1] - neighborhood_size // 2):
        local_window = harris_response[i - neighborhood_size // 2: i + neighborhood_size // 2 + 1,
                                        j - neighborhood_size // 2: j + neighborhood_size // 2 + 1]
        if harris_response[i, j] == np.max(local_window):
            local_maxima[i, j] = harris_response[i, j]
        else:
            local_maxima[i, j] = 0
        # if harris_response[i, j] < np.max(local_window):
        #     local_maxima[i, j] = 0

harris_response=local_maxima
plt.imshow(harris_response)
plt.title("Harris Response after Non-Maximal Suppression")
plt.show()

# 7. Selektiranje k-najvećih odziva #TODO not surpressed?
nonzero_coords = np.nonzero(harris_response)

k_largest_indices = np.argpartition(harris_response[nonzero_coords], -int(k))[-int(k):]

k_largest_coords = np.column_stack((nonzero_coords[0][k_largest_indices],
                                    nonzero_coords[1][k_largest_indices]))

plt.imshow(img)
plt.title("Detected Corners")
plt.scatter(k_largest_coords[:, 1], k_largest_coords[:, 0], s=30, facecolors='none', edgecolors='r')
plt.show()



# Cannyev detektor rubova

# 1. Učitavanje slike
img = np.array(Image.open("house.jpg"))

if len(img.shape) == 3:
    height, width, channels = img.shape
else:
    height, width = img.shape
    channels = 1  # grayscale img

print("Image Height:", height)
print("Image Width:", width)

# grayscale img operation
if channels == 3:
    img = np.mean(img, axis=-1)
else:
    img = img

min_intensity = np.min(img)
max_intensity = np.max(img)
print("Min Intensity:", min_intensity)
print("Max Intensity:", max_intensity)

img = img.astype(float)

# 2. Gaussovo zaglađivanje
sigma = 1.5
img = gaussian_filter(img,sigma=sigma)

# 3. Izračun gradijenata
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gradient_x = convolve(img, sobel_x)
gradient_y = convolve(img, sobel_y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(gradient_x), cmap='gray')
plt.title('Gradient X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(gradient_y), cmap='gray')
plt.title('Gradient Y')
plt.axis('off')

plt.show()

# calculate second moments of gradients
gradient_xx = convolve(gradient_x**2, np.ones((3, 3)))
gradient_yy = convolve(gradient_y**2, np.ones((3, 3)))
gradient_xy = convolve(gradient_x * gradient_y, np.ones((3, 3)))

# 4. Izračun magnitude i kuta gradijenta
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_angle = np.arctan2(gradient_y, gradient_x)

# Normalizacija magnituda
normalized_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255

# Vizualizacija normaliziranih magnituda
plt.imshow(normalized_magnitude, cmap='gray')
plt.title("Normalized Gradient Magnitude")
plt.show()

# 5. Potiskivanje nemaksimalnih odziva
suppressed_magnitude = np.copy(normalized_magnitude)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        theta = gradient_angle[i, j]

        # determine the gradient direction
        if (22.5 < np.degrees(theta) <= 67.5) or (-157.5 < np.degrees(theta) <= -112.5):
            neighbor1 = normalized_magnitude[i - 1, j - 1]
            neighbor2 = normalized_magnitude[i + 1, j + 1]
        elif (67.5 < np.degrees(theta) <= 112.5) or (-112.5 < np.degrees(theta) <= -67.5):
            neighbor1 = normalized_magnitude[i - 1, j]
            neighbor2 = normalized_magnitude[i + 1, j]
        elif (112.5 < np.degrees(theta) <= 157.5) or (-67.5 < np.degrees(theta) <= -22.5):
            neighbor1 = normalized_magnitude[i - 1, j + 1]
            neighbor2 = normalized_magnitude[i + 1, j - 1]
        else:
            neighbor1 = normalized_magnitude[i, j - 1]
            neighbor2 = normalized_magnitude[i, j + 1]

        # suppress non-maximal pixels
        if normalized_magnitude[i, j] < max(neighbor1, neighbor2):
            suppressed_magnitude[i, j] = 0

plt.imshow(suppressed_magnitude, cmap='gray')
plt.title("Suppressed Gradient Magnitude")
plt.show()


# 6. Uspoređivanje s dva praga - histereza
min_val = 10
max_val = 90

edges = np.zeros_like(suppressed_magnitude)
strong_edges = (suppressed_magnitude > max_val)
weak_edges = np.logical_and(suppressed_magnitude >= min_val, suppressed_magnitude <= max_val)

neighborhood_size = 14

# connect weak edges to strong edges in the local neighborhood
for i in range(neighborhood_size // 2, height - neighborhood_size // 2):
    for j in range(neighborhood_size // 2, width - neighborhood_size // 2):
        if weak_edges[i, j]:
            local_window = suppressed_magnitude[i - neighborhood_size // 2:i + neighborhood_size // 2 + 1,
                                                j - neighborhood_size // 2:j + neighborhood_size // 2 + 1]
            if np.any(strong_edges[i - neighborhood_size // 2:i + neighborhood_size // 2 + 1,
                                   j - neighborhood_size // 2:j + neighborhood_size // 2 + 1] > 0):
                edges[i, j] = 1

plt.imshow(edges, cmap='gray')
plt.title("Strong Edges")
plt.show()

final_result = np.zeros_like(edges)
final_result[strong_edges] = 1
plt.imshow(final_result, cmap='gray')
plt.title("Final Result")
plt.show()
