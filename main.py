import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough(image, edge_image, num_rhos=180, num_thetas=180):
  edge_height, edge_width = edge_image.shape[:2]
  d = np.sqrt(np.square(edge_height) + np.square(edge_width))

  dtheta = 180 / num_thetas
  drho = (2 * d) / num_rhos
  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)
  accumulator = np.zeros((len(rhos), len(rhos)))

  figure = plt.figure(figsize=(20, 20))
  subplot1 = figure.add_subplot(1, 4, 1)
  subplot1.imshow(image)
  subplot2 = figure.add_subplot(1, 4, 2)
  subplot2.imshow(edge_image, cmap="gray")

  subplot1.title.set_text("Original Image")
  subplot2.title.set_text("Edge Image")
  plt.show()
  return accumulator, rhos, thetas


path = r'C:\Users\Lokesh\PycharmProjects\pythonProject18\input.png'
image = cv2.imread(path)
# print(image.shape) # rows, columns, dimensions(/channels)

# converting our RGB image to grayscale image
edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blurring the image
edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)

# selecting the optimised parameters for the canny edge detection method using ostu's method or you can apply
hgh_th, thresh_im = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_th = 0.5 * hgh_th
print(hgh_th, low_th)
edge_image = cv2.Canny(edge_image, 100 , 200)

# applying Closing operation (a morphological operation )
edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
hough(image, edge_image)
edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

hough(image, edge_image)