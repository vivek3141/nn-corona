import cv2  
  
# path  
path = r"home.png"
  
# Reading an image in grayscale mode 
image = cv2.imread(path, 0) 
  
# Window name in which image is displayed 
window_name = 'image'
  
# Using cv2.imshow() method  
# Displaying the image  
cv2.imshow(window_name, image) 