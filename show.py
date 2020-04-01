import cv2  
path = r"home.png"
image = cv2.imread(path, 0) 
window_name = 'image'
while True:
    cv2.imshow(window_name, image) 
    if cv2.waitKey(0) & 0xFF == 27:
        exit()
