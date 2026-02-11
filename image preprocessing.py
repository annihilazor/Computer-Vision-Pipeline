import cv2

img = cv2.imread('test.jpg')
resizing = cv2.resize(img, (600, 480))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(grey, (5,5  ), 0)
edge = cv2.Canny(img, 102, 20)

cv2.imshow('Resized Image', resizing)
cv2.imshow('Grey Image', grey)  
cv2.imshow('Blurring Image', blurring)
cv2.imshow('Edge Image', edge)  
cv2.waitKey(0)
cv2.destroyAllWindows()
 