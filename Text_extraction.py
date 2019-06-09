# import the necessary packages
from PIL import Image
import pytesseract
import cv2
import os

#Import image
image = cv2.imread('image.jpg')

#Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Binary
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

#Blurring
blurred = cv2.medianBlur(thresh , 3)
 
#Writing the resultant image into disk as a temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, thresh)

#Path to access Puytesseract (If only you are using Windows OS)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#Applying OCR
text = pytesseract.image_to_string(Image.open(filename))
print(text)

#Deleting the temporarily created file
os.remove(filename)

 


