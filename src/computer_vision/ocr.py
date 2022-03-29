import cv2
import pytesseract as ptz
import numpy as np
ptz.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#
image = cv2.imread(r'data\raw\others\pytz_nic.jpg')

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OCR
response = ptz.image_to_string(img_rgb)
print(response)

# Boxes
response = ptz.image_to_boxes(image)
print(response)

# Data
"""
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
       
OCR Engine modes: (see https://github.com/tesseract-ocr/tesseract/wiki#linux)
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
"""
image = cv2.imread(r"data\raw\others\pytz_receipt.jpg")
config = r"--oem 3 --psm 6"

image_drawn = image.copy()

# PreProcessing
image_drawn = cv2.cvtColor(image_drawn, cv2.COLOR_BGR2GRAY)
kernel_size = 3
image_drawn = cv2.GaussianBlur(image_drawn, (kernel_size, kernel_size), 0)
thresh_it = cv2.adaptiveThreshold(image_drawn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

BLACK = 255
# thresh_it = cv2.copyMakeBorder(thresh_it, 1,1,1,1, cv2.BORDER_CONSTANT, value=BLACK)

# cv2.imshow("image_drawn", image_drawn)
cv2.imshow("imag2e_drawn", thresh_it)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(thresh_it, 0, 100)
output = cv2.bitwise_and(image, image, mask=mask)

ret, thresh = cv2.threshold(mask, 40, 255, 0)
contours, hierarchy = cv2.findContours(thresh_it, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]


# Alternate
# thresh_it = cv2.adaptiveThreshold(image_drawn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# dilated = cv2.dilate(thresh_it, kernel)
# contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(output, contours, -1, (0,0,255), 1)

    # find the biggest countour (c) by the area
    c = max(contours, key = lambda x: cv2.arcLength(x, True))

    x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)

cropped_receipt = thresh.copy()[y:y+h, x:x+w]

cv2.imwrite("bin.jpg", cropped_receipt)
# show the images
cv2.imshow("Result", output)
cv2.imshow("cropped_receipt", cropped_receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()


response = ptz.image_to_data(cropped_receipt, output_type=ptz.Output.DICT, config=config)
print(response)
n_boxes = len(response['level'])

for i in range(n_boxes):
    _ = response['text'][i]
    if(_ != "") and (0 <= str(_).__len__() <= 40) and (len(set(_)) != 1):
        print(_)
        (x, y, w, h) = (response['left'][i], response['top'][i], response['width'][i], response['height'][i])
        cv2.rectangle(cropped_receipt, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("image", image)
cv2.imshow("image_drawn", image_drawn)
cv2.imshow("thresh", thresh)
cv2.imshow("cropped_receipt", cropped_receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()

