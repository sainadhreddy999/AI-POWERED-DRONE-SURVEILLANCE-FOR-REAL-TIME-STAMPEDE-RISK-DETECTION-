import cv2, urllib.request, numpy as np
import sys
import app

print('Downloading test image...')
urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/4/41/Crowd_in_a_train_station.jpg', 'test_crowd.jpg')

img = cv2.imread('test_crowd.jpg')
if img is None:
    print('Failed to read image')
    sys.exit(1)

h, w = img.shape[:2]
print(f'Image size: {w}x{h}')

processed, status, count = app.process_media_content(img, w, h, 0, False, 'Normal')
print(f'Status: {status}')
print(f'Count: {count}')

cv2.imwrite('test_crowd_processed.jpg', processed)
print('Saved processed image.')
