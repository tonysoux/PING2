#open C:\Users\robin\Desktop\PING2\video\video_balle_noire.mp4

import cv2
import numpy as np

video = cv2.VideoCapture("C:/Users/robin/Desktop/PING2/video/video_balle_noire.mp4")

def resize_for_display(frame, max_display_size=800):
    height, width = frame.shape[:2]
    if height > width:
        new_height = min(height, max_display_size)
        new_width = int(width * new_height / height)
    else:
        new_width = min(width, max_display_size)
        new_height = int(height * new_width / width)
    return cv2.resize(frame, (new_width, new_height))

# selection d'une région d'intérêt avec des points, on enregistre dans un fichier json
# on peut aussi charger une région d'intérêt depuis un fichier json qu'on appellera "roi.json"
roi1 = None
ROI1_mask = None

def select_roi(frame, max_display_size=800):
    global roi1
    resized=resize_for_display(frame, max_display_size)
    resize_factor = frame.shape[1] / resized.shape[1]
    roi1 = cv2.selectROI("video", resized)
    roi1 = tuple(int(x * resize_factor) for x in roi1)
    with open("roi.json", "w") as file:
        json.dump(roi1, file)
    #creation d'un masque pour la roi
    global ROI1_mask
    ROI1_mask = np.zeros_like(frame)
    x, y, w, h = roi1
    ROI1_mask[y:y+h, x:x+w] = 1
    

#on regarde si le fichier roi.json existe
try:
    import json
    with open("roi.json", "r") as file:
        roi1 = tuple(json.load(file))
except FileNotFoundError:
    print("Le fichier roi.json n'existe pas")


#selectionner une couleur dans l'image
ball_color = None
def select_color(frame):
    global ball_color
    ball_roi = cv2.selectROI("video", frame)
    #compute the average color in the selected region
    average_color = cv2.mean(frame[ball_roi[1]:ball_roi[1]+ball_roi[3], ball_roi[0]:ball_roi[0]+ball_roi[2]])
    ball_color = np.array([[average_color[:3]]], dtype=np.uint8) # color format is BGR
    # print(ball_color)
    #on enregistre la couleur dans un fichier json
    with open("color.json", "w") as file:
        json.dump(ball_color.tolist(), file)
        
#on regarde si le fichier color.json existe
try:
    with open("color.json", "r") as file:
        ball_color = np.array(json.load(file))
except FileNotFoundError:
    print("Le fichier color.json n'existe pas")
    

def apply_roi1(frame):
    if roi1 is not None:
        x, y, w, h = roi1
        frame = frame[y:y+h, x:x+w]
    return frame

def apply_color_mask(frame, tolerance=10):
    global ball_color
    lower = np.array([ball_color[0][0][0] - tolerance, ball_color[0][0][1] - tolerance, ball_color[0][0][2] - tolerance])
    upper = np.array([ball_color[0][0][0] + tolerance, ball_color[0][0][1] + tolerance, ball_color[0][0][2] + tolerance])
    mask = cv2.inRange(frame, lower, upper)
    return mask


#background subtraction with shadow detection
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

def background_subtraction(frame, remove_shadow=True):
    mask = subtractor.apply(frame)
    if remove_shadow:
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    return mask

while True:
    ret, frame = video.read()
    roi1_applied = apply_roi1(frame)
    moving_mask = background_subtraction(frame)
    #get contours and draw moving object as rectangles
    contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    p
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if ball_color is not None:
        color_mask = apply_color_mask(frame)
        cv2.imshow("color mask", resize_for_display(color_mask))   
        
    # partage des eaux
    # Appliquer la segmentation par partage des eaux
    # markers = cv2.watershed()

    # Afficher les résultats
    cv2.imshow('Original with Contours', resize_for_display(frame))

    cv2.imshow("moving mask", resize_for_display(moving_mask))
    cv2.imshow("video", resize_for_display(frame))
    key = cv2.waitKey(1)
    if roi1 is None:
        select_roi(frame)
    if ball_color is None and key & 0xFF == ord('c'):
        select_color(frame)
    if key & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
