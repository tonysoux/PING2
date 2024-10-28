import cv2
import numpy as np
import json

# Initialisation du soustracteur de fond pour détecter le mouvement
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Déclaration de variables globales
roi1 = None
ball_color = None
ROI1_mask = None

# Redimensionne la frame pour l'affichage
def resize_for_display(frame, max_display_size=800):
    height, width = frame.shape[:2]
    if height > width:
        new_height = min(height, max_display_size)
        new_width = int(width * new_height / height)
    else:
        new_width = min(width, max_display_size)
        new_height = int(height * new_width / width)
    return cv2.resize(frame, (new_width, new_height))

# Sélection de la ROI
def select_roi(frame, max_display_size=800):
    global roi1, ROI1_mask
    resized = resize_for_display(frame, max_display_size)
    resize_factor = frame.shape[1] / resized.shape[1]
    roi1 = cv2.selectROI("Sélectionner la ROI", resized)
    roi1 = tuple(int(x * resize_factor) for x in roi1)
    with open("roi.json", "w") as file:
        json.dump(roi1, file)
    x, y, w, h = roi1
    ROI1_mask = np.zeros_like(frame)
    ROI1_mask[y:y+h, x:x+w] = 1

# Sélection de la couleur de la balle
def select_color(frame):
    global ball_color
    ball_roi = cv2.selectROI("Sélectionner la couleur de la balle", frame)
    average_color = cv2.mean(frame[ball_roi[1]:ball_roi[1]+ball_roi[3], ball_roi[0]:ball_roi[0]+ball_roi[2]])
    ball_color = np.array([[average_color[:3]]], dtype=np.uint8)
    with open("color.json", "w") as file:
        json.dump(ball_color.tolist(), file)

# Application de la ROI
def apply_roi(frame):
    if roi1 is not None:
        x, y, w, h = roi1
        return frame[y:y+h, x:x+w]
    return frame

# Masque de couleur pour isoler la balle
def apply_color_mask(frame, tolerance=6):
    global ball_color
    lower = np.array([ball_color[0][0][0] - tolerance, ball_color[0][0][1] - tolerance, ball_color[0][0][2] - tolerance])
    upper = np.array([ball_color[0][0][0] + tolerance, ball_color[0][0][1] + tolerance, ball_color[0][0][2] + tolerance])
    return cv2.inRange(frame, lower, upper)

# Soustraction de fond
def background_subtraction(frame, remove_shadow=True):
    mask = bg_subtractor.apply(frame)
    if remove_shadow:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

# Fonction principale pour détecter la balle noire
def detect_black_ball(frame, roi):
    # Soustraction de fond
    motion_mask = background_subtraction(roi)

    # Masque de couleur
    color_mask = apply_color_mask(roi) if ball_color is not None else np.zeros_like(motion_mask)

    # Combinaison des masques
    combined_mask = cv2.bitwise_and(motion_mask, color_mask)
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Détection de cercles (forme circulaire)
    circles = cv2.HoughCircles(combined_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=100)
    if circles is not None:
        x, y, radius = circles[0][0].astype(int)
        return x, y, radius
    return None

# Chargement de la vidéo
video_path = 'video/video_balle_noire.mp4'
cap = cv2.VideoCapture(video_path)

# Chargement de la ROI et couleur si disponibles
try:
    with open("roi.json", "r") as file:
        roi1 = tuple(json.load(file))
except FileNotFoundError:
    print("Sélection de la ROI nécessaire.")

try:
    with open("color.json", "r") as file:
        ball_color = np.array(json.load(file))
except FileNotFoundError:
    print("Sélection de la couleur de la balle nécessaire.")

# Boucle principale
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Sélection de la ROI au premier passage
    if roi1 is None:
        select_roi(frame)
    roi_frame = apply_roi(frame)

    # Sélection de la couleur de la balle
    if ball_color is None:
        select_color(frame)
    # Détection de la balle noire
    ball_position = detect_black_ball(frame, roi_frame)
    if ball_position:
        x, y, radius = ball_position
        cv2.circle(frame, (x + roi1[0], y + roi1[1]), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x + roi1[0], y + roi1[1]), 2, (0, 0, 255), 3)
        print(f"Balle détectée aux coordonnées : ({x + roi1[0]}, {y + roi1[1]}), Rayon : {radius}")

    # Affichage des masques et de la détection
    cv2.imshow("Flux vidéo - Balle détectée", resize_for_display(frame))
    cv2.imshow("Masque de Mouvement", resize_for_display(background_subtraction(roi_frame)))
    cv2.imshow("Masque de Couleur", resize_for_display(apply_color_mask(roi_frame)))

    # Contrôles
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
