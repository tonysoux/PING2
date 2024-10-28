import cv2
import numpy as np

# Initialiser le tracker CSRT pour suivre la balle après détection initiale
tracker = cv2.TrackerCSRT_create()
# Initialisation du soustracteur de fond
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Fonction pour détecter les cercles et initier le suivi
def find_ball(frame, tracker_initialized):
    roi = frame[frame.shape[0] // 3:, :]  # Région d'intérêt dans le tiers inférieur
    roi_top = frame.shape[0] // 3  # Position supérieure du ROI

    # Conversion en niveaux de gris
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Application du flou pour réduire le bruit
    blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)


 # Appliquer le soustracteur de fond pour obtenir les zones en mouvement
    fg_mask = bg_subtractor.apply(blurred_frame)

    # Appliquer un seuil pour isoler les zones en mouvement
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    # Transformée de Hough pour détecter les cercles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=60, minRadius=20, maxRadius=100)

    if circles is not None and not tracker_initialized:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, radius) in circles:
            if radius > 20:
                # Filtrer les objets non circulaires
                mask = np.zeros_like(gray_frame)
                cv2.circle(mask, (x, y), radius, 255, -1)

                # Calculer les contours dans la région du cercle détecté
                circle_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(circle_contours) > 0:
                    contour = circle_contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)

                    if perimeter == 0:
                        continue
                    
                    circularity = 4 * np.pi * (area / (perimeter ** 2))
                    
                    if 0.85 < circularity < 1.1:  # Valider la circularité
                        # Initialiser le tracker avec la position du premier cercle valide
                        bbox = (x - radius, y + roi_top - radius, 2 * radius, 2 * radius)
                        return bbox, True

    return None, tracker_initialized

# Charger la vidéo
video_path = 'video/WIN_20241017_11_06_23_Pro.mp4'  # Remplacer par le chemin correct
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible de lire la vidéo.")
    exit()

# Récupérer les propriétés de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Définir le codec et initialiser le VideoWriter pour sauvegarder la vidéo avec les détections
output_path = 'video/video_detection_tracked.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Variables pour la détection et le suivi
tracking_started = False  # Suivi initialisé ou non
bbox = None  # Boîte englobante de la balle

# Boucle de traitement vidéo
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if not tracking_started:
        # Détection initiale de la balle
        bbox, tracking_started = find_ball(frame, tracking_started)

        if tracking_started and bbox:
            # Initialiser le tracker avec la boîte englobante (bbox) de la balle détectée
            tracker.init(frame, bbox)

    else:
        # Mettre à jour la position de l'objet (balle) à l'aide du tracker
        success, bbox = tracker.update(frame)

        if success:
            # Dessiner la boîte englobante autour de la balle suivie
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            print(f'Balle suivie aux coordonnées : {p1}, {p2}')
        else:
            print('Perte du suivi de la balle')
            tracking_started = False  # Réinitialiser la détection si le suivi est perdu

    # Écrire la frame traitée dans la vidéo de sortie
    out.write(frame)

    # Afficher la frame traitée
    cv2.imshow('Flux vidéo - Suivi de la balle', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vidéo enregistrée sous {output_path}")
