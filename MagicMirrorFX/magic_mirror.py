import cv2
import numpy as np
import pygame
import os
from datetime import datetime
# Initialize pygame mixer
pygame.mixer.init()
capture_sound = pygame.mixer.Sound('capture_sound.mp3')
# Load Haarcascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load all filters
filters = {
    "crown": cv2.imread('assets/golden crown.png', cv2.IMREAD_UNCHANGED),
    "dog": cv2.imread('assets/DOG.png', cv2.IMREAD_UNCHANGED),
    "hat": cv2.imread('assets/HAT.png', cv2.IMREAD_UNCHANGED),
    "beard": cv2.imread('assets/BEARD.png', cv2.IMREAD_UNCHANGED),
    "birthday": cv2.imread('assets/BRITHDAY.png', cv2.IMREAD_UNCHANGED),
    "cute": cv2.imread('assets/CUTE.png', cv2.IMREAD_UNCHANGED),
    "devil": cv2.imread('assets/DEVIL.png', cv2.IMREAD_UNCHANGED),
    "girl": cv2.imread('assets/GIRL.png', cv2.IMREAD_UNCHANGED),
    "gold": cv2.imread('assets/GOLD.png', cv2.IMREAD_UNCHANGED),
    "music": cv2.imread('assets/MUSIC.png', cv2.IMREAD_UNCHANGED),
    "rabbit": cv2.imread('assets/RABBIT.png', cv2.IMREAD_UNCHANGED)
}
# Preview icons (small 50x50)
preview_icons = {}
for name, img in filters.items():
    if img is not None:
        preview_icons[name] = cv2.resize(img, (50, 50))
# Check for filter loading errors
for name, f in filters.items():
    if f is None:
        print(f"⚠️ Failed to load {name} filter!")
# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Ensure selfies directory exists
os.makedirs('MagicMirrorFX/selfies', exist_ok=True)
# Default active filter
current_filter = None
# Function to overlay transparent images
def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if y + h > bg.shape[0] or x + w > bg.shape[1] or x < 0 or y < 0:
        return bg
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0
    inv_mask = 1 - mask
    bg[y:y + h, x:x + w] = (mask * overlay_rgb + inv_mask * bg[y:y + h, x:x + w]).astype(np.uint8)
    return bg
# 1. Fancy Intro Screen
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame[:] = (30, 30, 30)

    cv2.putText(frame, 'Magic Mirror FX 🎭', (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    filters_menu = [
        "1 - Crown 👑",
        "2 - Dog 🐶",
        "3 - Hat 🎩",
        "4 - Beard 🧔",
        "5 - Birthday 🎉",
        "6 - Cute 💖",
        "7 - Devil 😈",
        "8 - Girl 👧",
        "9 - Gold ✨",
        "M - Music 🎵",
        "R - Rabbit 🐰",
        "0 - No Filter 🚫",
        "Q - Quit ❌"
    ]

    y_start = 100
    for i, text in enumerate(filters_menu):
        cv2.putText(frame, text, (50, y_start + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, 'Press any key to start...', (120, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Magic Mirror FX", frame)
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        break
# 2. Main Webcam Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    # Live filter preview bar
    x_pos = 10
    y_pos = 10
    for name, icon in preview_icons.items():
        if current_filter == name:
            cv2.rectangle(frame, (x_pos - 5, y_pos - 5), (x_pos + 55, y_pos + 55), (0, 255, 255), 3)

        if icon.shape[2] == 4:
            frame = overlay_image(frame, icon, x_pos, y_pos)
        else:
            frame[y_pos:y_pos + 50, x_pos:x_pos + 50] = icon

        x_pos += 60

    # Apply filter on faces
    for (x, y, w, h) in faces:
        if current_filter and current_filter in filters:
            overlay = filters[current_filter]

            # Resize filter
            filter_width = w
            filter_height = int(overlay.shape[0] * (filter_width / overlay.shape[1]))
            resized_filter = cv2.resize(overlay, (filter_width, filter_height))

            # Adjust position based on filter type
            if current_filter in ['crown', 'hat', 'birthday', 'devil']:
                y_offset = y - filter_height + 20  # Above head
                x_offset = x
            elif current_filter in ['beard']:
                y_offset = y + h - int(filter_height / 2)  # Chin
                x_offset = x
            else:
                y_offset = y
                x_offset = x

            frame = overlay_image(frame, resized_filter, x_offset, y_offset)

    cv2.imshow("Magic Mirror FX", frame)

    key = cv2.waitKey(1) & 0xFF

    # Key mappings
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = datetime.now().strftime('MagicMirrorFX/selfies/selfie_%Y%m%d_%H%M%S.png')
        cv2.imwrite(filename, frame)
        capture_sound.play()
    elif key == ord('1'):
        current_filter = "crown"
    elif key == ord('2'):
        current_filter = "dog"
    elif key == ord('3'):
        current_filter = "hat"
    elif key == ord('4'):
        current_filter = "beard"
    elif key == ord('5'):
        current_filter = "birthday"
    elif key == ord('6'):
        current_filter = "cute"
    elif key == ord('7'):
        current_filter = "devil"
    elif key == ord('8'):
        current_filter = "girl"
    elif key == ord('9'):
        current_filter = "gold"
    elif key == ord('m'):
        current_filter = "music"
    elif key == ord('r'):
        current_filter = "rabbit"
    elif key == ord('0'):
        current_filter = None  # No filter
# Cleanup
cap.release()
cv2.destroyAllWindows()
