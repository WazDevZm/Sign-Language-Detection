import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7  # Increased for better accuracy
)

DATA_DIR = './data'

# Loop through each folder in ./data
for dir_name in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_name)
    if not os.path.isdir(dir_path):
        continue

    image_files = os.listdir(dir_path)
    if not image_files:
        print(f"No images found in {dir_path}")
        continue

    img_path = os.path.join(dir_path, image_files[1])
    print(f"Processing image: {img_path}")

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw hand landmarks if found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        print("Hand landmarks detected.")
    else:
        print("No hand landmarks detected.")

    # Display the image using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.title(f'Hand Landmarks - {image_files[0]}')

# Clean up
hands.close()
plt.show()
