import cv2
import mediapipe as mp
import numpy as np
import random
import streamlit as st
from PIL import Image

import base64
import os

def get_base64_image():
    for ext in ["webp", "jpg", "jpeg", "png"]:
        image_path = f"back.{ext}"
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    return None

bg_img = get_base64_image()

# --- Page Setup ---
if bg_img:
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.35), rgba(255, 255, 255, 0)),
                        url("data:image/png;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255, 248, 243, 0.45);
            padding: 2rem 3rem;
            border-radius: 18px;
            margin-top: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #c04600;
            font-family: 'Segoe UI', sans-serif;
        }}
        .export-buttons {{
            margin-top: 20px;
        }}
        .sidebar .sidebar-content {{
            background-color: #c04600;
            color: #f0e0d7;
        }}
        .sidebar .sidebar-content label, .sidebar .sidebar-content input, .sidebar .sidebar-content textarea, .sidebar .sidebar-content select {{
            color: #f0e0d7 !important;
        }}
        .sidebar .sidebar-content .stNumberInput > div > input,
        .sidebar .sidebar-content .stTextInput > div > input,
        .sidebar .sidebar-content .stTextArea > div > textarea {{
            background-color: #d15410;
            color: #f0e0d7 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# Streamlit app configuration
st.title("Facial Health Analysis")
st.write("Analyze facial regions and detect possible health indicators based on Mediapipe's Face Mesh.")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define regions, conditions, and their linked organs
regions = {
    "Forehead": {"points": [10, 338, 297, 332, 284, 251], "condition": "Redness", "implication": "Can indicate liver stress or excess toxins.", "organ": "Liver", "color": (0, 0, 255)},
    "Left Eye": {"points": [145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], "condition": "Dark Circles", "implication": "Linked to sleep deprivation", "organ": "Kidneys", "color": (255, 0, 0)},
    "Right Eye": {"points": [374, 380, 386, 387, 362, 398, 384, 385, 386, 387, 388, 466], "condition": "Dark Circles", "implication": "Kidney stress", "organ": "Kidneys", "color": (255, 0, 0)},
    "Left Cheek": {"points": [50, 101, 118, 202, 210, 58], "condition": "Dehydration", "implication": "May suggest low water intake or dry skin.", "organ": "Stomach", "color": (0, 255, 255)},
    "Right Cheek": {"points": [280, 352, 361, 429, 420, 408], "condition": "Dehydration", "implication": "Stomach issues", "organ": "Stomach", "color": (0, 255, 255)},
    "Nose": {"points": [1, 4, 5, 197, 195, 6], "condition": "Oiliness", "implication": "Hormonal imbalance", "organ": "Heart", "color": (0, 255, 0)},
    "Chin": {"points": [152, 148, 176, 149, 150, 400], "condition": "Dryness", "implication": "Hormonal changes", "organ": "Hormonal System", "color": (255, 255, 0)},
}

def main():
    # Show reference face map
    st.image("acne face map.jpg", caption="Reference Image", use_container_width=True)

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        height, width, _ = image.shape

        # Convert to RGB (required by Mediapipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = face_mesh.process(rgb_image)

        # Create overlay
        overlay = np.zeros_like(image, dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for region_name, details in regions.items():
                    region_indices = details["points"]
                    region_color = details["color"]

                    region_polygon = np.array(
                        [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in region_indices],
                        dtype=np.int32,
                    )

                    # Shift eye regions slightly downward
                    if region_name in ["Left Eye", "Right Eye"]:
                        region_polygon[:, 1] += 20

                    # Generate random points inside the polygon
                    random_points = []
                    x_min, y_min = np.min(region_polygon, axis=0)
                    x_max, y_max = np.max(region_polygon, axis=0)
                    while len(random_points) < 200:
                        x = random.randint(x_min, x_max)
                        y = random.randint(y_min, y_max)
                        if cv2.pointPolygonTest(region_polygon, (x, y), False) >= 0:
                            random_points.append((x, y))

                    for point in random_points:
                        cv2.circle(overlay, point, 1, region_color, -1)

        # Combine and show
        output_image = cv2.addWeighted(image, 0.8, overlay, 0.6, 0)
        st.image(output_image, caption="Analyzed Image", use_container_width=True)

        # Show analysis summary
        st.subheader("Analysis Details")
        for region_name, details in regions.items():
            st.write(f"**{region_name}**: {details['condition']} — *{details['implication']}*")

# Fix the __main__ call
if __name__ == "__main__":
    main()
