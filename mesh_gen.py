import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import sys
import shutil

# Initialize MediaPipe Face Mesh and Face Detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Reset
media_dir = "media"
output_dir = "final_outputs"

# Remove and recreate the media directory
if os.path.exists(media_dir):
    shutil.rmtree(media_dir)
os.makedirs(media_dir, exist_ok=True)

# Remove and recreate the output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# UI Setup
root = tk.Tk()
root.title("Face Mesh Setup")
root.geometry("900x500")
root.resizable(False, False)
root.configure(bg="#f0f0f0")
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0")

# Global Variables
folder_path = ""
video_path = ""
frame_interval = 1
face_data = []

def clear_ui():
    for widget in root.winfo_children():
        widget.destroy()

# Functions to handle button clicks
def select_images_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                cv2.imwrite(os.path.join(media_dir, os.path.basename(file_path)), cv2.imread(file_path))
        root.config(cursor="watch")
        check()

def select_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        setup_video_ui()

def setup_video_ui():
    clear_ui()
    ttk.Label(root, text="Analyze every X frames:").pack(pady=(20, 5))
    entry = ttk.Entry(root)
    entry.pack(pady=(0, 10))
    ttk.Button(root, text="Continue", command=lambda: continue_video_processing(entry)).pack(pady=(10, 20))

def continue_video_processing(entry):
    global frame_interval
    try:
        frame_interval = int(entry.get())
    except ValueError:
        frame_interval = 1

    process_video()

def process_video():
    clear_ui()
    label = ttk.Label(root, text="Analyzing video... Please wait.")
    label.pack(pady=(20, 10))
    root.update()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = os.path.join(media_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_name, frame)

        frame_count += 1

    cap.release()
    root.config(cursor="watch")
    check()

def check():
    clear_ui()
    label = ttk.Label(root, text="Checking media... Please wait.")
    label.pack(pady=(20, 10))
    root.update()

    valid_images = 0
    for image_file in os.listdir(media_dir):
        image_path = os.path.join(media_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            os.remove(image_path)
            continue

        results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections and len(results.detections) == 1:
            valid_images += 1
        else:
            os.remove(image_path)

    finalize_ui(valid_images)

def finalize_ui(valid_images):
    clear_ui()
    label = ttk.Label(root, text=f"{valid_images} valid images found.")
    label.pack(pady=(20, 10))
    root.config(cursor="")
    if valid_images > 0:
        ttk.Button(root, text="Create Mesh", command=create_mesh).pack(pady=(10, 20))

def create_mesh():
    clear_ui()
    root.config(cursor="watch")
    image_files = [f for f in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, f))]
    total_images = len(image_files)

    frame_label = ttk.Label(root)
    frame_label.pack(pady=(20, 10))

    processing_label = ttk.Label(root, text="")
    processing_label.pack(pady=(0, 10))

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(media_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            continue

        for face_landmarks in results.multi_face_landmarks:
            face_data.append([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_pil = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb).resize((400, 200)))

        frame_label.config(image=frame_pil)
        frame_label.image = frame_pil
        processing_label.config(text=f"Processing frame {idx + 1} of {total_images}")

        root.update()

    save_face_data()
    create_final_images()
    finalize_results_ui()

def save_face_data():
    with open(os.path.join(output_dir, "face_data.obj"), "w") as f:
        for landmarks in face_data:
            for x, y, z in landmarks:
                f.write(f"v {x} {y} {z}\n")

def create_final_images():
    aggregated_data = np.mean(np.array(face_data), axis=0)
    aggregated_image = np.zeros((512, 512, 3), dtype=np.uint8)

    for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        start_idx, end_idx = connection
        start_point = (int(aggregated_data[start_idx][0] * 512), int(aggregated_data[start_idx][1] * 512))
        end_point = (int(aggregated_data[end_idx][0] * 512), int(aggregated_data[end_idx][1] * 512))
        cv2.line(aggregated_image, start_point, end_point, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "final_mesh_lines.png"), aggregated_image)

    textured_image = np.zeros_like(aggregated_image)
    for landmark in aggregated_data:
        x, y = int(landmark[0] * 512), int(landmark[1] * 512)
        cv2.circle(textured_image, (x, y), 1, (200, 180, 150), -1)

    cv2.imwrite(os.path.join(output_dir, "final_textured_face.png"), textured_image)

def finalize_results_ui():
    root.config(cursor="")
    clear_ui()
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    ttk.Button(scrollable_frame, text="Save Face Model (OBJ)", command=save_face_model).pack(pady=10)

    final_images = ["final_mesh_lines.png", "final_textured_face.png"]
    for img in final_images:
        img_path = os.path.join(output_dir, img)
        image = Image.open(img_path)
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        img_display = ImageTk.PhotoImage(image)
        img_label = ttk.Label(scrollable_frame, image=img_display)
        img_label.image = img_display
        img_label.pack(pady=5)
        img_label.bind("<Button-1>", lambda e, path=img_path: open_file_explorer(path))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    ttk.Button(scrollable_frame, text="Done", command=restart).pack(pady=20)

def save_face_model():
    file_path = filedialog.asksaveasfilename(defaultextension=".obj", filetypes=[("OBJ files", "*.obj")])
    if file_path:
        os.rename(os.path.join(output_dir, "face_data.obj"), file_path)
        messagebox.showinfo("Saved", f"Face model saved at {file_path}")

def open_file_explorer(path):
    os.startfile(path)

def restart():
    clear_ui()
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    messagebox.showinfo("Restart", "All temporary files deleted. Restarting the app.")
    root.destroy()
    os.execl(sys.executable, sys.executable, *sys.argv)

# Initial Buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=(40, 40), padx=(10, 10), fill=tk.BOTH, expand=True)

image_button = ttk.Button(button_frame, text="Select Images Folder", command=select_images_folder)
video_button = ttk.Button(button_frame, text="Select Video", command=select_video)

image_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
video_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

root.mainloop()