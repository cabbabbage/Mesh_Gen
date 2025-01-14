import os
import sys
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# External Libraries
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import trimesh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Delaunay


# Initialize MediaPipe Face Mesh and Face Detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=.4)
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
root.title("Mesh Gen")
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
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov*; *.mkv")])
    if video_path:
        setup_video_ui()

def setup_video_ui():
    clear_ui()
    ttk.Label(root, text="Enter frame sampling rate:").pack(pady=(20, 5))
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
    root.config(cursor="watch")
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

# Perspective Adjustment Function
def adjust_perspective(landmarks, image_width, image_height):
    adjusted_landmarks = []
    for lm in landmarks:
        x = lm.x * image_width
        y = lm.y * image_height
        z = lm.z * max(image_width, image_height)  # Scale Z dimension proportionally
        adjusted_landmarks.append((x, y, z))
    return np.array(adjusted_landmarks)

def create_mesh():
    clear_ui()
    root.config(cursor="watch")

    image_files = [f for f in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, f))]
    total_images = len(image_files)
    aggregated_landmarks = []

    frame_label = ttk.Label(root)
    frame_label.pack(pady=(20, 10))

    processing_label = ttk.Label(root, text="")
    processing_label.pack(pady=(0, 10))

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(media_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_height, image_width, _ = image.shape

        results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            continue

        for face_landmarks in results.multi_face_landmarks:
            adjusted_landmarks = adjust_perspective(face_landmarks.landmark, image_width, image_height)
            aggregated_landmarks.append(adjusted_landmarks)

            # Draw landmarks on the image
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

    # Average the landmarks to create the final mesh
    if aggregated_landmarks:
        processing_label.config(text=f"Generating Face Mesh...")
        root.update()
        averaged_landmarks = np.mean(aggregated_landmarks, axis=0)
        save_face_data(averaged_landmarks)
        #gen = imgs("final_outputs/face_data.obj")
        #gen.create_gif()      
        #gen.create_gif_with_surface()
        finalize_results_ui()
    root.config(cursor="")

def save_face_data(landmarks):
    obj_filepath = os.path.join(output_dir, "face_data.obj")
    with open(obj_filepath, "w") as f:
        f.write("# Generated by Mesh Gen\n")
        for x, y, z in landmarks:
            y = y * -1
            z = z * -1
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        f.write("\n")
    print(f"OBJ file saved at: {obj_filepath}")

# Normalize landmarks to fit within the image canvas
def normalize_landmarks(landmarks, image_size):
    normalized = []
    for lm in landmarks:
        x = int(lm[0] * image_size)
        y = int(lm[1] * image_size)
        z = lm[2]
        normalized.append((x, y, z))
    return normalized



def display_3d_obj():
    filepath = os.path.join(output_dir, "face_data.obj")
    if not os.path.exists(filepath):
        messagebox.showerror("Error", "OBJ file not found.")
        return

    try:
        # Load the OBJ file
        mesh = trimesh.load(filepath)

        # Create a new Tkinter window
        root = tk.Toplevel()
        root.title("3D Viewer")

        # Create a Matplotlib figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Function to set equal aspect ratio
        def set_equal_aspect_ratio(ax, vertices):
            x_range = max(vertices[:, 0]) - min(vertices[:, 0])
            y_range = max(vertices[:, 1]) - min(vertices[:, 1])
            z_range = max(vertices[:, 2]) - min(vertices[:, 2])
            max_range = max(x_range, y_range, z_range)

            ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])

        # Function to draw the point cloud
        def draw_dots():
            ax.clear()
            ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], c='lightblue', marker='o', s=4)
            set_equal_aspect_ratio(ax, mesh.vertices)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(False)
            ax.set_facecolor('white')
            ax.view_init(elev=90, azim=-90)
            canvas.draw()

        # Function to draw the surface
        def draw_surface():
            ax.clear()
            tri = Delaunay(mesh.vertices[:, :2])
            ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=tri.simplices, color='lightblue', alpha=0.7)
            set_equal_aspect_ratio(ax, mesh.vertices)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(False)
            ax.set_facecolor('white')
            ax.view_init(elev=90, azim=-90)
            canvas.draw()

        # Function to toggle between dots and surface
        def toggle_view():
            nonlocal plot_mode
            if plot_mode == "dots":
                draw_surface()
                plot_mode = "surface"
            else:
                draw_dots()
                plot_mode = "dots"

        # Create a canvas to embed the Matplotlib plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toggle button
        toggle_button = tk.Button(root, text="Toggle View", command=toggle_view)
        toggle_button.pack(pady=10)

        # Set initial plot mode and draw the dots view
        plot_mode = "dots"
        draw_dots()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def finalize_results_ui():
    root.config(cursor="")
    clear_ui()

    # Create a main frame to center all elements
    main_frame = ttk.Frame(root)
    main_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Add a button to display the 3D model
    view_3d_button = ttk.Button(main_frame, text="View 3D Model", command=lambda: display_3d_obj())
    view_3d_button.grid(row=0, column=0, pady=10)

    # Add a button to save the face_data.obj file
    save_button = ttk.Button(main_frame, text="Download Face Model (OBJ)", command=save_face_model)
    save_button.grid(row=1, column=0, pady=10)

    # Add a Done button to restart the app
    done_button = ttk.Button(main_frame, text="Done", command=restart)
    done_button.grid(row=2, column=0, pady=20)


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






# Centering the UI Elements
main_frame = ttk.Frame(root)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

label_instruction = ttk.Label(main_frame, text="To generate a face model, please select either a video or image folder containing a single face.")
label_instruction.grid(row=0, column=0, columnspan=2, pady=(0, 20))

button_frame = ttk.Frame(main_frame)
button_frame.grid(row=1, column=0, columnspan=2)

image_button = ttk.Button(button_frame, text="Select Images Folder", command=select_images_folder)
video_button = ttk.Button(button_frame, text="Select Video", command=select_video)

image_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
video_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

root.mainloop()

