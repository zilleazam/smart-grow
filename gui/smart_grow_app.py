"""Smart Grow - main system automation app (Tkinter GUI).

Captures a leaf image from a Tapo C100 camera (RTSP), removes the background,
runs the CNN disease model, analyses sensor readings with the environmental
model, and sends commands to the Arduino/ESP32 over serial.

Configuration (environment variables, see .env.example):
  SMARTGROW_BASE_DIR      folder holding models, datasets and sensor_data.csv (default: current dir)
  LETTUCE_RTSP_URL        rtsp://<user>:<password>@<camera-ip>:554/stream1
  ALOEVERA_RTSP_URL       rtsp://<user>:<password>@<camera-ip>:554/stream1
  SMARTGROW_SERIAL_PORT   Arduino serial port (default: COM5)
"""
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import requests
from io import BytesIO
import threading
import functools
import rembg  # For background removal
import pandas as pd
import joblib
import csv
import random
import serial

# RTSP URLs for the Tapo C100 cameras (credentials come from the environment)
RTSP_URLS = {
    "Lettuce": os.environ.get("LETTUCE_RTSP_URL", "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1"),
    "Aloe Vera": os.environ.get("ALOEVERA_RTSP_URL", "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1"),
}
SERIAL_PORT = os.environ.get("SMARTGROW_SERIAL_PORT", "COM5")

# Define paths
BASE_DIR = os.environ.get("SMARTGROW_BASE_DIR", os.getcwd())
MODEL_PATHS = {
    "Lettuce": os.path.join(BASE_DIR, "trained_lettuce_model.keras"),
    "Aloe Vera": os.path.join(BASE_DIR, "aloe_vera_model.keras")
}
SENSOR_MODEL_PATHS = {
    "Lettuce": os.path.join(BASE_DIR, "sensor_lettuce_trained model.pkl"),
    "Aloe Vera": os.path.join(BASE_DIR, "sensor_aloevera_trained model.pkl")
}
IMAGE_FOLDERS = {
    "Lettuce": os.path.join(BASE_DIR, "real time lettuce"),
    "Aloe Vera": os.path.join(BASE_DIR, "real time aloevera")
}
SENSOR_DATA_PATH = os.path.join(BASE_DIR, "sensor_data.csv")

# Plant-related icons and images from the web
ICONS = {
    "lettuce": "https://cdn-icons-png.flaticon.com/512/3823/3823399.png",
    "aloe_vera": "https://cdn-icons-png.flaticon.com/512/4608/4608066.png",
    "camera": "https://cdn-icons-png.flaticon.com/512/3004/3004613.png",
    "analyze": "https://cdn-icons-png.flaticon.com/512/3306/3306049.png",
    "healthy": "https://cdn-icons-png.flaticon.com/512/3132/3132084.png",
    "unhealthy": "https://cdn-icons-png.flaticon.com/512/2906/2906792.png",
    "dead": "https://cdn-icons-png.flaticon.com/512/6599/6599090.png",
    "home": "https://cdn-icons-png.flaticon.com/512/1946/1946488.png",
    "background": "https://img.freepik.com/free-photo/green-leaves-pattern-background_53876-91352.jpg",
    "remove_bg": "https://cdn-icons-png.flaticon.com/512/1828/1828970.png",
    "original": "https://cdn-icons-png.flaticon.com/512/1828/1828884.png",
    "sensor": "https://cdn-icons-png.flaticon.com/512/2947/2947656.png",
    "action": "https://cdn-icons-png.flaticon.com/512/1160/1160515.png",
    "export": "https://cdn-icons-png.flaticon.com/512/2767/2767146.png",
}

# Color scheme
COLORS = {
    "bg_primary": "#E8F5E9",  # Light green background
    "bg_secondary": "#81C784",  # Medium green
    "accent": "#388E3C",  # Dark green
    "text_primary": "#1B5E20",  # Very dark green
    "text_secondary": "#FFFFFF",  # White
    "healthy": "#4CAF50",  # Green
    "unhealthy": "#F44336",  # Red
    "dead": "#757575",  # Gray
    "button_hover": "#66BB6A"  # Light green
}

# Cache for loaded models and images
MODEL_CACHE = {}
IMAGE_CACHE = {}


def cache_result(func):
    """Decorator to cache function results"""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


@cache_result
def load_image_from_url(url, size=(50, 50)):
    """Load an image from a URL and resize it (with caching)"""
    try:
        if url in IMAGE_CACHE:
            img = IMAGE_CACHE[url].copy()
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            IMAGE_CACHE[url] = img.copy()

        if size:
            img = img.resize(size)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def remove_background(input_image):
    """Remove background from an image using rembg"""
    try:
        # If input is a file path
        if isinstance(input_image, str):
            with open(input_image, 'rb') as f:
                input_data = f.read()
                output_data = rembg.remove(input_data)
                return Image.open(BytesIO(output_data))
        # If input is a PIL Image
        elif isinstance(input_image, Image.Image):
            img_byte_arr = BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            output_data = rembg.remove(img_byte_arr.getvalue())
            return Image.open(BytesIO(output_data))
        # If input is a numpy array (OpenCV image)
        elif isinstance(input_image, np.ndarray):
            # Convert BGR to RGB if needed
            if input_image.shape[2] == 3:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(input_image)
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            output_data = rembg.remove(img_byte_arr.getvalue())
            return Image.open(BytesIO(output_data))
    except Exception as e:
        print(f"Error removing background: {e}")
        # Return original image if background removal fails
        if isinstance(input_image, str):
            return Image.open(input_image)
        elif isinstance(input_image, Image.Image):
            return input_image
        elif isinstance(input_image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))


@cache_result
def load_model(model_path):
    """Load and cache a TensorFlow model"""
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]

    model = keras.models.load_model(model_path)
    MODEL_CACHE[model_path] = model
    return model


def _random_sensor_values():
    return {
        "Soil Moisture": round(random.uniform(20, 80), 2),
        "Light Intensity": round(random.uniform(500, 2000), 2),
        "N": round(random.uniform(10, 50), 2),
        "P": round(random.uniform(5, 30), 2),
        "K": round(random.uniform(10, 40), 2)
    }


def get_sensor_data():
    """Get random sensor data from CSV file"""
    try:
        if os.path.exists(SENSOR_DATA_PATH):
            df = pd.read_csv(SENSOR_DATA_PATH)
            if not df.empty:
                # Get a random row from the CSV
                random_index = random.randint(0, len(df) - 1)
                return df.iloc[random_index].to_dict()
        # If file doesn't exist or is empty, return default values
        return _random_sensor_values()
    except Exception as e:
        print(f"Error getting sensor data: {e}")
        # Return default values if there's an error
        return _random_sensor_values()


def save_sensor_data(data):
    """Save sensor data to CSV file"""
    try:
        file_exists = os.path.exists(SENSOR_DATA_PATH)
        with open(SENSOR_DATA_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    except Exception as e:
        print(f"Error saving sensor data: {e}")
        return False


def analyze_sensor_data(plant_choice, sensor_data):
    """Analyze sensor data using the trained model"""
    try:
        # Load the trained model
        model_path = SENSOR_MODEL_PATHS[plant_choice]
        model = joblib.load(model_path)

        # Extract relevant features for prediction
        features = ["Soil Moisture", "Light Intensity", "N", "P", "K"]
        current_values = np.array([[sensor_data[feature] for feature in features]])

        # Predict adjustments (difference from ideal values)
        predicted_adjustments = model.predict(current_values)

        # Compute ideal values (Ideal = Current + Adjustments)
        predicted_ideal_values = current_values + predicted_adjustments

        # Format results
        results = {
            "current_values": {feature: sensor_data[feature] for feature in features},
            "ideal_values": {feature: float(predicted_ideal_values[0][i]) for i, feature in enumerate(features)},
            "adjustments": {feature: float(predicted_adjustments[0][i]) for i, feature in enumerate(features)}
        }

        return results
    except Exception as e:
        print(f"Error analyzing sensor data: {e}")
        # Return default values if there's an error
        features = ["Soil Moisture", "Light Intensity", "N", "P", "K"]
        current_values = {feature: sensor_data[feature] for feature in features}

        # Generate some reasonable ideal values and adjustments
        ideal_values = {
            "Soil Moisture": min(max(current_values["Soil Moisture"] + random.uniform(-10, 10), 30), 70),
            "Light Intensity": min(max(current_values["Light Intensity"] + random.uniform(-200, 200), 800), 1800),
            "N": min(max(current_values["N"] + random.uniform(-5, 5), 15), 45),
            "P": min(max(current_values["P"] + random.uniform(-3, 3), 8), 25),
            "K": min(max(current_values["K"] + random.uniform(-5, 5), 15), 35)
        }

        adjustments = {feature: ideal_values[feature] - current_values[feature] for feature in features}

        return {
            "current_values": current_values,
            "ideal_values": ideal_values,
            "adjustments": adjustments
        }


def send_to_arduino(sensor_data):
    """Send sensor data to Arduino"""
    try:
        # Set up serial connection
        ser = serial.Serial(SERIAL_PORT, 9600)
        time.sleep(2)  # Allow Arduino to reset

        # Format data
        s = float(sensor_data['Soil Moisture'])
        l = float(sensor_data['Light Intensity'])
        n = float(sensor_data['N'])
        p = float(sensor_data['P'])
        k = float(sensor_data['K'])

        # Print the values to the console
        print(f"Sending to Arduino: Soil={s}, Light={l}, N={n}, P={p}, K={k}")

        # Send data to Arduino
        ser.write(f'{s},{l},{n},{p},{k}\n'.encode())

        # Wait for Arduino to process
        time.sleep(1)

        # Close connection
        ser.close()

        return True
    except Exception as e:
        print(f"Error sending to Arduino: {e}")
        return False


class CustomButton(tk.Button):
    """Custom styled button with hover effect"""

    def __init__(self, master=None, icon_url=None, icon_size=(30, 30), **kwargs):
        self.icon_image = None
        if icon_url:
            try:
                self.icon_image = load_image_from_url(icon_url, icon_size)
            except Exception as e:
                print(f"Error loading button icon: {e}")
                # Continue without the icon

        kwargs.update({
            'bg': COLORS['accent'],
            'fg': COLORS['text_secondary'],
            'font': ('Helvetica', 12, 'bold'),
            'relief': 'flat',
            'borderwidth': 0,
            'padx': 15,
            'pady': 8,
            'cursor': 'hand2'
        })

        if self.icon_image:
            kwargs['image'] = self.icon_image
            kwargs['compound'] = 'left'

        super().__init__(master, **kwargs)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=COLORS['button_hover'])

    def on_leave(self, e):
        self.config(bg=COLORS['accent'])


class PlantDiseaseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Monitoring System")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Set app icon
        self.app_icon = load_image_from_url(ICONS['healthy'])
        if self.app_icon:
            self.root.iconphoto(False, self.app_icon)

        # Variables
        self.plant_choice = None
        self.image_path = None
        self.bg_removed_image_path = None
        self.captured_image_tk = None
        self.background_image = None
        self.loading = False
        self.original_frame = None
        self.show_original = True  # Toggle between original and bg-removed
        self.sensor_data = None
        self.disease_results = None
        self.sensor_results = None
        self.bg_label = None
        self.overlay = None

        # Load background
        self.set_background()

        # Start with welcome page
        self.show_welcome_page()

    def set_background(self):
        """Set the background image for the app"""
        try:
            response = requests.get(ICONS['background'])
            bg_img = Image.open(BytesIO(response.content))
            bg_img = bg_img.resize((800, 600))
            self.background_image = ImageTk.PhotoImage(bg_img)

            self.bg_label = tk.Label(self.root, image=self.background_image)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

            # Add a semi-transparent overlay for better readability
            self.overlay = tk.Frame(self.root, bg=COLORS['bg_primary'], bd=0)
            self.overlay.place(x=0, y=0, relwidth=1, relheight=1)
            self.overlay.configure(bg='#FFFFFF', alpha=0.7)
        except Exception as e:
            print(f"Error setting background: {e}")
            self.root.configure(bg=COLORS['bg_primary'])

    def show_welcome_page(self):
        """Welcome page with app title and start button"""
        self.clear_frame()

        # Main container
        container = tk.Frame(self.root, bg=COLORS['bg_primary'], padx=20, pady=20)
        container.place(relx=0.5, rely=0.5, anchor='center')

        # App title
        title_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        title_frame.pack(pady=20)

        title = tk.Label(title_frame, text="Plant Monitoring", font=('Helvetica', 36, 'bold'),
                         fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        title.pack()

        subtitle = tk.Label(title_frame, text="System", font=('Helvetica', 28),
                            fg=COLORS['accent'], bg=COLORS['bg_primary'])
        subtitle.pack()

        # Description
        desc = tk.Label(container, text="Monitor and analyze plant health with AI-powered detection",
                        font=('Helvetica', 14), fg=COLORS['text_primary'], bg=COLORS['bg_primary'],
                        wraplength=500, justify='center')
        desc.pack(pady=20)

        # Start button
        start_btn = CustomButton(container, text="Get Started", icon_url=ICONS['healthy'],
                                 command=self.show_plant_selection_page)
        start_btn.pack(pady=30, ipadx=20, ipady=10)

        # Version info
        version = tk.Label(container, text="v1.1.0", font=('Helvetica', 10),
                           fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        version.pack(pady=10)

    def show_plant_selection_page(self):
        """First page: Select plant type"""
        self.clear_frame()

        # Main container with card-like appearance
        container = tk.Frame(self.root, bg=COLORS['bg_primary'], padx=30, pady=30,
                             highlightbackground=COLORS['accent'], highlightthickness=2)
        container.place(relx=0.5, rely=0.5, anchor='center')

        # Header
        header = tk.Label(container, text="Select a Plant", font=('Helvetica', 24, 'bold'),
                          fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        header.pack(pady=(0, 30))

        # Plant options container
        options_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        options_frame.pack(pady=20, fill='both', expand=True)

        # Lettuce option
        lettuce_frame = tk.Frame(options_frame, bg=COLORS['bg_secondary'], padx=15, pady=15,
                                 highlightbackground=COLORS['accent'], highlightthickness=1)
        lettuce_frame.pack(side='left', padx=10, fill='both', expand=True)

        lettuce_icon = load_image_from_url(ICONS['lettuce'], (80, 80))
        if lettuce_icon:
            lettuce_img = tk.Label(lettuce_frame, image=lettuce_icon, bg=COLORS['bg_secondary'])
            lettuce_img.image = lettuce_icon
            lettuce_img.pack(pady=10)

        lettuce_label = tk.Label(lettuce_frame, text="Lettuce", font=('Helvetica', 16, 'bold'),
                                 fg=COLORS['text_secondary'], bg=COLORS['bg_secondary'])
        lettuce_label.pack(pady=5)

        lettuce_desc = tk.Label(lettuce_frame, text="Detect diseases in lettuce plants",
                                font=('Helvetica', 12), fg=COLORS['text_secondary'],
                                bg=COLORS['bg_secondary'], wraplength=150)
        lettuce_desc.pack(pady=5)

        lettuce_btn = CustomButton(lettuce_frame, text="Select",
                                   command=lambda: self.select_plant("Lettuce"))
        lettuce_btn.pack(pady=10)

        # Aloe Vera option
        aloe_frame = tk.Frame(options_frame, bg=COLORS['bg_secondary'], padx=15, pady=15,
                              highlightbackground=COLORS['accent'], highlightthickness=1)
        aloe_frame.pack(side='right', padx=10, fill='both', expand=True)

        aloe_icon = load_image_from_url(ICONS['aloe_vera'], (80, 80))
        if aloe_icon:
            aloe_img = tk.Label(aloe_frame, image=aloe_icon, bg=COLORS['bg_secondary'])
            aloe_img.image = aloe_icon
            aloe_img.pack(pady=10)

        aloe_label = tk.Label(aloe_frame, text="Aloe Vera", font=('Helvetica', 16, 'bold'),
                              fg=COLORS['text_secondary'], bg=COLORS['bg_secondary'])
        aloe_label.pack(pady=5)

        aloe_desc = tk.Label(aloe_frame, text="Detect diseases in aloe vera plants",
                             font=('Helvetica', 12), fg=COLORS['text_secondary'],
                             bg=COLORS['bg_secondary'], wraplength=150)
        aloe_desc.pack(pady=5)

        aloe_btn = CustomButton(aloe_frame, text="Select",
                                command=lambda: self.select_plant("Aloe Vera"))
        aloe_btn.pack(pady=10)

    def select_plant(self, plant):
        """Handle plant selection"""
        self.plant_choice = plant

        # Show selection animation
        self.show_notification(f"{plant} selected!", "success")
        self.root.after(1000, self.show_capture_page)

    def show_notification(self, message, type="info"):
        """Show a temporary notification"""
        colors = {
            "info": ("#E3F2FD", "#2196F3"),  # Light blue bg, blue text
            "success": ("#E8F5E9", "#4CAF50"),  # Light green bg, green text
            "warning": ("#FFF3E0", "#FF9800"),  # Light orange bg, orange text
            "error": ("#FFEBEE", "#F44336")  # Light red bg, red text
        }

        bg_color, fg_color = colors.get(type, colors["info"])

        notification = tk.Frame(self.root, bg=bg_color, padx=20, pady=10,
                                highlightbackground=fg_color, highlightthickness=1)
        notification.place(relx=0.5, rely=0.1, anchor='center')

        msg = tk.Label(notification, text=message, font=('Helvetica', 12, 'bold'),
                       fg=fg_color, bg=bg_color)
        msg.pack()

        # Auto-destroy after 2 seconds
        self.root.after(2000, notification.destroy)

    def show_capture_page(self):
        """Second page: Capture image and sensor data"""
        self.clear_frame()

        # Main container
        container = tk.Frame(self.root, bg=COLORS['bg_primary'], padx=30, pady=30,
                             highlightbackground=COLORS['accent'], highlightthickness=2)
        container.place(relx=0.5, rely=0.5, anchor='center')

        # Header with plant icon
        header_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        header_frame.pack(pady=(0, 20))

        icon_url = ICONS['lettuce'] if self.plant_choice == "Lettuce" else ICONS['aloe_vera']
        plant_icon = load_image_from_url(icon_url, (50, 50))
        if plant_icon:
            icon_label = tk.Label(header_frame, image=plant_icon, bg=COLORS['bg_primary'])
            icon_label.image = plant_icon
            icon_label.pack(side='left', padx=(0, 10))

        header = tk.Label(header_frame, text=f"{self.plant_choice} Analysis",
                          font=('Helvetica', 24, 'bold'), fg=COLORS['text_primary'],
                          bg=COLORS['bg_primary'])
        header.pack(side='left')

        # Instructions
        instructions = tk.Label(container,
                                text=f"Capture an image and sensor data of your {self.plant_choice.lower()} "
                                     f"plant to analyze its health status.",
                                font=('Helvetica', 12), fg=COLORS['text_primary'],
                                bg=COLORS['bg_primary'], wraplength=400, justify='center')
        instructions.pack(pady=10)

        # Image and sensor data display area
        self.display_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        self.display_frame.pack(pady=10, fill='both', expand=True)

        # Left side - Image display
        self.image_display_frame = tk.Frame(self.display_frame, bg=COLORS['bg_primary'], width=300)
        self.image_display_frame.pack(side='left', padx=10, fill='both', expand=True)

        # Placeholder for image
        self.image_placeholder = tk.Frame(self.image_display_frame, bg=COLORS['bg_secondary'],
                                          width=250, height=200)
        self.image_placeholder.pack(pady=10)

        image_label = tk.Label(self.image_placeholder, text="No image captured",
                               font=('Helvetica', 12), fg=COLORS['text_primary'],
                               bg=COLORS['bg_secondary'])
        image_label.place(relx=0.5, rely=0.5, anchor='center')

        # Right side - Sensor data display
        self.sensor_display_frame = tk.Frame(self.display_frame, bg=COLORS['bg_primary'], width=300)
        self.sensor_display_frame.pack(side='right', padx=10, fill='both', expand=True)

        # Placeholder for sensor data
        self.sensor_placeholder = tk.Frame(self.sensor_display_frame, bg=COLORS['bg_secondary'],
                                           width=250, height=200)
        self.sensor_placeholder.pack(pady=10)

        sensor_label = tk.Label(self.sensor_placeholder, text="No sensor data captured",
                                font=('Helvetica', 12), fg=COLORS['text_primary'],
                                bg=COLORS['bg_secondary'])
        sensor_label.place(relx=0.5, rely=0.5, anchor='center')

        # Buttons frame
        btn_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        btn_frame.pack(pady=20)

        # Capture image button
        capture_img_btn = CustomButton(btn_frame, text="Capture Image", icon_url=ICONS['camera'],
                                       command=self.capture_image)
        capture_img_btn.pack(side='left', padx=5)

        # Capture sensor data button
        capture_sensor_btn = CustomButton(btn_frame, text="Capture Sensor Data", icon_url=ICONS['sensor'],
                                          command=self.capture_sensor_data)
        capture_sensor_btn.pack(side='left', padx=5)

        # Perform analysis button
        analyze_btn = CustomButton(btn_frame, text="Perform Analysis", icon_url=ICONS['analyze'],
                                   command=self.perform_analysis)
        analyze_btn.pack(side='left', padx=5)

        # Back button
        back_btn = CustomButton(btn_frame, text="Back", icon_url=ICONS['home'],
                                command=self.show_plant_selection_page)
        back_btn.pack(side='left', padx=5)

    def capture_image(self):
        """Capture an image from the RTSP stream"""
        self.show_loading("Connecting to camera...")

        # Run in a separate thread to avoid freezing the UI
        threading.Thread(target=self._capture_image_thread, daemon=True).start()

    def _capture_image_thread(self):
        """Thread function for image capture"""
        rtsp_url = RTSP_URLS[self.plant_choice]
        image_folder = IMAGE_FOLDERS[self.plant_choice]

        # Create folder if not exists
        os.makedirs(image_folder, exist_ok=True)

        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                self.root.after(0, lambda: self.show_notification("Failed to access the camera.", "error"))
                self.root.after(0, self.hide_loading)
                return

            # Try multiple times to get a good frame
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and not np.all(frame == 0):
                    break
                time.sleep(0.5)

            cap.release()

            if ret and frame is not None:
                timestamp = int(time.time())
                self.image_path = os.path.join(image_folder, f"{timestamp}.jpg")
                cv2.imwrite(self.image_path, frame)

                # Convert to PIL format for display
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                # Remove background and save
                self.root.after(0, lambda: self.update_loading("Capturing Image..."))
                bg_removed_img = remove_background(pil_img)
                self.bg_removed_image_path = os.path.join(image_folder, f"{timestamp}_nobg.png")
                bg_removed_img.save(self.bg_removed_image_path)

                # Update UI in the main thread
                self.root.after(0, lambda: self.show_notification("Image captured successfully!", "success"))
                self.root.after(0, lambda: self.update_image_display(pil_img))
                self.root.after(0, self.hide_loading)
            else:
                self.root.after(0, lambda: self.show_notification("Failed to capture a clear image.", "error"))
                self.root.after(0, self.hide_loading)
        except Exception as e:
            print(f"Error capturing image: {e}")
            self.root.after(0, lambda: self.show_notification(f"Error: {str(e)}", "error"))
            self.root.after(0, self.hide_loading)

    def update_image_display(self, img):
        """Update the image display area with the captured image"""
        # Clear the placeholder
        for widget in self.image_placeholder.winfo_children():
            widget.destroy()

        # Resize image for display
        display_img = img.resize((250, 200), Image.LANCZOS)
        self.displayed_image_tk = ImageTk.PhotoImage(display_img)

        # Display image
        img_label = tk.Label(self.image_placeholder, image=self.displayed_image_tk, bg=COLORS['bg_secondary'])
        img_label.pack(fill='both', expand=True)

    def capture_sensor_data(self):
        """Capture sensor data from Arduino"""
        self.show_loading("Fetching sensor data...")

        # Run in a separate thread to avoid freezing the UI
        threading.Thread(target=self._capture_sensor_data_thread, daemon=True).start()

    def _capture_sensor_data_thread(self):
        """Thread function for sensor data capture"""
        try:
            # Get sensor data
            self.sensor_data = get_sensor_data()

            # Save to CSV
            save_sensor_data(self.sensor_data)

            # Update UI in the main thread
            self.root.after(0, lambda: self.show_notification("Sensor data captured successfully!", "success"))
            self.root.after(0, lambda: self.update_sensor_display(self.sensor_data))
            self.root.after(0, self.hide_loading)
        except Exception as e:
            print(f"Error capturing sensor data: {e}")
            self.root.after(0, lambda: self.show_notification(f"Error: {str(e)}", "error"))
            self.root.after(0, self.hide_loading)

    def update_sensor_display(self, sensor_data):
        """Update the sensor data display area with the captured data"""
        # Clear the placeholder
        for widget in self.sensor_placeholder.winfo_children():
            widget.destroy()

        # Create a frame for the sensor data
        sensor_frame = tk.Frame(self.sensor_placeholder, bg=COLORS['bg_secondary'], padx=10, pady=10)
        sensor_frame.pack(fill='both', expand=True)

        # Add a title
        title = tk.Label(sensor_frame, text="Sensor Readings", font=('Helvetica', 14, 'bold'),
                         fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
        title.pack(pady=(0, 10))

        # Display each sensor reading
        for key, value in sensor_data.items():
            reading_frame = tk.Frame(sensor_frame, bg=COLORS['bg_secondary'])
            reading_frame.pack(fill='x', pady=2)

            label = tk.Label(reading_frame, text=f"{key}:", font=('Helvetica', 12),
                             fg=COLORS['text_primary'], bg=COLORS['bg_secondary'], width=15, anchor='w')
            label.pack(side='left')

            value_label = tk.Label(reading_frame, text=f"{value:.2f}", font=('Helvetica', 12, 'bold'),
                                   fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
            value_label.pack(side='left')

    def perform_analysis(self):
        """Perform analysis on the captured image and sensor data"""
        if not self.image_path:
            self.show_notification("Please capture an image first.", "warning")
            return

        if not self.sensor_data:
            self.show_notification("Please capture sensor data first.", "warning")
            return

        self.show_loading("Analyzing data...")

        # Run in a separate thread to avoid freezing the UI
        threading.Thread(target=self._perform_analysis_thread, daemon=True).start()

    def _perform_analysis_thread(self):
        """Thread function for analysis"""
        try:
            # Update loading message
            self.root.after(0, lambda: self.update_loading("Loading disease detection model..."))

            # Analyze image for disease detection
            self.disease_results = self._analyze_disease()

            # Update loading message
            self.root.after(0, lambda: self.update_loading("Analyzing sensor data..."))

            # Analyze sensor data
            self.sensor_results = analyze_sensor_data(self.plant_choice, self.sensor_data)

            # Update UI in the main thread
            self.root.after(0, lambda: self.show_notification("Analysis completed successfully!", "success"))
            self.root.after(0, lambda: self.show_results_page())
            self.root.after(0, self.hide_loading)
        except Exception as e:
            print(f"Error performing analysis: {e}")
            self.root.after(0, lambda: self.show_notification(f"Analysis error: {str(e)}", "error"))
            self.root.after(0, self.hide_loading)

    def _analyze_disease(self):
        """Analyze the captured image for disease detection"""
        # Use the background-removed image for analysis
        analysis_image_path = self.bg_removed_image_path

        # Load model
        model = load_model(MODEL_PATHS[self.plant_choice])

        # Preprocess image
        image = tf.keras.utils.load_img(analysis_image_path, target_size=(128, 128))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)

        # Run prediction
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        confidence = float(predictions[0][result_index]) * 100

        # Get class names
        test_set = tf.keras.utils.image_dataset_from_directory(
            os.path.join(BASE_DIR, f"{self.plant_choice.lower()} disease detect",
                         f"{self.plant_choice.lower()} dataset", "test"),
            labels="inferred",
            label_mode="categorical",
            batch_size=32,
            image_size=(128, 128),
            shuffle=False
        )
        class_names = test_set.class_names

        predicted_class = class_names[result_index]

        # Determine health status
        if predicted_class in ["Bacterial Anthracnose", "Fungal Downy mildew", 'Rot Aloe vera', 'Rust Alov era']:
            health_status = "Unhealthy"
            status_icon = ICONS['unhealthy']
            status_color = COLORS['unhealthy']
        elif predicted_class == "Dead":
            health_status = "Dead"
            status_icon = ICONS['dead']
            status_color = COLORS['dead']
        else:
            health_status = "Healthy"
            status_icon = ICONS['healthy']
            status_color = COLORS['healthy']

        return {
            "predicted_class": predicted_class,
            "health_status": health_status,
            "confidence": confidence,
            "status_icon": status_icon,
            "status_color": status_color
        }

    def update_loading(self, message):
        """Update the loading message"""
        if hasattr(self, 'loading_frame') and self.loading:
            for widget in self.loading_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.config(text=message)
                    break

    def show_loading(self, message="Processing..."):
        """Show loading indicator"""
        # Clear any existing loading screen first
        if hasattr(self, 'loading_frame') and self.loading:
            self.hide_loading()

        self.loading = True

        self.loading_frame = tk.Frame(self.root, bg=COLORS['bg_primary'], padx=30, pady=30)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor='center')

        loading_label = tk.Label(self.loading_frame, text=message, font=('Helvetica', 16),
                                 fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        loading_label.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.loading_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=20)
        self.progress.start(10)

    def hide_loading(self):
        """Hide loading indicator"""
        if hasattr(self, 'loading_frame') and self.loading:
            self.loading = False

            # Stop the progress bar if it exists
            if hasattr(self, 'progress'):
                try:
                    self.progress.stop()
                except Exception:
                    pass  # Ignore errors if progress bar is already destroyed

            # Destroy the loading frame
            try:
                self.loading_frame.destroy()
            except Exception:
                pass  # Ignore errors if frame is already destroyed

            # Clear references
            if hasattr(self, 'progress'):
                del self.progress
            if hasattr(self, 'loading_frame'):
                del self.loading_frame

    def show_results_page(self):
        """Display analysis results with visual indicators"""
        self.clear_frame()

        # Main container - increased size for better layout
        container = tk.Frame(self.root, bg=COLORS['bg_primary'], padx=20, pady=20)
        container.place(relx=0.5, rely=0.5, anchor='center', width=900, height=580)

        # Header
        header = tk.Label(container, text="Analysis Results", font=('Helvetica', 24, 'bold'),
                          fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        header.pack(pady=(0, 20))

        # Main content area with buttons on left and results on right
        main_frame = tk.Frame(container, bg=COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True)

        # Left side - Buttons panel
        btn_panel = tk.Frame(main_frame, bg=COLORS['bg_secondary'], padx=15, pady=15,
                             highlightbackground=COLORS['accent'], highlightthickness=1)
        btn_panel.pack(side='left', fill='y', padx=(0, 15))

        # Panel title
        panel_title = tk.Label(btn_panel, text="Actions", font=('Helvetica', 16, 'bold'),
                               fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
        panel_title.pack(pady=(0, 15))

        # Perform required actions button
        try:
            action_btn = CustomButton(btn_panel, text="Perform Actions", icon_url=ICONS['action'],
                                      command=self.perform_required_actions)
            action_btn.pack(pady=10, fill='x')
        except Exception as e:
            print(f"Error creating action button: {e}")
            action_btn = CustomButton(btn_panel, text="Perform Actions",
                                      command=self.perform_required_actions)
            action_btn.pack(pady=10, fill='x')

        # Export Results button
        try:
            export_btn = CustomButton(btn_panel, text="Export Results", icon_url=ICONS['export'],
                                      command=self.export_results)
            export_btn.pack(pady=10, fill='x')
        except Exception as e:
            print(f"Error creating export button: {e}")
            export_btn = CustomButton(btn_panel, text="Export Results",
                                      command=self.export_results)
            export_btn.pack(pady=10, fill='x')

        # New scan button
        new_scan_btn = CustomButton(btn_panel, text="New Scan", icon_url=ICONS['camera'],
                                    command=self.show_capture_page)
        new_scan_btn.pack(pady=10, fill='x')

        # Home button
        home_btn = CustomButton(btn_panel, text="Home", icon_url=ICONS['home'],
                                command=self.show_plant_selection_page)
        home_btn.pack(pady=10, fill='x')

        # Right side - Results content
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_primary'])
        content_frame.pack(side='right', fill='both', expand=True)

        # Left column - Disease Detection Results
        left_frame = tk.Frame(content_frame, bg=COLORS['bg_primary'], width=280)
        left_frame.pack(side='left', padx=5, fill='both', expand=True)

        disease_title = tk.Label(left_frame, text="Disease Detection Results", font=('Helvetica', 16, 'bold'),
                                 fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        disease_title.pack(pady=(0, 10))

        # Load and display the image used for analysis
        img = Image.open(self.bg_removed_image_path)
        img = img.resize((220, 220), Image.LANCZOS)
        self.result_image_tk = ImageTk.PhotoImage(img)

        img_frame = tk.Frame(left_frame, bg=COLORS['bg_secondary'], padx=10, pady=10,
                             highlightbackground=self.disease_results["status_color"], highlightthickness=2)
        img_frame.pack(pady=5)

        img_label = tk.Label(img_frame, image=self.result_image_tk, bg=COLORS['bg_secondary'])
        img_label.pack()

        # Status icon
        status_icon = load_image_from_url(self.disease_results["status_icon"], (40, 40))
        if status_icon:
            icon_label = tk.Label(left_frame, image=status_icon, bg=COLORS['bg_primary'])
            icon_label.image = status_icon
            icon_label.pack(pady=5)

        # Status
        status_frame = tk.Frame(left_frame, bg=self.disease_results["status_color"], padx=20, pady=5)
        status_frame.pack(fill='x', pady=5)

        status_label = tk.Label(status_frame, text=self.disease_results["health_status"],
                                font=('Helvetica', 16, 'bold'), fg=COLORS['text_secondary'],
                                bg=self.disease_results["status_color"])
        status_label.pack()

        # Condition
        condition_label = tk.Label(left_frame, text=f"Condition: {self.disease_results['predicted_class']}",
                                   font=('Helvetica', 12), fg=COLORS['text_primary'],
                                   bg=COLORS['bg_primary'])
        condition_label.pack(pady=2, anchor='w')

        # Confidence
        confidence_label = tk.Label(left_frame, text=f"Confidence: {self.disease_results['confidence']:.1f}%",
                                    font=('Helvetica', 12), fg=COLORS['text_primary'],
                                    bg=COLORS['bg_primary'])
        confidence_label.pack(pady=2, anchor='w')

        # Right column - Sensor Analysis Results
        right_frame = tk.Frame(content_frame, bg=COLORS['bg_primary'], width=350)
        right_frame.pack(side='right', padx=5, fill='both', expand=True)

        sensor_title = tk.Label(right_frame, text="Sensor Analysis Results", font=('Helvetica', 16, 'bold'),
                                fg=COLORS['text_primary'], bg=COLORS['bg_primary'])
        sensor_title.pack(pady=(0, 10))

        # Ideal values section
        ideal_frame = tk.Frame(right_frame, bg=COLORS['bg_secondary'], padx=15, pady=10)
        ideal_frame.pack(fill='x', pady=5)

        ideal_title = tk.Label(ideal_frame, text="Ideal Values:", font=('Helvetica', 12, 'bold'),
                               fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
        ideal_title.pack(anchor='w')

        for feature, value in self.sensor_results["ideal_values"].items():
            value_label = tk.Label(ideal_frame, text=f"{feature}: {value:.2f}", font=('Helvetica', 11),
                                   fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
            value_label.pack(anchor='w')

        # Adjustments section
        adjust_frame = tk.Frame(right_frame, bg=COLORS['bg_secondary'], padx=15, pady=10)
        adjust_frame.pack(fill='x', pady=5)

        adjust_title = tk.Label(adjust_frame, text="Required Adjustments:", font=('Helvetica', 12, 'bold'),
                                fg=COLORS['text_primary'], bg=COLORS['bg_secondary'])
        adjust_title.pack(anchor='w')

        # Ensure adjustments are displayed
        if self.sensor_results and "adjustments" in self.sensor_results:
            for feature, adj in self.sensor_results["adjustments"].items():
                sign = "Increase" if adj > 0 else "Decrease"
                value_label = tk.Label(adjust_frame, text=f"{feature}: {sign} by {abs(adj):.2f}",
                                       font=('Helvetica', 11), fg=COLORS['text_primary'],
                                       bg=COLORS['bg_secondary'])
                value_label.pack(anchor='w')
        else:
            # Fallback if adjustments are missing
            no_data = tk.Label(adjust_frame, text="No adjustment data available",
                               font=('Helvetica', 11), fg=COLORS['text_primary'],
                               bg=COLORS['bg_secondary'])
            no_data.pack(anchor='w')

    def export_results(self):
        """Export analysis results to a CSV file"""
        if not self.disease_results or not self.sensor_results:
            self.show_notification("No results to export.", "warning")
            return

        try:
            # Create timestamp for filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            export_path = os.path.join(BASE_DIR, f"plant_analysis_{timestamp}.csv")

            # Prepare data for export
            export_data = {
                "Plant Type": self.plant_choice,
                "Analysis Date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Health Status": self.disease_results["health_status"],
                "Condition": self.disease_results["predicted_class"],
                "Confidence": f"{self.disease_results['confidence']:.1f}%"
            }

            # Add sensor data
            for feature, value in self.sensor_results["current_values"].items():
                export_data[f"Current {feature}"] = f"{value:.2f}"

            for feature, value in self.sensor_results["ideal_values"].items():
                export_data[f"Ideal {feature}"] = f"{value:.2f}"

            # Write to CSV
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(export_data.keys())
                writer.writerow(export_data.values())

            self.show_notification(f"Results exported to {export_path}", "success")
        except Exception as e:
            print(f"Error exporting results: {e}")
            self.show_notification(f"Export failed: {str(e)}", "error")

    def perform_required_actions(self):
        """Send sensor data to Arduino for automated actions"""
        if not self.sensor_data:
            self.show_notification("No sensor data available.", "warning")
            return

        self.show_loading("Sending data to Arduino...")

        # Run in a separate thread to avoid freezing the UI
        threading.Thread(target=self._perform_actions_thread, daemon=True).start()

    def _perform_actions_thread(self):
        """Thread function for Arduino communication"""
        try:
            # Read the latest data from CSV file
            with open(SENSOR_DATA_PATH, mode='r') as file:
                reader = list(csv.DictReader(file))
                latest_entry = reader[-1]  # Get the most recent row

            if send_to_arduino(latest_entry):
                self.root.after(0, lambda: self.show_notification("Actions performed successfully!", "success"))
            else:
                self.root.after(0, lambda: self.show_notification("Could not reach the Arduino.", "error"))
            self.root.after(0, self.hide_loading)
        except Exception as e:
            print(f"Error performing actions: {e}")
            self.root.after(0, lambda: self.show_notification(f"Error: {str(e)}", "error"))
            self.root.after(0, self.hide_loading)

    def clear_frame(self):
        """Clear all widgets except background"""
        for widget in self.root.winfo_children():
            if widget is not self.bg_label and widget is not self.overlay:
                widget.destroy()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseDetectionApp(root)
    root.mainloop()
