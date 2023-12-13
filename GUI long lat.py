import os
import tkinter as tk
from PIL import Image, ImageTk
import random  # Import the random module for demonstration purposes

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")

        # Replace with the actual path to your image folder
        self.image_folder = r"C:\Users\zarak\OneDrive\Desktop\Test Images"
        self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.current_image_index = 0

        # Initialize coordinates with random values (for demonstration)
        self.longitude = random.uniform(-180, 180)
        self.latitude = random.uniform(-90, 90)

        self.load_image()
        self.create_widgets()

        # Periodically check for changes in the image folder (every 1000 milliseconds)
        self.root.after(1000, self.check_for_changes)

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

    def create_widgets(self):
        self.image_label = tk.Label(self.root, image=self.photo)
        self.image_label.pack(padx=10, pady=10)

        self.coordinates_label = tk.Label(self.root, text=f"Longitude: {self.longitude:.6f}, Latitude: {self.latitude:.6f}", font=("Helvetica", 12))
        self.coordinates_label.pack(pady=10)

        self.text_label = tk.Label(self.root, text="Damaged", font=("Helvetica", 16))
        self.text_label.pack(pady=10)

        self.prev_button = tk.Button(self.root, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = tk.Button(self.root, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

    def show_prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        self.load_image()
        self.update_coordinates_label()
        self.image_label.config(image=self.photo)

    def show_next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        self.load_image()
        self.update_coordinates_label()
        self.image_label.config(image=self.photo)

    def check_for_changes(self):
        # Check for changes in the image folder
        current_images = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if current_images != self.image_list:
            # Update the image list and reload the current image
            self.image_list = current_images
            self.load_image()
            self.update_coordinates_label()
            self.image_label.config(image=self.photo)

        # Schedule the next check after 1000 milliseconds
        self.root.after(1000, self.check_for_changes)

    def update_coordinates_label(self):
        # Update the coordinates label with random values (for demonstration)
        self.longitude = random.uniform(-180, 180)
        self.latitude = random.uniform(-90, 90)
        self.coordinates_label.config(text=f"Longitude: {self.longitude:.6f}, Latitude: {self.latitude:.6f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
