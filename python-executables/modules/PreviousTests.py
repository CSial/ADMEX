import os
from PIL import Image
import streamlit as st

class PreviousTests:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_images(self):
        images = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.folder_path, filename)
                image = Image.open(image_path)
                images.append(image)
        return images
