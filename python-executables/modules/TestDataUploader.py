import streamlit as st
import h5py
import numpy as np
import torch
import os
from torchvision import transforms

class TestDataUploader:
    def __init__(self):

        #path hidden for privacy, in future use this can be improved to be dynamic path and not hard coded
        self.base_path = r"C:\Users\csial\Desktop\Thesis\Data\Dataset"

        #system takes as input the clean .h5 files from the patch cameloyn dataset
        self.choices = {
            "Test": ("camelyonpatch_level_2_split_test_x.h5", "camelyonpatch_level_2_split_test_y.h5"),
            "Train": ("camelyonpatch_level_2_split_train_x.h5", "camelyonpatch_level_2_split_train_y.h5"),
            "Validation": ("camelyonpatch_level_2_split_valid_x.h5", "camelyonpatch_level_2_split_valid_y.h5"),
        }

    def run(self):
        st.title("Upload Dataset")

        split = st.selectbox("Select dataset split:", list(self.choices.keys()))
        n_samples = st.slider("Select number of samples to load:", min_value=10, max_value=300000, step=10, value=100)

        if st.button("Load Data"):
            x_filename, y_filename = self.choices[split]
            x_path = os.path.join(self.base_path, x_filename)
            y_path = os.path.join(self.base_path, y_filename)

            #transoform the content of the h5 files into arrays
            try:
                with h5py.File(x_path, "r") as data_file, h5py.File(y_path, "r") as label_file:
                    total = len(data_file["x"])
                    st.info(f"Total available samples in **{split}** set: {total}")

                    labels_array = np.array(label_file["y"]).squeeze()
                    normal_indices = np.where(labels_array == 0)[0]
                    tumor_indices = np.where(labels_array == 1)[0]

                    n_per_class = n_samples // 2
                    n_normals = min(len(normal_indices), n_per_class)
                    n_tumors = min(len(tumor_indices), n_per_class)

                    #make sure that clean and tumor images are the same number for balanced result. can be modified to allow the user to select the persentage of tumor vs clean
                    if n_normals + n_tumors < n_samples:
                        remaining = n_samples - (n_normals + n_tumors)
                        if n_normals < n_per_class and len(tumor_indices) - n_tumors > 0:
                            extra = min(remaining, len(tumor_indices) - n_tumors)
                            n_tumors += extra
                        elif n_tumors < n_per_class and len(normal_indices) - n_normals > 0:
                            extra = min(remaining, len(normal_indices) - n_normals)
                            n_normals += extra

                    if n_normals == 0 or n_tumors == 0:
                        st.error("Not enough samples in one of the classes to create a balanced selection.")
                        return

                    #randomly select from images to avoid repeatition
                    chosen_normals = np.random.choice(normal_indices, size=n_normals, replace=False)
                    chosen_tumors = np.random.choice(tumor_indices, size=n_tumors, replace=False)

                    indices = np.concatenate([chosen_normals, chosen_tumors])
                    np.random.shuffle(indices) 

                    x = np.array(data_file["x"])[indices]
                    y = np.array(label_file["y"])[indices]

            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            x_tensor = torch.stack([transform(img) for img in x])
            y_tensor = torch.tensor(y, dtype=torch.long)

            st.session_state["test_images"] = x_tensor.numpy()
            st.session_state["test_labels"] = y_tensor.numpy()

            st.success(f"Loaded {len(indices)} balanced samples ({n_normals} normal, {n_tumors} tumor) from the **{split}** set.")

            self.preview_samples(x, y)

    #preview 9 images from the uploaded dataset
    def preview_samples(self, x, y):
        st.markdown("Preview Samples")
        cols = st.columns(3)
        for i in range(min(9, len(x))):
            with cols[i % 3]:
                #change the printed label for more user friendly experience, current label is [[1]]/[[0]]
                label = y[i][0] if isinstance(y[i], (list, np.ndarray)) else y[i]
                st.image(x[i], caption=f"Label: {label}", width=150)
