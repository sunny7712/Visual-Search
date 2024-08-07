import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import torchvision.transforms as T
from similarity import find_similar_images
from transformers import AutoImageProcessor, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = AutoModel.from_pretrained("vit-base-fashion")
extractor = AutoImageProcessor.from_pretrained("vit-base-fashion")
model.eval()

transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

def pp(batch):
    images = batch["image"]
    image_batch_transformed = torch.stack(
        [transformation_chain(image) for image in images]
    )
    new_batch = {"pixel_values": image_batch_transformed}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        del image_batch_transformed
    return {"embeddings": embeddings}


ids = np.load("ids.npy")
img_dir = r"fashion-dataset\images"

st.title("Visual Search")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Find Similar Images'):
        # Get the embedding for the uploaded image
        img_processed = transformation_chain(image)
        img_processed = img_processed.unsqueeze(0)
        embedding = model(img_processed).last_hidden_state[:, 0].detach().cpu()

        # Find similar images
        distances, indices = find_similar_images(embedding, k = 6)
        indices = [ids[idx] for idx in indices] # file names
        imgs_path = [os.path.join(img_dir, name + ".jpg") for name in indices]
        # Display results
        st.subheader("Similar Images:")
        for i, path in enumerate(imgs_path):
            # st.write(f"Image {i+1}: Distance = {distance:.4f}")
            
            # Assuming your dataset has a way to get the image from an index
            similar_image = Image.open(path)  # Adjust this based on your dataset structure
            
            # Convert the image to bytes for display
            buf = BytesIO()
            similar_image.save(buf, format="PNG")
            st.image(buf.getvalue(), caption=f"Image {i+1}", use_column_width=True)

