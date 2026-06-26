import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# Locate project root (works with both ../ and ./)
# =====================================================
root = None

for path in [Path(".."), Path(".")]:
    model_path = path / "artifacts" / "model" / "best_model.pkl"
    assets_path = path / "artifacts" / "assets"

    if model_path.exists() and assets_path.exists():
        root = path
        break

if root is None:
    raise FileNotFoundError("Could not locate the 'artifacts' directory.")

# =====================================================
# Load Model
# =====================================================
model = joblib.load(root / "artifacts" / "model" / "best_model.pkl")

# Assets directory
asset_dir = root / "artifacts" / "assets"

# =====================================================
# Species Mapping
# =====================================================
species_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

flower_images = {
    0: ("iris-setosa.png", "Iris-setosa"),
    1: ("iris-versicolor.png", "Iris-versicolor"),
    2: ("iris-virginica.png", "Iris-virginica")
}

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(
    page_title="Iris Flower Classification",
    layout="centered"
)

st.title(" Iris Flower Classification")

st.write("Enter the flower measurements below:")

# =====================================================
# User Inputs
# =====================================================
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2,
    step=0.1
)

# =====================================================
# Prediction
# =====================================================
if st.button("Predict Species"):

    # Feature Engineering
    sepal_area = sepal_length * sepal_width
    petal_area = petal_length * petal_width

    input_data = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width],
        "Sepal_Area": [sepal_area],
        "Petal_Area": [petal_area]
    })

    prediction = model.predict(input_data)[0]
    predicted_species = species_map[prediction]

    # Display flower image
    filename, caption = flower_images[prediction]
    st.image(asset_dir / filename, caption=caption, width=300)

    # Display prediction
    st.success(f"Predicted Species: **{predicted_species}**")

    # Display input features
    st.subheader("Input Features")
    st.dataframe(input_data)

    # Display engineered features
    st.subheader("Engineered Features")
    st.write(f"**Sepal Area:** {sepal_area:.2f}")
    st.write(f"**Petal Area:** {petal_area:.2f}")