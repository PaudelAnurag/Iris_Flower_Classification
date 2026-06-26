import streamlit as st
import pandas as pd
import joblib

# Load trained model

try:
    model = joblib.load("../artifacts/model/best_model.pkl") 
except:
    pass

try:
    model = joblib.load("artifacts/model/best_model.pkl")  
except:
    pass

# Species mapping
species_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

st.set_page_config(
    page_title="Iris Flower Classification",
    layout="centered"
)

st.title("Iris Flower Classification")

st.write("Enter the flower measurements below:")

# Input features
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

if st.button("Predict Species"):

    # Engineered features used during training
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

    if (prediction == 0):
        st.image("../artifacts/assets/iris-setosa.png", caption="Iris-setosa")
        
    elif (prediction == 1 ):
        st.image("../artifacts/assets/iris-versicolor.png", caption="Iris-versicolor")
        
    else:
        st.image("../artifacts/assets/iris-virginica.png", caption="Iris-virginica")
        
    st.success(f"Predicted Species: {predicted_species}")
    

    st.subheader("Input Features")

    st.dataframe(input_data)

    st.subheader("Engineered Features")

    st.write(f"Sepal Area = {sepal_area:.2f}")
    st.write(f"Petal Area = {petal_area:.2f}")