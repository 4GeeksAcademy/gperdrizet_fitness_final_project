import pickle
import pandas as pd
import streamlit as st

st.markdown(
    """
    <style>
    /* 🔹 Background Gradient */
    .stApp {
        background: linear-gradient(to bottom, #0F2027, #203A43, #2C5364);
    }
    
    /* 🔹 Custom Styled Buttons */
    .stButton>button {
        background-color: #32CD32;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #228B22;
        transform: scale(1.05);
    }
    .stSuccess {
        font-size: 18px;
        font-weight: bold;
        color: #1E90FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("static/features-bg.jpg",  use_container_width=True)

# st.markdown(
#     """
#     <div style="position: relative; text-align: center;">
#         <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
#             font-size: 36px; color: white; background: rgba(0, 0, 0, 0.5); padding: 10px;">
#             Track Your Fitness Smarter 💪🔥
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

with open("../models/pipelines.pkl", "rb") as f:
    pipelines = pickle.load(f)

# print(pipelines.keys())  # Shows all available keys in the dictionary

# model = pipelines["calorie_model_pipeline"]

# print(type(model))  # Should output <class 'sklearn.pipeline.Pipeline'>

st.markdown("<h1 style='text-align: center; color: #FFD700;'>Fitness Prediction App 🏋️‍♂️♂️</h1>", unsafe_allow_html=True)
st.write("Enter details to predict fitness expenditure.")

model_option = st.selectbox("Select how many calories you want to burn today, or How much time you have available to workout:", ["Calories to Burn", "Workout Duration"])

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    height = st.number_input("Height (cm)", min_value=100, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200)

if model_option == "Calories to Burn": 
    model = pipelines["calorie_model_pipeline"]
    duration = st.number_input("Workout Duration (mins)", min_value=1, max_value=300)
    user_input = {"Age": age, "Gender": gender, "Height": height, "Weight": weight, "Duration": duration}
    target_feature = "Calories"

elif model_option == "Workout Duration": 
    model = pipelines["time_model_pipeline"]
    calories = st.number_input("Calories Burned", min_value=10, max_value=2000)
    user_input = {"Age": age, "Gender": gender, "Height": height, "Weight": weight, "Calories": calories}
    target_feature = "Duration"


input_data = pd.DataFrame([user_input])

# age = st.number_input("Age", min_value=1, max_value=100)
# gender = st.selectbox("Gender", ["Male", "Female"])
# height = st.number_input("Height (cm)", min_value=100, max_value=250)
# weight = st.number_input("Weight (kg)", min_value=20, max_value=200)
# duration = st.number_input("Workout Duration (mins)", min_value=1, max_value=300)

# input_data = pd.DataFrame({
#     "Age": [age],
#     "Gender": [gender],
#     "Height": [height],
#     "Weight": [weight],
#     "Duration": [duration]
# })

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Estimated {target_feature}: {prediction[0]:.2f}")