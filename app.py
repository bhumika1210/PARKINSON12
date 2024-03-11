import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from streamlit_extras.switch_page_button import switch_page
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid
import pickle
from streamlit_option_menu import option_menu
import joblib 


# Create a session state to store login status and username
if 'login_state' not in st.session_state:
    st.session_state.login_state = False
    st.session_state.username = ""

__login__obj = __login__(auth_token="courier_auth_token",
                         company_name="Shims",
                         width=200, height=250,
                         logout_button_name='Logout', hide_menu_bool=False,
                         hide_footer_bool=False,
                         lottie_url='https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

@st.cache_data
def build_login_ui():
    return __login__obj.build_login_ui()

LOGGED_IN = build_login_ui()

if LOGGED_IN:
   st.session_state.login_state = True
   st.session_state.username = __login__obj.get_username()


   # Load the pre-trained model and scaler
   model = joblib.load("voice/tools/model_joblib")
   scaler = joblib.load("voice/tools/scaler_joblib")
   drawing_model = load_model("spiral/keras_model.h5", compile=False)
   image_model = load_model("spiral/keras_model.h5", compile=False)

   # sidebar navigation
   with st.sidebar:
    
        selected = option_menu('Parkinson Detection', 
                           ['Voice Model',
                            'Spiral Model'],
                           icons=['mic','tornado'],
                           default_index=0)

   if (selected == 'Voice Model'):
    
      # Function to predict Parkinson's disease
      def predict_parkinsons(data):
         # Transform the input data using the pre-trained scaler
         scaled_data = scaler.transform(data)
         # Make predictions
         prediction = model.predict(scaled_data)
         return prediction[0]

      st.title("Parkinson's Disease Prediction")

      st.write("Enter the following medical record values:")

      # Create input fields for medical record parameters
      mdvp_fo = st.number_input("MDVP:Fo (Hz)")
      mdvp_fhi = st.number_input("MDVP:Fhi (Hz)")
      mdvp_flo = st.number_input("MDVP:Flo (Hz)")
      mdvp_jitter = st.number_input("MDVP:Jitter (%)")
      mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)")
      mdvp_rap = st.number_input("MDVP:RAP")
      mdvp_ppq = st.number_input("MDVP:PPQ")
      jitter_ddp = st.number_input("Jitter:DDP")
      mdvp_shimmer = st.number_input("MDVP:Shimmer")
      mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)")
      shimmer_apq3 = st.number_input("Shimmer:APQ3")
      shimmer_apq5 = st.number_input("Shimmer:APQ5")
      mdvp_apq = st.number_input("MDVP:APQ")
      shimmer_dda = st.number_input("Shimmer:DDA")
      nhr = st.number_input("NHR")
      hnr = st.number_input("HNR")
      rpde = st.number_input("RPDE")
      dfa = st.number_input("DFA")
      spread1 = st.number_input("spread1")
      spread2 = st.number_input("spread2")
      d2 = st.number_input("D2")
      ppe = st.number_input("PPE")

      # Create a button to predict
      if st.button("Predict"):
         # Create a DataFrame from the user input
         user_data = pd.DataFrame({
            "MDVP:Fo(Hz)": [mdvp_fo],
            "MDVP:Fhi(Hz)": [mdvp_fhi],
            "MDVP:Flo(Hz)": [mdvp_flo],
            "MDVP:Jitter(%)": [mdvp_jitter],
            "MDVP:Jitter(Abs)": [mdvp_jitter_abs],
            "MDVP:RAP": [mdvp_rap],
            "MDVP:PPQ": [mdvp_ppq],
            "Jitter:DDP": [jitter_ddp],
            "MDVP:Shimmer": [mdvp_shimmer],
            "MDVP:Shimmer(dB)": [mdvp_shimmer_db],
            "Shimmer:APQ3": [shimmer_apq3],
            "Shimmer:APQ5": [shimmer_apq5],
            "MDVP:APQ": [mdvp_apq],
            "Shimmer:DDA": [shimmer_dda],
            "NHR": [nhr],
            "HNR": [hnr],
            "RPDE": [rpde],
            "DFA": [dfa],
            "spread1": [spread1],
            "spread2": [spread2],
            "D2": [d2],
            "PPE": [ppe]
         })

         # Make a prediction
         prediction = predict_parkinsons(user_data)

         # Display the prediction result
         if prediction == 1:
               st.error("Based on the input data, the person is likely to have Parkinson's disease.")
         else:
               st.success("Based on the input data, the person is not likely to have Parkinson's disease.")

         st.write("Please enter the medical record values in the input fields above and click the 'Predict' button.")


   if (selected == 'Spiral Model'):
      # sidebar navigation
      with st.sidebar:
    
        selected = option_menu('Parkinson Detection - Spiral model', 
                           ['Dynamic Spiral Model',
                            'Visual Input Spiral Model'],
                           icons=['pen','camera'],
                           default_index=0)
    

      if (selected == 'Dynamic Spiral Model'): 
         st.header("Detecting Parkinson's Disease - Dynamic Spiral Model")
         with st.sidebar:
            # Specify canvas parameters in application
            drawing_mode = "freedraw"

            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke colour : ")
            bg_color = st.sidebar.color_picker("Background colour : ", "#eee")

            realtime_update = st.sidebar.checkbox("Update in realtime", True)

         # Split the layout into two columns
         col1, col2 = st.columns(2)

         # Define the canvas size
         canvas_size = 345

         with col1:
            # Create a canvas component
            st.subheader("Drawable Interface")
            canvas_image = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                width=canvas_size,
                height=canvas_size,
                update_streamlit=realtime_update,
                drawing_mode=drawing_mode,
                key="canvas",
            )

         with col2:
            st.subheader("Overview")
            if canvas_image.image_data is not None:
                # Get the numpy array (4-channel RGBA 100,100,4)
                input_numpy_array = np.array(canvas_image.image_data)
                # Get the RGBA PIL image
                input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
                st.image(input_image, use_column_width=True)

         def generate_user_input_filename():
            unique_id = uuid.uuid4().hex
            filename = f"user_input_{unique_id}.png"
            return filename

         def predict_parkinsons(img_path):
            best_model = load_model("spiral/keras_model.h5", compile=False)

            # Load the labels
            class_names = open("spiral/labels.txt", "r").readlines()

            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Get the numpy array (4-channel RGBA 100,100,4)
            input_numpy_array = np.array(img_path.image_data)

            # Get the RGBA PIL image
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")

            # Generate a unique filename for the user input
            user_input_filename = generate_user_input_filename()

            # Save the image with the generated filename
            input_image.save(user_input_filename)
            print("Image Saved!")   

            # Replace this with the path to your image
            image = Image.open(user_input_filename).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = best_model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
            os.remove(user_input_filename)
            print("Image Removed!")
            return Detection_Result, prediction

         submit = st.button(label="Submit Sketch")
         if submit:
            st.subheader("Output")
            classified_label, prediction = predict_parkinsons(canvas_image)
            with st.spinner(text="This may take a moment..."):
                st.write(classified_label)

                class_names = open("spiral/labels.txt", "r").readlines()

                data = {
                    "Class": class_names,
                    "Confidence Score": prediction[0],
                }

                df = pd.DataFrame(data)

                df["Confidence Score"] = df["Confidence Score"].apply(
                    lambda x: f"{str(np.round(x*100))[:-2]}%"
                )

                df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])

                st.subheader("Confidence Scores on other classes:")
                st.write(df)


            if (selected == 'Visual Input Spiral Model'): 

               # Add the second part of the script
               st.header("Detecting Parkinson's Disease - Visual Input Spiral Model")

               st.write("Upload an image to classify into Healthy or Parkinson's.")
               st.warning("Warning: Supported image formats: PNG, JPG, JPEG.")

               uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
               # Display the uploaded image
               st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

            # Process the image and make a prediction
            if st.button("Classify"):
                # Save the uploaded image temporarily
                user_input_filename = "user_input.png"
                with open(user_input_filename, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Load the trained model
                model = load_model("spiral/keras_Model.h5", compile=False)

                # Load the labels
                class_names = open("spiral/labels.txt", "r").readlines()

                # Create the array of the right shape to feed into the Keras model
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

                # Open the uploaded image
                image = Image.open(user_input_filename).convert("RGB")

                # Resize the image to be at least 224x224 and then crop from the center
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

                # Convert the image into a numpy array
                image_array = np.asarray(image)

                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

                # Load the image into the array
                data[0] = normalized_image_array

                # Make a prediction
                prediction = model.predict(data)
                confidence_score = prediction[0][0]  # Assuming 0 is the index for Parkinson's class

                # Display the result
                st.subheader("Classification Result:")
                if confidence_score >= 0.5:
                    st.write("The model has classified the image as Healthy.")
                else:
                    st.write("The model has classified the image as Parkinson's.")

                # Remove the temporary image file
                os.remove(user_input_filename)
