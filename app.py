import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from streamlit_extras.switch_page_button import switch_page
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageDraw
from PIL import ImageFont
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid
import pickle
from streamlit_option_menu import option_menu
from reportlab.lib.utils import ImageReader
import joblib 
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from datetime import datetime
import json

# Create a session state to store login status and username
if 'login_state' not in st.session_state:
    st.session_state.login_state = False
    st.session_state.username = ""

__login__obj = __login__(auth_token="courier_auth_token",
                         company_name="Parkin-Sight",
                         width=200, height=250,
                         logout_button_name='Logout', hide_menu_bool=False,
                         hide_footer_bool=False,
                         lottie_url='https://lottie.host/230a88d1-53f2-4c62-85db-28d5f4a92a66/Lv3agOWDP5.json')

def build_login_ui():
    return __login__obj.build_login_ui()

# def get_user_info():
#       if st.session_state.login_state:
#          st.write(f'**Username:** {st.session_state.username}')
         
      
#       # Add other user information retrieval code here
#       else:
#          st.warning("Please log in to view user information.")



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
      st.markdown(
            f'<h2 style="font-size:18px;margin-bottom: -10;"></h2>'
            f'<p style="font-size:28px; color:black;">Hi,{st.session_state.username}👋</p>', 
            unsafe_allow_html=True
        )
      
      
      selected = option_menu('Parkinson Detection', 
                               ['Spiral Model',
                                'Voice Model'],  # Add 'Data' option
                               icons=['tornado','mic'],  # Add icon for 'Data'
                               default_index=0)  
        
        
   

   # if (selected == 'Data'):
   #     st.write(f'**Username:** {st.session_state.username}')
   #     st.write(f'**Name:** {st.session_state.name}')
   #     st.write(f'**Email:** {st.session_state.username}')
 
      

   if (selected == 'Spiral Model'):
      # sidebar navigation
      with st.sidebar:
    
        selected = option_menu('Parkinson Detection - Spiral model', 
                           ['Dynamic Spiral Model',
                            'Visual Input Spiral Model'],
                           icons=['pen','camera'],
                           default_index=0)

      if (selected == 'Dynamic Spiral Model'): 
      
         # Function to generate a unique filename for user input
         def generate_user_input_filename():
            unique_id = uuid.uuid4().hex
            filename = f"user_input_{unique_id}.png"
            return filename
         def load_user_data(username):
            try:
               with open('_secret_auth_.json', 'r') as file:
                     data = json.load(file)
                     if isinstance(data, list):
                        for user in data:
                           if user.get('username') == username:
                                 return user
            except (FileNotFoundError, json.JSONDecodeError):
               return None
            
         # Function to generate PDF
         def generate_pdf(classified_label,user_info, prediction, image_data):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # Set font
            c.setFont("Helvetica", 12)

            # Draw background image containing logo and disclaimer
            background_image_path = "receipt3.jpg"  # Replace with the path to your background image
            c.drawImage(background_image_path, 0, 0, width=letter[0], height=letter[1])

            # Set text color to black
            c.setFillColorRGB(0, 0, 0)  
            if user_info:
               username = user_info.get('name', 'Unknown')
               email = user_info.get('email', 'Unknown')
               
               c.setFont("Helvetica", 15)
               c.setFillColorRGB(0, 0, 0)  
               # Draw user information before main report content
               c.drawString(100, 640, f"Name: {username}")
               c.drawString(100, 620, f"Email: {email}")
               current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               c.drawString(100, 600, f"Time: {current_time}")

               c.setFont("Helvetica-Bold", 16)
               c.setFillColorRGB(0.647, 0.165, 0.165)
               # Write text to PDF
               c.drawString(70, 540, "Parkinson's Disease Prediction Report - Dynamic Spiral Model")
         
               c.drawString(100, 230, "Spiral Drawing Classification:")
               c.drawString(100, 200, classified_label)
               
               c.setFillColorRGB(0, 0, 0)  
               c.drawString(120, 500, "Spiral Drawing: ")
               # Draw the spiral image on the PDF
               pil_image = Image.fromarray(image_data.astype("uint8"), "RGBA")
               c.drawImage(ImageReader(pil_image), 200, 280, width=200, height=200)
               
            else:
               c.drawString(100, 540, "User data not available")


            c.save()
            buffer.seek(0)
            return buffer


         # Function to predict Parkinson's disease
         def predict_parkinsons(image_data):
            # Load your model and labels here
            best_model = load_model("spiral/keras_model.h5", compile=False)
            class_names = {
                  0: "Healthy",
                  1: "Parkinson Diseased"
            }

            # Preprocess the image
            image = Image.fromarray(image_data.astype("uint8"), "RGBA")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_rgb = image.convert("RGB")  # Convert RGBA to RGB
            image_array = np.asarray(image_rgb)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make prediction
            prediction = best_model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Prepare result message
            detection_result = f"The model has detected the image as {class_name}"

            return detection_result, confidence_score
   

         # Main interface
         st.header("Detecting Parkinson's Disease - Dynamic Spiral Model")

         # Specify canvas parameters in application
         with st.sidebar:
            drawing_mode = "freedraw"
            stroke_width = st.slider("Stroke width: ", 1, 25, 3)
            stroke_color = st.color_picker("Stroke colour : ")
            bg_color = st.color_picker("Background colour : ", "#eee")
            realtime_update = st.checkbox("Update in realtime", True)

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

         # Submit button to predict
         submit = st.button(label="Submit Sketch")
         if submit:
            st.subheader("Output")
            with st.spinner(text="This may take a moment..."):
               # Get the image data from the CanvasResult object
               image_data = np.array(canvas_image.image_data)
               # Make prediction
               username = st.session_state.username

               # Load user data based on the retrieved username
               user_info = load_user_data(username)
               classified_label, prediction = predict_parkinsons(image_data)
               # Generate PDF
               pdf_buffer = generate_pdf(classified_label ,user_info,prediction, image_data)
               
               # Print prediction result
               st.write(f"Spiral Drawing Classification: {classified_label}")
               
               # Download PDF
               st.download_button(label="Download PDF", data=pdf_buffer, file_name="Dynamic_Spiral_report.pdf", mime="application/pdf", key="pdf-download")

      if (selected == 'Visual Input Spiral Model'): 

         # Function to generate a unique filename for user input
         def generate_user_input_filename():
            unique_id = uuid.uuid4().hex
            filename = f"user_input_{unique_id}.png"
            return filename
         def load_user_data(username):
            try:
               with open('_secret_auth_.json', 'r') as file:
                     data = json.load(file)
                     for user in data:
                        if user.get('username') == username:
                           return user
            except (FileNotFoundError, json.JSONDecodeError):
               return None

         # Function to generate PDF report
         def generate_pdf(classified_label, user_info, confidence_score, image_data):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)

            # Set font
            c.setFont("Helvetica", 12)

            # Draw background image containing logo and disclaimer
            background_image_path = "receipt3.jpg"  # Replace with the path to your background image
            c.drawImage(background_image_path, 0, 0, width=letter[0], height=letter[1])

            # Set text color to black
            c.setFillColorRGB(0, 0, 0)

            if user_info:

               username = user_info.get('name', 'Unknown')
               email = user_info.get('email', 'Unknown')
               
               c.setFont("Helvetica", 15)
               c.setFillColorRGB(0, 0, 0)  
               # Draw user information before main report content
               c.drawString(100, 640, f"Name: {username}")
               c.drawString(100, 620, f"Email: {email}")
               current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               c.drawString(100, 600, f"Time: {current_time}")

               c.setFont("Helvetica-Bold", 16)
               c.setFillColorRGB(0.647, 0.165, 0.165)
               # Write text to PDF
               c.drawString(70, 540, "Parkinson's Disease Prediction Report - Visual Input Spiral Model")
         
               c.drawString(100, 230, "Spiral Drawing Classification:")
               c.drawString(100, 200, f"The model has classified the image as {classified_label}")
               
               c.setFillColorRGB(0, 0, 0)  
               c.drawString(120, 500, "Spiral Drawing: ")
               
               # Draw the uploaded image on the PDF
               pil_image = Image.fromarray(image_data.astype("uint8"), "RGB")
               c.drawImage(ImageReader(pil_image), 200, 280, width=200, height=200)  # Adjusted y coordinate
            else:
               c.drawString(100, 550, "User data not available")

            c.save()
            buffer.seek(0)
            return buffer

         # Function to predict Parkinson's disease
         def predict_parkinsons(image_data):
            # Load your model and labels here
            model = load_model("spiral/keras_model.h5", compile=False)
            class_names = open("spiral/labels.txt", "r").readlines()

            # Create the array of the right shape to feed into the Keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Normalize the image
            normalized_image_array = (image_data.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Make prediction
            prediction = model.predict(data)
            confidence_score = prediction[0][0]  # Assuming 0 is the index for Parkinson's class

            # Determine classification label based on confidence score
            if confidence_score >= 0.5:
               classified_label = "Healthy"
            else:
               classified_label = "Parkinson diseased"

            return classified_label, confidence_score

         # Main interface
         def main():
            st.header("Detecting Parkinson's Disease - Visual Input Spiral Model")

            st.write("Upload an image to classify into Healthy or Parkinson's.")
            st.warning("Warning: Supported image formats: PNG, JPG, JPEG.")

            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

            # Process the image and make a prediction
            if st.button("Classify") and uploaded_file is not None:
               # Open the uploaded image
               image = Image.open(uploaded_file).convert("RGB")

               # Resize the image to be at least 224x224 and then crop from the center
               size = (224, 224)
               image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

               # Convert the image into a numpy array
               image_array = np.asarray(image)

               # Make prediction
               classified_label, confidence_score = predict_parkinsons(image_array)

               # Generate PDF with prediction and user info
               username = st.session_state.username

               # Load user data based on the retrieved username
               user_info = load_user_data(username) # Load user info based on username
               pdf_buffer = generate_pdf(classified_label, user_info, confidence_score, image_array)

               # Display the classification result
               st.subheader("Classification Result:")
               st.write(f"The model has classified the image as {classified_label}")

               # Download PDF report
               st.download_button(
                     label="Download PDF",
                     data=pdf_buffer,
                     file_name="Visual_Input_Spiral_report.pdf",
                     mime="application/pdf",
                     key="pdf-download"
               )

         if __name__ == "__main__":
            main()

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
         def load_user_data(username):
            try:
               with open('_secret_auth_.json', 'r') as file:
                     data = json.load(file)
                     if isinstance(data, list):
                        for user in data:
                           if user.get('username') == username:
                                 return user
            except (FileNotFoundError, json.JSONDecodeError):
               return None

         # Function to generate PDF report
         def generate_pdf(user_info, prediction):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)

            # Draw background image containing logo and disclaimer
            background_image_path = "receipt3.jpg"  # Replace with the path to your background image
            c.drawImage(background_image_path, 0, 0, width=letter[0], height=letter[1])

            # Set text color to black
            c.setFillColorRGB(0, 0, 0)

            # Retrieve user information
            if user_info:
               username = user_info.get('name', 'Unknown')
               email = user_info.get('email', 'Unknown')

               c.setFont("Helvetica", 15)
               c.setFillColorRGB(0, 0, 0)  
               # Draw user information before main report content
               c.drawString(100, 640, f"Name: {username}")
               c.drawString(100, 620, f"Email: {email}")
               # Display time of report generation after main report content
               current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               c.drawString(100, 600, f"Time: {current_time}")

               c.setFont("Helvetica-Bold", 15)
               c.setFillColorRGB(0.647, 0.165, 0.165)
               # Draw report content
               c.drawString(110, 500, "Parkinson's Disease Prediction Report - Voice Model")
    

               # Display prediction result
               if prediction == 1:
                  prediction_result = "likely to have Parkinson's disease."
               else:
                  prediction_result = "not likely to have Parkinson's disease."

               c.drawString(40, 420, "Prediction Result: ")
               c.drawString(40, 380, f"Based on the input data, the person is {prediction_result}")

               
            else:
               c.drawString(100, 550, "User data not available")

            # Save and return the PDF buffer
            c.save()
            buffer.seek(0)
            return buffer

         # Main Streamlit app code
         def main():
            
            # Check if user is logged in
            if 'username' not in st.session_state:
               st.warning("Please log in to view this page.")
               return

            # Retrieve authenticated user's information
            username = st.session_state.username
            user_info = load_user_data(username)

            # Generate PDF report for the authenticated user
            if user_info:
      
               prediction = 1  # Placeholder for prediction result (replace with actual prediction)
               pdf_buffer = generate_pdf(user_info, prediction)
               st.download_button(label="Download PDF", data=pdf_buffer, file_name="Voice_report.pdf", mime="application/pdf", key="pdf-download")
            else:
               st.warning("User data not found.")

         # Run the app
         if __name__ == "__main__":
            main()
