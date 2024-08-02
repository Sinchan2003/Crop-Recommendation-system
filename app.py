import streamlit as st
import numpy as np
import joblib
import pickle

model = joblib.load('model/model.joblib')
sc = pickle.load(open('model/standscaler.pkl','rb'))
ms = pickle.load(open('model/minmaxscaler.pkl','rb'))

st.set_page_config(layout="wide")
st.title("Crop Recommendation System")

n = st.slider("Nitrogen", 0, 140, 50, 1)
p = st.slider("Phosphorous", 5, 145, 53, 1)
k = st.slider("Potassium" ,5, 205, 48, 1 )
t = st.slider("Temperature" , 8, 45, 25, 1)
h = st.slider("Humidity", 14, 100, 72, 1)
pH = st.slider("pH", 3, 9, 6 ,1)
r = st.slider("Rainfall", 20, 300,100, 1)

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

if(st.button("Predict",type="primary")):
    features = np.array([n, p, k, t, h, pH, r]).reshape(1, -1)
    scaled_features = ms.transform(features)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)


    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    st.write(result)

st.sidebar.title("Crops recommended")
crops = [crop for crop in crop_dict.values()]
cropstring = ""
for crop in crops:
    cropstring += crop + "\n"

st.sidebar.text(cropstring)
    