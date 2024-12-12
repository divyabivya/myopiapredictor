import streamlit as st
import joblib
import numpy as np



# navigation buttons
if st.button("go to home page"):
    # read and display the html home page
    with open("home.html", "r") as file:
        html_content = file.read()
    st.markdown(html_content, unsafe_allow_html=True)


# loads the trained model
classifier = joblib.load('classifier.pkl')

# defines function to make predictions using the model
def make_prediction(features):
    prediction = classifier.predict([features])  # Features is a list or array
    return prediction[0]  # The model returns an array, so return the first element


# Front-end interface for user input
html_code = """

    <h1 style="color: pink;">⟡Myopia Predicter⟡</h1>
    <h6 style="color: pink;">Using Machine Learning with a 90% Accuracy (for children)</h6>
</body>
"""
st.markdown(html_code, unsafe_allow_html=True)
# a little bit of HTML and CSS styling


# input fields for the three features: SPHEQ, AL, ACD
spheq = st.number_input('SPHEQ (Spherical Equivalent)', min_value=-10.0, max_value=10.0, step=0.1, value=0.0)
al = st.number_input('AL (Axial Length)', min_value=20.0, max_value=30.0, step=0.1, value=24.0)
acd = st.number_input('ACD (Anterior Chamber Depth)', min_value=2.0, max_value=5.0, step=0.1, value=3.0)

# button to make prediction
if st.button('Make Prediction'):
    # Combine the input features into an array
    features = np.array([spheq, al, acd])

    # Get the prediction from the model
    result = make_prediction(features)

    # display the result as 'Yes' or 'No' based on the binary prediction
    if result == 1:
        st.success('Prediction: Yes, you may be myopic within the next 5 years.')
    else:
        st.success('Prediction: No, you may not be myopic within the next 5 years.')

    