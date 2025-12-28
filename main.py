import streamlit as st
import joblib
import numpy as np

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

st.title("Spam email Classifier")
st.write("predict if the email is spam or not")

email_text = st.text_area( 
    label="input email",
    height=200
)


if st.button("Predict spam or not"):
    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        tfidf_text = vectorizer.transform([email_text])
        prediction = model.predict(tfidf_text)[0]
        probability = model.predict_proba(tfidf_text)[0][1]

        if prediction == 1:
            st.error(f"🚨 Spam Email (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {(1 - probability)*100:.2f}%)")


# Samples

# Congratulations! You have won a FREE cash prize of $1000.
# Click the link below to claim your reward now.
# This offer expires today.

# Hey, just checking in about the meeting tomorrow.
# Let me know if 11 AM still works for you.
# Thanks!

# Reminder: Your subscription will renew tomorrow.
# If you wish to cancel, visit your account settings.

# URGENT: Your account has been selected for a special offer.
# Verify your details immediately to avoid suspension.
# Click here now.





