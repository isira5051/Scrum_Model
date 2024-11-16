import streamlit as st
import requests

st.title('Agile Chatbot')
st.write('Ask me anything!')

# User input
user_input = st.text_input("Enter  query:")

# Handle query submission
if st.button("Submit"):
    if user_input:

        url = 'http://192.168.1.6:5000/query'
        response = requests.post(url, json={"query": user_input})
        if response.status_code == 200:
            response_data = response.json()
            st.write("Response:", response_data.get("response", "No response"))
        else:
            st.write("Error: Failed to get response .")
    else:
        st.write("enter a query.")
