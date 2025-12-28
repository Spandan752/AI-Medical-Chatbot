import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Medical Chatbot", page_icon=":robot_face:")
st.title("Medical Chatbot 🤖")

# -------Session State-------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------- Display Messages -------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------- User Input -------

if prompt := st.chat_input("Ask me a medical question:"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the user input to FastAPI backend
    with st.spinner("Thinking..."):
        response = requests.post(FASTAPI_URL, json={"input": prompt})
        answer = response.json()["response"]

    # Assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
