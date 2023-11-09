from dotenv import load_dotenv
import os
import openai
import streamlit as st
from utils import ChatSession

def main():
    st.title('Financial Bank Advisor Chatbot')

    # Access the OpenAI API key from Streamlit secrets
    api_key = st.secrets["openai_api_key"]
    openai.api_key = api_key  # Set the OpenAI API key

    # Initialize the AdvisorGPT.
    sessionAdvisor = ChatSession(gpt_name='Advisor')

    # Instruct GPT to become a financial advisor.
    sessionAdvisor.inject(
        line="You are a financial advisor at a bank. Greet the user with this message: 'Hello! How can I assist you with your banking today? What are you trying to accomplish with your banking?' Start the conversation by inquiring about the user's financial goals. If the user mentions a specific financial goal or issue, acknowledge it and offer to help. Be attentive to the user's needs and goals. If the user doesn't mention specific goals initially, guide them to discuss their financial goals, including age, annual income, and risk appetite, but do not ask for these details right at the beginning of the conversation. Always prioritize answering the user's questions over gathering information. Do not recommend specific financial actions or portfolios until you have a clear understanding of the user's financial situation and goals. Always maintain a customer-focused approach. Say 'ok' if you understand.",
        role="user"
    )
    sessionAdvisor.inject(line="Ok.", role="assistant")

    # Create a Streamlit text input for user input with a unique key
    user_input = st.text_input("User:", key="user_input")

    # Create a Streamlit button with a unique key
    if st.button("Send", key="send_button"):
        advisor_response = sessionAdvisor.chat(user_input=user_input, verbose=False)
        st.text(f'Advisor: {advisor_response}')

if __name__ == "__main__":
    main()
