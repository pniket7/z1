import openai
from utils import ChatSession

def main():
    # Initialize the AdvisorGPT.
    sessionAdvisor = ChatSession(gpt_name='Advisor')
    
    # Instruct GPT to become a financial advisor.
    sessionAdvisor.inject(
        line="You are a financial advisor at a bank. Greet the user with this message: 'Hello! How can I assist you with your banking today? What are you trying to accomplish with your banking?' Start the conversation by inquiring about the user's financial goals. If the user mentions a specific financial goal or issue, acknowledge it and offer to help. Be attentive to the user's needs and goals. If the user doesn't mention specific goals initially, guide them to discuss their financial goals, including age, annual income, and risk appetite, but do not ask for these details right at the beginning of the conversation. Always prioritize answering the user's questions over gathering information. Do not recommend specific financial actions or portfolios until you have a clear understanding of the user's financial situation and goals. Always maintain a customer-focused approach. Say 'ok' if you understand.",
        role="user"
    )
    sessionAdvisor.inject(line="Ok.", role="assistant")

    # Start the conversation.
    user_input = ''
    sessionAdvisor.chat(user_input=user_input, verbose=False)
    print('Advisor:', sessionAdvisor.messages[-1].content)

    # Continue the conversation with a more flexible flow
    while True:
        user_input = input("User: ")  # Get user input
        sessionAdvisor.chat(user_input=user_input, verbose=False)
        assistant_response = sessionAdvisor.messages[-1].content
        print('Advisor:', assistant_response)

if __name__ == "__main__":
    # Load or set the OpenAI API key
    openai.api_key = "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"

    print('Connecting you to the financial advisor...')
    try:
        main()
    except Exception as e:
        print(f'Connection failed. Please start a new chat. Error: {e}')
