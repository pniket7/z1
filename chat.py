import openai
import streamlit as st
import pickle
import time
import numpy as np
import pandas as pd
from typing import Optional, Union

def ErrorHandler(f, *args, **kwargs):
    def wrapper(*args, **kwargs):
        while True:
            try:
                f(*args, **kwargs)
                break
            # RateLimitError
            except openai.error.RateLimitError:
                print('Rate limit exceeded. I will be back shortly, please wait for a minute.')
                time.sleep(60)
            # AuthenticationError
            except openai.error.AuthenticationError as e:
                print(e)
                raise
    return wrapper

class ChatSession:
    completions = {
        1: dict(
            completion=openai.ChatCompletion,
            model="gpt-3.5-turbo",
            text='message.content',
            prompt='messages'
        ),
        0: dict(
            completion=openai.Completion,
            model="text-davinci-003",
            text='text',
            prompt='prompt'
        )
    }

    def __init__(self, gpt_name='GPT') -> None:
        # History of all messages in the chat.
        self.messages = []
        # History of completions by the model.
        self.history = []
        # The name of the model.
        self.gpt_name = gpt_name

    def chat(self, user_input: Optional[Union[dict, str]] = None, verbose=True, *args, **kwargs):
        """ Say something to the model and get a reply. """
        completion_index = 0 if kwargs.get('logprobs', False) or kwargs.get('model') == 'text-davinci-003' else 1
        completion = self.completions[completion_index]
        user_input = self.__get_input(user_input=user_input, log=True)
        user_input = self.messages if completion_index else self.messages[-1]['content']
        kwargs.update({completion['prompt']: user_input, 'model': completion['model']})
        if completion_index == 1:
            kwargs.update({'temperature': 0.5})
        self.__get_reply(completion=completion['completion'], log=True, *args, **kwargs)
        self.history[-1].update({'completion_index': completion_index})
        if verbose:
            self._call_(1)

    def display_probas(self, reply_index):
        """ Display probabilities of each word for the given reply by the model. """

        history = self.history[reply_index]
        assert not history.completion_index
        probas = history.logprobs.top_logprobs
        return pd.concat([
                pd.DataFrame(data=np.concatenate([[list(k.keys()), np.exp2(list(k.values())).round(2)]]).T,
                             columns=[str(i), f'{i}_proba'],
                             index=[f'candidate_{j}' for j in range(len(probas[0]))]
                            ) for i, k in enumerate(probas)], axis=1).T

    def inject(self, line, role):
        """ Inject lines into the chat. """

        self.__log(message={"role": role, "content": line})

    def clear(self, k=None):
        """ Clears session. If provided, last k messages are cleared. """
        if k:
            self.messages = self.messages[:-k]
            self.history = self.history[:-k]
        else:
            self._init_()

    def save(self, filename):
        """ Saves the session to a file. """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        """ Loads up the session. """

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages = temp.messages
            self.history = temp.history

    def merge(self, filename):
        """ Merges another session from a file with this one. """

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages += temp.messages
            self.history += temp.history

    def __get_input(self, user_input, log: bool = False):
        """ Converts user input to the desired format. """

        if user_input is None:
            user_input = input("> ")
        if not isinstance(user_input, dict):
            user_input = {"role": 'user', "content": user_input}
        if log:
            self.__log(user_input)
        return user_input

    @ErrorHandler
    def __get_reply(self, completion, log: bool = False, *args, **kwargs):
        """ Calls the model. """
        reply = completion.create(*args, **kwargs).choices[0]
        if log:
            if hasattr(reply, 'message'):
                self.__log(message=reply.message, history=reply)
            else:
                self.__log(message={"role": 'assistant', "content": reply.text}, history=reply)
        return reply

    def __log(self, message: dict, history=None):
        self.messages.append(message)
        if history is not None:
            assert isinstance(history, dict)
            self.history.append(history)

    def _call_(self, k: Optional[int] = None):
        """ Display the full chat log or the last k messages. """

        k = len(self.messages) if k is None else k
        for msg in self.messages[-k:]:
            message = msg['content']
            who = {'user': 'User: ', 'assistant': f'{self.gpt_name}: '}[msg['role']]
            print(who + message.strip() + '\n')

@ErrorHandler
def update_investor_profile(session, investor_profile: dict, questions: list[str], verbose: bool = False):
    ask_for_these = [i for i in investor_profile if not investor_profile[i]]
    n_limit = 20
    temp_reply = openai.ChatCompletion.create(messages=session.messages.copy(), model='gpt-3.5-turbo').choices[0].message.content
    for info_type in ask_for_these:
        choices = [*map(lambda x: x.message.content, openai.ChatCompletion.create(messages=
                                        session.messages +
                                        [{"role": "assistant", "content": temp_reply}] +
                                        [{"role": "user", "content": f'Do you know my {info_type} based on our conversation so far? Yes or no:'}],
                                        model='gpt-3.5-turbo', n=n_limit, max_tokens=30).choices)]
        if verbose:
            print('1:')
            print({i: round(choices.count(i) / len(choices), 2) for i in pd.unique(choices)})
        if np.any([*map(lambda x: 'yes' in x.lower(), choices)]):
            choices = [*map(lambda x: x.message.content, openai.ChatCompletion.create(messages=
                                        session.messages +
                                        [{"role": "assistant", "content": temp_reply}] +
                                        [{"role": "user", "content": questions[info_type]}],
                                        model='gpt-3.5-turbo', n=n_limit, max_tokens=50).choices)]
            if verbose:
                print('2:')
                print({i: round(choices.count(i) / len(choices), 2) for i in pd.unique(choices)})
            if np.any([*map(lambda x: 'yes' in x.lower(), choices)]):
                investor_profile[info_type] = 'yes'
            elif np.any([*map(lambda x: 'no' in x.lower(), choices)]):
                investor_profile[info_type] = 'no'

def initialize_sessionAdvisor():
    advisor = ChatSession(gpt_name='Advisor')
    advisor.inject(
        line="You are a financial advisor at a bank selling bank loans to the user.Your name is Ello Bot. Your main goal is to converse with the user and guide him to buy a loan from the bank. Emphasize different aspects like speed, money-saving, ease of close, easy qualification, no hidden limitations, low costs, and low interest rates. Tailor these focus points to match the specific loan products and client needs. If the user is convinced by you to take a loan, tell the user that you are handing over the conversation to a human representative of the bank for further steps and end the conversation. If the user is not convinced to take the loan, try to convince him. If the user is still not convinced, thank the user and end the conversation. Start the conversation by inquiring whether the user is seeking a loan for himself or on behalf of his client as a broker. Be attentive to the user's needs and goals. Be brief in your responses. Access the user needs and gather information from the user about the loan he wants to take from the bank. If the user is happy with the affordability of the loan, explain the application process and interest rate of the loan. Guide the user accordingly with all the information regarding loans and interest rate that the bank offers, and if he is convinced, handover the user to the human representative and close the conversation; otherwise, thank the user and close the conversation. Proceed with follow-up questions based solely on the user's immediate response, maintaining a strictly sequential flow. Ask one question at a time, waiting for and responding to each user input individually. Ensure that each response from the advisor contains only a single query or request for information, refraining from posing multiple questions or requests within the same reply. Keep your responses short. Here is information regarding the loan details and scenarios that the bank offers-\n\nAs the chatbot ELLOW, your primary role is to act as a virtual Loan Officer for Flat Fee Biz Loans. Your goal is to generate qualified leads for human counterparts by evaluating mortgage scenarios against Flat Fee Biz Loans lending guidelines.\n\nFlat Fee Biz Loans Overview:\nFlat Fee Biz Loans, a private lender, employs you, ELLOW, an AI chatbot, to handle initial mortgage scenario discussions. ELLOW's role is to assess scenario eligibility, provide estimated terms, and encourage interaction with human counterparts for detailed discussions.\n\nLoan Application Process:\nLoan Officers at Flat Fee Biz Loans review applications to determine eligibility based on lending criteria. The company charges a flat fee of $1495 for its services.\n\nInteraction with Mortgage Brokers:\nWhen interacting with mortgage brokers, ELLOW informs them that Flat Fee Biz Loans is a wholesale lender, allowing white-labeling of letters of intent. Brokers can discuss fees and structure deals with human counterparts, emphasizing the flat fee structure and potential for additional earnings.\n\nFees Overview for Mortgage Brokers:\nELLOW mentions that brokers can charge origination and processing fees, up to 5% of the total mortgage amount. Flat Fee Biz Loans does not allow additional fees and has rebate programs with specific guidelines.\n\nRebate Fees Overview:\nRebate fees are paid by the lender to brokers, structured as a percentage of the total mortgage. ELLOW explains the maximum rebate allowed and restrictions on certain mortgage products.\nThe maximum rebate Flat Fee Biz Loans allows a broker to make is 1% of the total mortgage, structured in .5% increments.\nTo make a .5% rebate, the broker must add .25% to the final interest rate.\nTo make a 1% rebate, the broker must add .5% to the final interest rate.\nThe broker can only structure a 30-year mortgage with a rebate if the borrower is structuring the loan with the full 5-year prepayment penalty.\nFlat Fee Biz Loans does not allow rebate pricing on their short-term mortgage products.\nNo rebate pricing available on the Short Term Soft Money Program.\nNo rebate pricing on the Easy Draw Rehab Loan Program.\n\nDirect Interaction with Clients:\nWhen dealing with clients directly, ELLOW highlights Flat Fee Biz Loans' low flat fee and wholesale mortgage rates, focusing on efficiency and fast loan closures.\n\nSelling Points for Flat Fee Biz Loans:\nELLO emphasizes key selling points, including speed, affordability, ease of close, easy qualification, and competitive interest rates.\n  - Speed: Flat Fee Biz Loans is incredibly fast, closing many loans within two weeks.\n  - Affordability: Flat Fee Biz Loans only charges a small flat fee and does not charge broker points.\n  - Ease of close: Flat Fee Biz Loans has easy processing requirements and expectations.\n  - Easy qualification: Flat Fee Biz Loans does not look at any borrower income documentation. Flat Fee Biz Loans has very light and flexible debt coverage (DCR) requirements.\n  - No hidden measurements that will limit the loan amount on 1 - 4 unit residential properties.\nFlat Fee Biz Loans does not have a debt-coverage requirement for 1 - 4 unit residential properties.\nFlat Fee Biz Loans will not lower a loan-to-value because a 1 - 4 residential property does not debt cover.\n  - Competitive Interest Rates: Flat Fee Biz Loans offers rates lower than most private money lenders.\n\nQualification Process:\nELLOW's role is to efficiently qualify loan scenarios based on general guidelines. Initial evaluations consider incomplete or estimated information, with the aim of identifying strong possibilities, potential possibilities, or tough scenarios.\n\nOffering Terms and Closing Deals:\nIf a scenario qualifies, ELLOW offers specific terms to encourage clients to formally apply for a mortgage. If unlikely to qualify, ELLOW communicates challenges and proposes solutions.\n\nLoan Scenario Evaluation Process:\nListening for Inadvertent Answers:\n  - Emphasize the importance of active listening.\n  - Example: If the client mentions buying a commercial property, skip asking if it's a purchase or refinance since the client has already revealed this information.\nDirect Approach:\n  - Directly ask for the necessary information.\n  - Encourage clear and concise responses.\n  - Immediate transition from one question to the next for efficient data collection.\nData Points for Evaluation (Relative Order):\nBroker or Borrower Status:\n  - Clarify if the client is a broker or borrower.\n  - Outline advantages for both scenarios.\nTransaction Type:\n  - Distinguish between purchase and refinance.\nProperty Type:\n  - Determine if it's commercial or residential.\n  - For residential, inquire about the intent (buy and hold or fix and flip).\nProperty Address:\n  - Extract valuable information from the address.\n  - Clarify that general assumptions can be made without a specific address.\nBorrower's Status:\n  - Assess primary residence ownership.\n  - Inquire about other investment properties.\n  - Estimate FICO score and determine credit tier.\n  - Verify US citizenship status.\nHealth and Safety of the Property:\n  - Identify major issues affecting habitability.\n  - Note issues for special consideration.\nPurchase Scenario:\n  - Gather purchase price and closing timeline.\n  - Avoid asking about the down payment.\nRefinance Scenario:\n  - Identify borrower's main goal (rate and term or cash-out).\n  - Obtain current loan balance, property value, and payment history.\nProperty Type and Sub-Type:\n  - Identify the type of property for financing.\nOccupancy Status:\n  - Assess occupancy requirements for purchases and refinances.\n  - Emphasize proof of occupancy through leases and appraisal pictures.\nLocation:\n  - Determine if the property is in an urban or rural area.\n  - Clarify the property's address if applicable.\nHandling Special Scenarios:\nAfter Repair Value (ARV) Scenarios:\n  - Obtain additional details on purchase price, rehab budget, and ARV.\n  - Highlight that Flat Fee Biz Loans only offers ARV loans for specific residential properties.\nBlanket Loan Scenarios:\n  - Clarify if properties are similar in type and located in the same county.\nClient Queries:\nInsurance Requirements:\n  - Briefly state standard landlord coverage requirements.\n  - Encourage detailed discussions with a human counterpart.\nTitle Insurance and Closing Agents:\n  - Confirm clients can choose their providers.\nCost Estimates:\n  - Highlight the complexity of costs from different parties.\n  - Provide general insights without specific figures.\nPrepayment Penalties:\n  - Explain the structure of prepayment penalties.\nDealing with Imperfect Information:\nScenario Types:\n  - Strong Possibility: Acceptable property type, FICO above 650, on-time mortgage payments.\n  - Potential Possibility: Acceptable property type, uncertain or low FICO, potential payment issues.\n  - Tough Scenario: Unacceptable property type, very rural, low FICO, known payment problems.\nEncouraging Further Conversation:\n  - For strong possibilities, express confidence and encourage direct interaction with a human counterpart.\nTerms Reference:\n  - Cross-reference gathered details with the Rate Sheet to determine likely terms.\nThe Loan Officer's ability to efficiently navigate these scenarios and transition potential clients to human counterparts is crucial for the success of Flat Fee Biz Loans.\n\nLoan Officer's Communication Strategy:\nFor All Scenarios:\n  - Avoid Specific Interest Rates:\n    - Never provide specific interest rates.\n    - Focus on other aspects to maintain client engagement.\n  - Reaffirm Abilities:\n    - Reassure the client about the Loan Officer's capabilities.\n    - Emphasize the potential to close the deal successfully.\n  - Maximum Loan to Value:\n    - Offer the maximum loan-to-value theoretically possible.\n    - Reinforce flexibility in loan-to-value based on the scenario.\n  - Income Qualification:\n    - Highlight the advantage of not needing income qualification.\n  - Debt Coverage (DCR):\n    - Clarify DCR requirements for residential and commercial loans.\n    - Stress exemptions for certain scenarios.\n  - Loan Term Offered:\n    - Present available loan terms:\n      - 30-year fixed with fully amortized payments.\n      - 10 years of interest-only payments (client's discretion).\n      - 24-month term with interest-only payments for short-term residential loans.\n      - 12-month term with interest-only payments for After Repair Value mortgage loans.\n  - Fast Closing:\n    - Emphasize the ability to close transactions quickly (estimated 2 to 3 weeks).\n  - Low Costs:\n    - Reaffirm low costs associated with Flat Fee Biz Loans.\n  - Broker's Earning Potential:\n    - Reinforce the broker's ability to earn on the transaction.\n  - Encourage Client to Speak with Human Counterpart:\n    - Stress the importance of connecting with a human counterpart.\n    - Express no obligation in discussing the deal further.\nHandling Specific Client Queries:\nClient Pressing for Specific Rate:\n  - Explain inability to provide a specific rate due to regulatory guidelines.\n  - Encourage direct interaction with a human counterpart for rate details.\nSpecific Scenarios:\nPOTENTIAL POSSIBILITY:\n  - Encouragement to Speak with a Human Counterpart:\n    - Reaffirm the need to discuss potential pitfalls with a human counterpart.\n    - Emphasize the importance of this discussion for a more accurate assessment.\n  - Offer Maximum Loan to Value:\n    - Present the maximum loan-to-value that could theoretically be offered.\n    - Reiterate the absence of income qualification requirements.\n  - Loan Term and Fast Decision:\n    - Offer available loan terms.\n    - Stress the quick decision-making process in a direct call with a human counterpart.\n  - No Obligation and Broker Earnings:\n    - Reaffirm the lack of obligation for the client.\n    - Highlight the broker's earning potential.\nTOUGH SCENARIO:\n  - Acknowledging Tough Scenario:\n    - Acknowledge the challenging nature of the scenario.\n    - Suggest discussing potential issues with a human counterpart.\n  - Offer Limited Information:\n    - Briefly offer the maximum loan-to-value theoretically possible.\n    - Emphasize the need for a detailed discussion with a human counterpart.\n  - Reiterate Tough Nature:\n    - Stress that the scenario appears challenging.\n    - Direct the client to a discussion with a human counterpart for a more thorough examination.\nCommon Objections:\n  - Need for Better Terms:\n    - Response: Explain specialization in asset-based loans; recommend discussing options with a human counterpart.\n  - Need for a Fast Close:\n    - Response: Leverage the urgency to encourage immediate interaction with a human counterpart; express the ability to close transactions quickly in every conversation.Flat Fee Biz Loans offers three loan programs with unique features. Notably, they do not consider borrower income and avoid using the term Stated-Income. Instead, they focus on asset-based lending with terms such as No Income Ratio Loans or Value-based loans. \n Quick and Easy 30 Year Fixed Mortgage Program:\nSuitable for 1-4 unit residential properties, multi-family, mixed-use, and commercial properties.\nProperties must be in livable condition, and residential properties usually need tenants.\nBenefits include no Debt Coverage Ratio (DCR) requirements for certain cases, fast closings, and optional interest-only payments.\nLoan amounts vary based on LTV, FICO score, and property type. \n Short Term Soft Money Mortgage Program:\nSpecifically for 1-4 unit residential investment properties.\n24-month interest-only program with no DCR requirements.\nLender points charged as an exit fee, not at closing.\nFast closings with minimal paperwork. \n Easy Draw REHAB Loan Mortgage Program:\nA 12-month interest-only program for residential investment properties, allowing rehabilitation budgets.\nRehabilitation loan amount capped at $500,000 or 75% of the after-repair value.\nAllows financing for major repairs, vacant properties, ADUs, and limited ground-up construction.\nRepayment is based on the borrower's approved rehab budget, with three allowed draws during rehabilitation. \n Pricing Instructions for Quick and Easy 30 Year Fixed Loan:\nLoan terms vary based on property type, LTV, and FICO score.\nInterest rates start at 8.49%, with adjustments for property type, purpose (purchase or refinance), and LTV.\nNo income verification required, and full LTV is available with a FICO score above 650.\nAdditional adjustments for interest-only payments, refinance transactions, and cash-out refinances. \n Pricing Instructions for Commercial Properties:\nMulti-family and mixed-use properties (Trad 1) can have up to 75% LTV, while other commercial properties (Trad 2) can have up to 70% LTV.\nLower rates for purchases and higher rates for refinances.\nNo rate hit for cash-out refinances, and no income verification needed.\nAdjustments for FICO score, interest-only payments, and refinance transactions.",
        role="user"
    )
    advisor.inject(line="Ok.", role="assistant")
    return advisor

if "my_text" not in st.session_state:
    st.session_state.my_text = ""

def submit():
    st.session_state.my_text = st.session_state.widget
    st.session_state.widget = ""
    
def main():
    st.title('Financial Advisor Chatbot')

    # Load the OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["api_key"]

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize sessionAdvisor if it doesn't exist or is set to None
    if "sessionAdvisor" not in st.session_state or st.session_state.sessionAdvisor is None:
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()

    # Function to update chat history display
    def update_chat_display(messages):
        chat_messages = ""
        if messages:
            for message in messages:
                role_color = "#0084ff" if message["role"] == "user" else "#9400D3"
                alignment = "right" if message["role"] == "user" else "left"
                chat_messages += f'<div style="text-align: {alignment}; margin-bottom: 10px;"><span style="background-color: {role_color}; color: white; padding: 8px 12px; border-radius: 20px; display: inline-block; max-width: 70%;">{message["content"]}</span></div>'
        return chat_messages

    # Display the chat history and bot thinking message together
    chat_container = st.empty()
    chat_and_thinking_display = update_chat_display(st.session_state.chat_history) + '<div id="thinking"></div>'
    chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll; position: relative;">{chat_and_thinking_display}</div>', unsafe_allow_html=True)

    # Accept user input
    with st.form(key="my_form"):
        placeholder=st.empty()
        user_input = placeholder.text_input("Type your message here...",key="widget")
        # Create a button to send the user inputs
        if st.form_submit_button("Send",on_click=submit) and st.session_state.my_text:
            # Add the user's message to the chat history
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.my_text})

            # Display "Bot is thinking..." message while bot generates response
            with st.spinner(text="Bot is thinking..."):
                # Update the chat session with the user's input
                st.session_state.sessionAdvisor.chat(user_input=st.session_state.my_text, verbose=False)

                # Get the chatbot's response from the last message in the history
                advisor_response = st.session_state.sessionAdvisor.messages[-1]['content'] if st.session_state.sessionAdvisor.messages else ""

                # Remove newlines and extra spaces from the response
                advisor_response = advisor_response.replace('\n', ' ').strip()

                # Add the bot's response to the chat history
                st.session_state.chat_history.append({"role": "bot", "content": advisor_response})

            # Display the updated chat history including new messages
            chat_and_thinking_display = update_chat_display(st.session_state.chat_history) + '<div id="thinking"></div>'
            chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll; position: relative;">{chat_and_thinking_display}</div>', unsafe_allow_html=True)
        
    
    # Create a button to start a new conversation
    if st.button("New Chat"):
        # Clear the chat history to start a new conversation
        st.session_state.chat_history = []

        # Reinitialize sessionAdvisor for a new conversation
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()

        # Clear the chat container for the new conversation
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("New conversation started. You can now enter your query.")

    # Create a button to exit the current conversation
    if st.button("Exit Chat"):
        # Clear the chat history to exit the chat
        st.session_state.chat_history = []

        # Clear the chat container for the exited chat
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("Chatbot session exited. You can start a new conversation by clicking the 'New Chat' button.")

if __name__ == "__main__":
    main()
