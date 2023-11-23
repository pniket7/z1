import openai
import streamlit as st
import pickle
import time
import numpy as np
import pandas as pd
from typing import Optional, Union
from gptcache import Cache  # Importing Cache from gptcache

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
    def __init__(self, gpt_name='GPT') -> None:
        self.messages = []
        self.history = []
        self.gpt_name = gpt_name
        self.gpt_cache = Cache(maxsize=1000)  # Creating an instance of Cache

    def chat(self, user_input: Optional[Union[dict, str]] = None, verbose=True, *args, **kwargs):
        completion_index = 0 if kwargs.get('logprobs', False) or kwargs.get('model') == 'text-davinci-003' else 1
        completion = self.completions[completion_index]
        user_input = self.__get_input(user_input=user_input, log=True)
        user_input = self.messages if completion_index else self.messages[-1]['content']
        kwargs.update({completion['prompt']: user_input, 'model': completion['model']})
        if completion_index == 1:
            kwargs.update({'temperature': 0.5})
        cache_key = str(kwargs)
        if cache_key in self.gpt_cache:
            reply = self.gpt_cache[cache_key]
        else:
            reply = self.__get_reply(completion=completion['completion'], log=True, *args, **kwargs)
            self.gpt_cache[cache_key] = reply
        self.history[-1].update({'completion_index': completion_index})
        if verbose:
            self.__call__(1)

    def display_probas(self, reply_index):
        history = self.history[reply_index]
        assert not history.completion_index
        probas = history.logprobs.top_logprobs
        return pd.concat([
                pd.DataFrame(data=np.concatenate([[list(k.keys()), np.exp2(list(k.values())).round(2)]]).T,
                             columns=[str(i), f'{i}_proba'],
                             index=[f'candidate_{j}' for j in range(len(probas[0]))]
                            ) for i, k in enumerate(probas)], axis=1).T

    def inject(self, line, role):
        self.__log(message={"role": role, "content": line})
    
    def clear(self, k=None):
        if k:
            self.messages = self.messages[:-k]
            self.history = self.history[:-k]
        else:
            self.__init__()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, filename):
        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages = temp.messages
            self.history = temp.history

    def merge(self, filename):
        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages += temp.messages
            self.history += temp.history

    def __get_input(self, user_input, log: bool = False):
        if user_input is None:
            user_input = input("> ")
        if not isinstance(user_input, dict):
            user_input = {"role": 'user', "content": user_input}
        if log:
            self.__log(user_input)
        return user_input

    @ErrorHandler
    def __get_reply(self, completion, log: bool = False, *args, **kwargs):
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

    def __call__(self, k: Optional[int] = None):
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
        line="You are a financial advisor at a bank. Start the conversation by inquiring about the user's financial goals. If the user mentions a specific financial goal or issue, acknowledge it and offer to help. Be attentive to the user's needs and goals. Be brief in your responses.",
        role="user"
    )
    advisor.inject(line="Ok.", role="assistant")
    return advisor

def main():
    st.title('Financial Advisor Chatbot')

    openai.api_key = st.secrets["api_key"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "sessionAdvisor" not in st.session_state or st.session_state.sessionAdvisor is None:
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()

    chat_container = st.empty()
    chat_messages = ""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            role_color = "#9400D3" if message["role"] == "user" else "#0084ff"
            alignment = "left" if message["role"] == "user" else "right"
            chat_messages += f'<div style="text-align: {alignment}; margin-bottom: 10px;"><span style="background-color: {role_color}; color: white; padding: 8px 12px; border-radius: 20px; display: inline-block; max-width: 70%;">{message["content"]}</span></div>'

    chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll;">{chat_messages}</div>', unsafe_allow_html=True)

    user_input = st.text_input("Type your message here...")

    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.sessionAdvisor.chat(user_input=user_input, verbose=False)
        advisor_response = st.session_state.sessionAdvisor.messages[-1]['content'] if st.session_state.sessionAdvisor.messages else ""
        advisor_response = advisor_response.replace('\n', ' ').strip()
        st.session_state.chat_history.append({"role": "bot", "content": advisor_response})
        chat_messages = ""
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role_color = "#9400D3" if message["role"] == "user" else "#0084ff"
                alignment = "left" if message["role"] == "user" else "right"
                chat_messages += f'<div style="text-align: {alignment}; margin-bottom: 10px;"><span style="background-color: {role_color}; color: white; padding: 8px 12px; border-radius: 20px; display: inline-block; max-width: 70%;">{message["content"]}</span></div>'
        
        chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll;">{chat_messages}</div>', unsafe_allow_html=True)

    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("New conversation started. You can now enter your query.")

    if st.button("Exit Chat"):
        st.session_state.chat_history = []
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("Chatbot session exited. You can start a new conversation by clicking the 'New Chat' button.")

if __name__ == "__main__":
    main()
