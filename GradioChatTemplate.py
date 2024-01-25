from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import os

#This template draws heavily on the Gradio Chatbot Template [1] and the linked Deeplearning.ai course [2]
# [1] https://www.gradio.app/guides/creating-a-chatbot-fast#a-streaming-example-using-openai
# [2] https://learn.deeplearning.ai/huggingface-gradio/lesson/6/chat-with-any-llm

#Get API key from .env file [1]; set model of ChatGPT to use; 3.5 used for price point [2]
#[1] Other options for securely storing API key at https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key
#[2] https://openai.com/pricing
load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]
model_id= "gpt-3.5-turbo"

# System message for model; explains the role of the chatbot assistant; a default included
sys_message  = """
You are a helpful assistant.
"""

# Gradio chat interface requires a function that takes in a message and history as args
# I am unsure about my implementation, tbh -James
def get_completions (message, history):
    
    #Add system message, retrieve previous chats from history, and add user message
    history_openai_format = [{"role": "system", "content": sys_message}]
    for turn in history:
        human, assistant = turn
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    #Start OpenAI chat session
    client = OpenAI(
        api_key=api_key
    )
    response =client.chat.completions.create(
        model=model_id,
        messages= history_openai_format,
        temperature=1.0 #Higher temperature will lead to more "creative" responses
    )
    return response.choices[0].message.content

# Launch chat interface; adjust placeholder text below as needed; add "share=True" as argument to launch() to get a shareable link
gr.ChatInterface(fn=get_completions, chatbot=gr.Chatbot(label="Chat Window"), textbox=gr.Textbox(placeholder="Instructions for User Submissions"), title="Insert Clever Chatbot Title Here", description="Insert Description of Chatbot", examples=["Sample Input 1", "Sample Input 2"]).launch()

