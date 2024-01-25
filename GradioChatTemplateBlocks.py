from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import os

#Get secrets from .env file

load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]
model_id = "gpt-3.5-turbo"

# System message to start conversations with model
# Matches system message in fine-tuned training set

sys_message  = """
You are a helpful assistant.
"""

#Properly format message history
def format_chat_prompt(message, chat_history, instruction):
    messages = [{"role": "system", "content": instruction}]
    for turn in chat_history:
        user_message, bot_message = turn
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})
    messages.append({"role": "user", "content": message})
    return messages

# Get the model response from current chat; add it to the chat history; return it and clear textbox
def respond(message, chat_history):
    formatted_messages = format_chat_prompt(message, chat_history, sys_message)
    client= OpenAI(
        api_key=api_key
    )
    response = client.chat.completions.create(messages=formatted_messages,
        model=model_id,
        temperature=1.0) # Higher temperature will lead to more "creative" responses
    bot_message = response.choices[0].message.content
    chat_history.append((message, bot_message))
    return "", chat_history

# Launch chat interface; adjust placeholder text below as needed; add "share=True" as argument to launch() to get a shareable link
with gr.Blocks() as demo:
    title = gr.Label("Insert Clever Chatbot Title Here", show_label=False)
    chatbot = gr.Chatbot(label="Chat Window")
    msg = gr.Textbox(label = "Enter text below", placeholder="Instructions for User Submissions")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.launch()

