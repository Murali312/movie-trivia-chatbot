import gradio as gr
from rag_backend import MovieBotBackend

# Initialize Backend
bot = MovieBotBackend(pdf_path="movie-trivia.pdf")

def chat_wrapper(user_input, history):
    # 1. Run Logic
    bot.generate_response(user_input)
    # 2. Return Dictionary Format (to match your environment's default)
    return "", bot.get_gradio_history()

def clear_wrapper():
    return bot.clear_memory()

with gr.Blocks() as demo:
    gr.Markdown("### Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!")
    
    # REMOVED the 'type' argument to fix the TypeError crash.
    # We are relying on your Gradio version defaulting to 'messages' (dictionaries).
    chatbot = gr.Chatbot(label="Chat History", height=400)
    msg = gr.Textbox(label="Your Question", placeholder="e.g., Who directed The Matrix?")
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat History")

    submit_btn.click(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_wrapper, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()