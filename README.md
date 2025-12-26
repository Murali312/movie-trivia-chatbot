# 🎬 MovieBot: Movie Trivia Chatbot

MovieBot is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about movies based on a custom PDF knowledge base. It uses **LangChain**, **Ollama** (local LLMs), **ChromaDB**, and **Gradio** for the user interface.

## 🚀 Features

* **RAG Pipeline**: Loads a movie trivia PDF, chunks text, and retrieves relevant context for accurate answers.
* **Local LLM Support**: Runs entirely locally using **Ollama** (Llama 3.2 for generation, Granite for embeddings).
* **Vector Search**: Uses **ChromaDB** for efficient similarity search.
* **Memory**: Tracks conversation history to handle follow-up questions.
* **Tracing**: Integrated with **LangSmith** for debugging and monitoring LLM traces.
* **User Interface**: Clean web interface built with **Gradio**.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.10+**
2.  **[Ollama](https://ollama.com/)** (Running in the background)
3.  **LangSmith API Key** (Optional, for tracing)

## ⚡ Quick Start Commands

Follow these commands in order to set up and run the bot on your machine.

### 1. Setup Environment
Open your terminal in the project folder and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (Mac/Linux)
# source venv/bin/activate
```

### 2. Download Ollama Models
You must pull the specific models used in the code before running the bot:

```bash
ollama pull llama3.2:3b
ollama pull granite-embedding:latest
```

### 3. Create Configuration File
Create a file named .env in the root folder and paste your keys:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=MovieTriviaBot
```

# 4. Run the Bot
Once everything is installed and Ollama is running in the background, run this command in your main terminal:

```bash
python app.py
```

### After running, open your browser and go to: 👉 http://127.0.0.1:7860 ###


# Install required libraries
pip install -r requirements.txt
