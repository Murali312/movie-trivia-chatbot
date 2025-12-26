import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class MovieBotBackend:
    def __init__(self, pdf_path="movie-trivia.pdf"):
        self.pdf_path = pdf_path
        self.chat_history = InMemoryChatMessageHistory()
        self.rag_chain = self._initialize_pipeline()

    def _initialize_pipeline(self):
        if not os.path.exists(self.pdf_path):
            print(f"CRITICAL: {self.pdf_path} not found.")
            return None
            
        print(f"Loading {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print("Splitting text...")
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        splits = text_splitter.split_documents(documents)

        print("Initializing Chroma...")
        embeddings = OllamaEmbeddings(model="granite-embedding:latest")
        vectorstore = Chroma.from_documents(splits, embeddings, collection_name="historical_figures")

        print("Initializing Llama3.2...")
        llm = ChatOllama(model="llama3.2:3b")

        template = """Use the context to answer. Context: {context} Question: {question} Helpful Answer:"""
        custom_prompt = PromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever()

        def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

        return ({"context": retriever | format_docs, "question": RunnablePassthrough()} 
                | custom_prompt | llm | StrOutputParser())

    def generate_response(self, user_input):
        if not self.rag_chain: return "System Error: Knowledge base not loaded."
        self.chat_history.add_message(HumanMessage(content=user_input))
        try:
            response = self.rag_chain.invoke(user_input)
        except Exception as e:
            response = f"Error: {str(e)}"
        self.chat_history.add_message(AIMessage(content=response))
        return response

    def get_gradio_history(self):
        """
        Returns dictionaries: [{'role': 'user', 'content': '...'}, ...]
        This satisfies the 'Data incompatible' error you saw in the browser.
        """
        gradio_format = []
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                gradio_format.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gradio_format.append({"role": "assistant", "content": msg.content})
        return gradio_format

    def clear_memory(self):
        self.chat_history.clear()
        return []