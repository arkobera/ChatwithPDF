import os
import torch
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def get_text(docs):
    """Extracts text from uploaded PDFs."""
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" 
    return text


def get_chunks(text):
    """Splits text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def get_vectordb(chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectordb


def get_response(user_query, history_retriever):
    """Retrieves context from the vector database and generates a response using LLM."""
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question "
        "If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_retriever, docs_chain)
    return retrieval_chain


def get_conversation_chain(user_query, vectordb):
    """Retrieves relevant context and generates a response using RAG."""
    llm = ChatGroq(groq_api_key=groq_api_key, model="deepseek-r1-distill-llama-70b")
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, vectordb.as_retriever(), context_q_prompt
    )
    rag_chain = get_response(user_query, history_aware_retriever)
    return rag_chain


def get_image(response):
    """Retrieves the most relevant image based on the response text."""
    image_dir = r"C:\Users\91991\Desktop\Langchain\TUT\Groq\images"
    
    # Get image file names without extension
    images = os.listdir(image_dir)
    image_names = [os.path.splitext(image)[0] for image in images if image.endswith(".png")]

    # Load a text similarity model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Convert response and image names to embeddings
    response_embedding = model.encode(response, convert_to_tensor=True)
    image_embeddings = model.encode(image_names, convert_to_tensor=True)

    # Compute similarity and get top images
    similarities = util.pytorch_cos_sim(response_embedding, image_embeddings)
    top_k = min(1, len(image_names))  # Fetch up to 3 relevant images
    top_indices = torch.topk(similarities, k=top_k).indices[0].tolist()

    relevant_images = [os.path.join(image_dir, f"{image_names[i]}.png") for i in top_indices]
    return relevant_images


def get_chat_history(user_query, vectordb):
    """Handles user interaction, retrieves response, and displays images."""
    rag_chain = get_conversation_chain(user_query, vectordb)
    response_data = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
    
    # Store conversation in session state
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response_data["answer"]))

    # Display chat messages
    st.subheader("Conversation History")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.warning(f"User 🙋: {msg.content}")
        else:
            st.success(f"Bot 🤖: {msg.content}")

    # Get relevant images for the response and display them
    relevant_images = get_image(response_data["answer"])
    if relevant_images:
        st.subheader("Relevant Images")
        for img in relevant_images:
            st.image(img, caption=os.path.basename(img), use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    st.header("Chat with Multiple PDFs 📖")

    with st.sidebar:
        st.subheader("Upload Your PDFs")
        docs = st.file_uploader("Upload files and click 'Process'", accept_multiple_files=True, type=["pdf"])

        if st.button("Process"):
            with st.spinner("Processing..."):
                text = get_text(docs)
                st.success("✅ Text Extraction Completed!")
                chunks = get_chunks(text)
                st.success("✅ Text Chunking Completed!")
                vectordb = get_vectordb(chunks)
                st.session_state.vectordb = vectordb  
                st.session_state.chat_history = []  
                st.success("✅ Vector Database Created! 🚀 Chatbot Ready!")

    user_input = st.text_input("Ask a question:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        if "vectordb" in st.session_state and st.session_state.vectordb:
            get_chat_history(user_input, st.session_state.vectordb)
        else:
            st.warning("Please upload and process a PDF first.")


if __name__ == "__main__":
    main()
