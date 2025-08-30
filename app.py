import os
import dotenv
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.environ.get("GOOGLE_API_KEY")
# Initialize embeddings
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

st.title("📰 News Research Tool")
st.sidebar.title("Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Initialize LLM
llm = init_chat_model(
    "gemini-2.5-flash", 
    model_provider="google_genai", 
    temperature=0.7, 
    max_output_tokens=500
)

# Step 1: Process URLs & create FAISS
if process_url and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("🔄 Loading data...")
    data = loader.load()

    main_placeholder.text("🔄 Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    main_placeholder.text("🔄 Building embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")
    main_placeholder.text("✅ Vector Store Saved!")

# Step 2: Ask a Question
query = st.text_input("Ask a question about the articles:")

if query:
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain.invoke({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.subheader(result.get("answer", "No answer found."))

        if "sources" in result:
            st.write("### Sources")
            st.write(result["sources"])
    else:
        st.error("❌ No FAISS index found. Please process URLs first.")
