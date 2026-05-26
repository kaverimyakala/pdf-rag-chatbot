from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =========================
# STEP 1 — LOAD EMBEDDINGS
# =========================

print("Loading Embedding Model...")

embeddings = OllamaEmbeddings(
    model="llama3"
)

# =========================
# STEP 2 — LOAD PDF
# =========================

print("Loading PDF...")

loader = PyPDFLoader("HR_Policies_Document.pdf")

documents = loader.load()

print(f"Loaded {len(documents)} pages")

# =========================
# STEP 3 — SPLIT DOCUMENT
# =========================

print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# =========================
# STEP 4 — CREATE VECTOR DB
# =========================

print("Creating Vector Database...")

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector DB Created Successfully")

# =========================
# STEP 5 — CREATE RETRIEVER
# =========================

retriever = vector_db.as_retriever()

# =========================
# STEP 6 — LOAD LLM
# =========================

print("Loading LLM...")

llm = OllamaLLM(
    model="llama3"
)

# =========================
# STEP 7 — CREATE PROMPT
# =========================

prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

Answer:
""")

# =========================
# STEP 8 — FORMAT DOCUMENTS
# =========================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# =========================
# STEP 9 — BUILD RAG CHAIN
# =========================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# STEP 10 — CHAT LOOP
# =========================

print("\nPDF RAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    query = input("Ask Question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    response = rag_chain.invoke(query)

    print("\nAnswer:")
    print(response)
    print("\n" + "="*50 + "\n")