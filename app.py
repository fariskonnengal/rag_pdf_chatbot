import os
from langchain_community.document_loaders import PyPDFLoader

DATA_FOLDER = "data"

all_documents = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".pdf"):
        file_path = os.path.join(DATA_FOLDER, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

print(f"Total pages loaded: {len(all_documents)}")

# Debug: show first document
print("Sample document:")
print(all_documents[0].page_content)
print('Metadata:',all_documents[0].metadata)

#Document Structure 
# Document(
#   page_content="text...",
#   metadata={
#     "source": "data/ai_basics.pdf",
#     "page": 0
#   }
# )





from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# ---- Chunking ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")

# Debug: inspect one chunk
print("Sample chunk text:")
print(chunks[0].page_content)
print("Sample chunk metadata:")
print(chunks[0].metadata)

# ---- Embeddings ----
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



from langchain_community.vectorstores import FAISS
# ---- Vector Store ----
vectorstore=FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("FAISS vector store created successfully!")


query = "What is machine learning?"
docs = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Text:", doc.page_content)
    print("Source:", doc.metadata)


def retrieve_context(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs




from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

qa_pipeline=pipeline(
    "text2text-generation",
    model='google/flan-t5-base',
    max_length=256,
    device=-1
)
llm=HuggingFacePipeline(pipeline=qa_pipeline)


from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate(
    input_variables=['context','question'],
    template="""
You are an AI assistent.Answer the question ONLY using the context below.
if the answer is not present,say I don't know.
Context:
{context}

Question:
{question}

Answer:
"""
)


def generate_answer(question):
    context, _ = retrieve_context(question) #“I’m intentionally ignoring the second value (docs).”
                                            #This is Python best practice when a return value is not needed.

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response

#Final Prompt = :
# You are an AI assistant. Answer strictly using the context below.

#Context:
# Machine learning is a subset of AI...

# Question:
# What is machine learning?

# Answer:






# -------------------- STREAMLIT UI --------------------
import streamlit as st

st.set_page_config(page_title="RAG-based Multi-PDF Question Answering System")
st.title("📄 RAG-based Multi-PDF Question Answering System")

question = st.text_input("Ask a question from the PDFs")

if st.button("Ask"):
    if question:
        with st.spinner("Searching PDFs..."):
            answer = generate_answer(question)
        st.subheader("Answer")
        st.write(answer)
