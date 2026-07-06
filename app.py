import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
def get_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(video_id)

    full_text = " ".join([snippet.text for snippet in transcript])

    return full_text
video_id = "LPZh9BOjkQs"

full_text = get_transcript(video_id)

print(full_text[:500])
def create_chunks(full_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([full_text])

    return chunks
chunks = create_chunks(full_text)

print("Number of Chunks:", len(chunks))
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embedding)

print("✅ Vector Store Created Successfully")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("✅ Retriever Created")
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

print("✅ Groq Connected")
prompt = PromptTemplate(
    template="""
Answer the question only from the given context.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

print("✅ Prompt Ready")
query = input("Ask Your Question: ")

docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

final_prompt = prompt.format(
    context=context,
    question=query
)

response = llm.invoke(final_prompt)

print("\nAnswer:\n")
print(response.content)