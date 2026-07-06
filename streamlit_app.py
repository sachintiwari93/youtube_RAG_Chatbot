import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ------------------ FUNCTIONS ------------------

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi().fetch(
    video_id,
    languages=["hi", "en"]
)
    full_text = " ".join([snippet.text for snippet in transcript])
    return full_text


def create_chunks(full_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([full_text])


# ------------------ LLM ------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

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

# ------------------ STREAMLIT UI ------------------

st.title("🎥 YouTube RAG Chatbot")

youtube_url = st.text_input(
    "Enter YouTube URL",
    "https://www.youtube.com/watch?v=LPZh9BOjkQs"
)

query = st.text_input("Ask Your Question")

# ------------------ MAIN ------------------

if youtube_url and query:

    from urllib.parse import urlparse, parse_qs

    def extract_video_id(url):
      parsed = urlparse(url)

      if parsed.hostname == "youtu.be":
        return parsed.path[1:]

      return parse_qs(parsed.query)["v"][0]

    video_id = extract_video_id(youtube_url)

    full_text = get_transcript(video_id)

    chunks = create_chunks(full_text)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)

    st.subheader("Answer")
    st.write(response.content)