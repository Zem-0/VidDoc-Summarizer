import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Shared Gemini Pro Configuration
youtube_prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here: """

pdf_prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n
Answer:
"""

# Functions for YouTube Summarizer
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Functions for PDF Q&A
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=pdf_prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Enable dangerous deserialization for trusted sources
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


# Main App
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an Option:", ["YouTube Video Summarizer", "PDF Q&A"])

if option == "YouTube Video Summarizer":
    st.title("YouTube Transcript to Detailed Notes Converter")
    youtube_link = st.text_input("Enter YouTube Video Link:")
    
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

    if st.button("Get Detailed Notes"):
        try:
            transcript_text = extract_transcript_details(youtube_link)
            summary = generate_gemini_content(transcript_text, youtube_prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "PDF Q&A":
    st.title("Chat with PDF using Gemini 💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sidebar.success("PDFs processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
