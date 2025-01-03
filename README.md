# VidDoc Summarizer

VidDoc is a powerful Streamlit application that simplifies the process of summarizing YouTube videos and PDF documents. It leverages Google Gemini Pro and other cutting-edge technologies to provide you with concise and insightful notes from video transcripts and documents.

## Key Features

- **YouTube Video Summarizer**: Extracts and summarizes YouTube video transcripts into detailed, actionable points.
- **PDF Q&A**: Upload PDFs and interact with them by asking questions. The app will provide context-aware responses based on the content.
- **AI-Powered Summarization**: Powered by Google Gemini Pro to generate high-quality summaries and intelligent Q&A responses.
- **User-Friendly Interface**: Streamlit-based interface with a sidebar for easy navigation between features.
## How It Works

### YouTube Video Summarizer
1. **Extract Transcript**: The app first extracts the transcript from a YouTube video using the `YouTubeTranscriptApi`. 
2. **Summarization**: The extracted transcript is sent to Google Gemini Pro, which summarizes the content into key points and provides a detailed, concise summary.
3. **Display Summary**: The summary is displayed on the app, offering a quick overview of the video’s content.
![Alt text](https://github.com/Zem-0/VidDoc-Summarizer/blob/main/Screenshot%202025-01-03%20230143.png)
![Alt text](https://github.com/Zem-0/VidDoc-Summarizer/blob/main/Screenshot%202025-01-03%20230206.png)

### PDF Q&A
1. **Text Extraction**: The app extracts text from uploaded PDF files using `PyPDF2`. Multiple PDFs can be uploaded simultaneously.
2. **Text Chunking**: The extracted text is split into smaller chunks using `langchain`'s `RecursiveCharacterTextSplitter` to ensure that the text can be processed efficiently.
3. **Vectorization**: Each chunk of text is converted into vectors using Google Gemini’s embedding model and stored in a FAISS vector store.
4. **Question Answering**: Users can ask questions related to the content of the PDFs. The app performs a similarity search on the vector store to find relevant text and answers the question using Google Gemini Pro.
5. **Context-Aware Responses**: The app generates detailed, context-based answers by combining the extracted text and the AI’s ability to understand and summarize the content.


## Requirements

To run this project locally, you'll need the following dependencies:

- `streamlit`
- `PyPDF2`
- `langchain`
- `google-generativeai`
- `langchain-google-genai`
- `faiss-cpu`
- `youtube-transcript-api`
- `python-dotenv`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/smartnotes.git
   cd smartnotes
