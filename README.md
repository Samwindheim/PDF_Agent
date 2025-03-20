# PDF Agent: A Chatbot for PDF Document Interaction

## Overview

PDF Agent is a chatbot application that processes PDF documents, extracts metadata, and allows users to interact with the content through a chat interface. It is constructeed using the Langchain framework and utilizes OpenAI's language model to generate responses based on the conversation history and relevant document context. The workflow includes setting a vector store for document embeddings, processing PDF files, and managing conversation memory.

## Key Features

- Load and process PDF documents.
- Extract metadata such as date and author.
- Maintain conversation history with memory management.
- Generate AI responses using document context and chat history.
  
## Dependencies

The following dependencies are required for the PDF Agent to function:

- `langchain`
- `langchain-chroma`
- `langtrace-python-sdk`
- `python-dotenv`
- `openai`
- `langchain-community`

## Usage

1. **Run the Application**:
   Execute the following command to start the PDF Agent:
   ```bash
   python PDF_agent.py path_to_your_pdf.pdf
   ```

2. **Interact with the Chatbot**:
   Once the application is running, you can enter questions related to the PDF document. The chatbot will respond based on the content of the PDF and the conversation history.
