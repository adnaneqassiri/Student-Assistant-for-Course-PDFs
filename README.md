Student Assistant for Course PDFs is an intelligent document interaction tool designed to transform the way students study. Instead of endlessly scrolling through hundreds of pages of lecture slides or textbooks, this app allows you to have a live conversation with your course materials.

## 🏗️ Architecture & Workflow

The application follows a standard RAG pipeline to ensure accurate and context-aware responses:

1.  **Data Ingestion:** Course PDFs are uploaded and parsed.
2.  **Chunking:** The text is broken down into smaller, manageable "chunks" to maintain context.
3.  **Embeddings:** Each chunk is converted into a numerical vector using an embedding model.
4.  **Vector Database:** These vectors are stored in a specialized database for fast similarity searching.
5.  **Retrieval:** When you ask a question (Prompt), the system searches the VectorDB for the most relevant chunks.
6.  **Augmented Generation:** The relevant chunks (Context) + your original question (Prompt) are sent to the LLM to generate a precise answer.

### System Diagram

<img src="imgs/Untitled Diagram.drawio (1).png">

-----



## 🛠️ Technologies Used

The project leverages a modern AI stack to build a robust and scalable RAG (Retrieval-Augmented Generation) application:

### **Core Frameworks**
* **[Streamlit](https://streamlit.io/):** Used to build the interactive web interface for file uploading and real-time chat.
* **[LangChain](https://www.langchain.com/):** The orchestration framework used to manage the flow between the PDF parser, embeddings, retriever, and the LLM.

### **AI & Machine Learning**
* **Large Language Model (LLM):** Powered by **OpenAI (GPT-4/3.5)** or **Google Gemini** for generating human-like responses based on the retrieved context.
* **Embeddings:** Utilizes **HuggingFace Instruct Embeddings** or **OpenAI Embeddings** to convert text chunks into high-dimensional vectors.

### **Data Management**
* **Vector Database**: **ChromaDB** for efficient similarity search and retrieval of document chunks.
* **Document Processing:** **LangChain Community PDF Loaders** for extracting raw text from course materials.