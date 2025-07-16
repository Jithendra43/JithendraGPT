# JithendraGPT: AI-Powered Document Insight Engine

A powerful Streamlit application that combines document processing, vector search, and conversational AI to provide intelligent insights from your documents.

## Features

- 📄 **Document Processing**: Upload and process PDF, DOCX, TXT, and CSV files
- 🔍 **Vector Search**: Intelligent document search using sentence transformers
- 💬 **Conversational AI**: Chat with your documents using advanced LLMs
- 🎤 **Speech Input**: Voice-to-text functionality (when available)
- 🔊 **Speech Output**: Text-to-speech responses
- 🎛️ **Multiple AI Models**: Support for Groq and OpenAI models
- 📊 **Persistent Storage**: FAISS vector database for reliable cloud deployment
- 🔄 **Fallback System**: Automatic fallback from FAISS to ChromaDB if needed

## Supported AI Models

### Groq Models
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
- deepseek-r1-distill-llama-70b
- mixtral-8x7b-32768
- qwen-2.5-32b

### OpenAI Models
- gpt-4o
- gpt-4-turbo
- gpt-3.5-turbo-0125

## Deployment on Streamlit Cloud

### Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **API Keys**: Obtain API keys from:
   - [Groq](https://console.groq.com/) (recommended for cost-effectiveness)
   - [OpenAI](https://platform.openai.com/) (optional)

### Step-by-Step Deployment

1. **Fork or Clone this Repository**
   ```bash
   git clone <your-repo-url>
   cd Document-Insight-Engine
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path to `main.py`
   - Choose Python 3.9 or higher

3. **Configure Secrets**
   - In your Streamlit Cloud app dashboard, go to "Settings" → "Secrets"
   - Add your API keys:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for the deployment to complete

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   - Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   - Or create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

3. **Run the Application**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload Documents**: Use the sidebar to upload your documents
2. **Configure Settings**: Adjust temperature and select your preferred AI model
3. **Choose Input/Output Modes**: Select text or speech for input and output
4. **Start Chatting**: Ask questions about your documents
5. **Clear History**: Use the sidebar button to clear conversation history

## File Structure

```
Document-Insight-Engine/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt           # System packages for Streamlit Cloud
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── secrets.toml.template  # Template for secrets configuration
├── temp_uploads/          # Temporary file storage
├── vector_db_dir/         # Vector database storage
└── README.md             # This file
```

## Configuration Files

- **`requirements.txt`**: Python package dependencies
- **`packages.txt`**: System-level packages for speech functionality
- **`.streamlit/config.toml`**: Streamlit app configuration
- **`secrets.toml.template`**: Template for API key configuration

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API keys are correctly added to Streamlit secrets
   - Check that the keys are valid and have sufficient credits

2. **Speech Recognition Issues**
   - Speech functionality may not work on all deployment environments
   - The app gracefully falls back to text-only mode

3. **Memory Issues**
   - Large documents may cause memory issues
   - Consider splitting large documents into smaller chunks

4. **Model Availability**
   - Some models may have rate limits or availability issues
   - Try switching to a different model if one is unavailable

5. **Vector Store Issues**
   - App automatically uses FAISS (recommended for cloud deployment)
   - If you see ChromaDB errors, the app will fallback to FAISS
   - FAISS is more reliable on Streamlit Cloud than ChromaDB

### Support

For issues and questions:
- Check the Streamlit Cloud logs for detailed error messages
- Ensure all required dependencies are installed
- Verify API key configuration

## Cost Optimization

- **Groq**: Offers competitive pricing and fast inference
- **OpenAI**: Higher cost but potentially better quality for complex queries
- **Hugging Face**: Free embeddings model for vector search

## Security

- API keys are stored securely in Streamlit secrets
- Uploaded files are processed locally and not stored permanently
- Vector database is stored locally within the app environment

## License

This project is open source and available under the MIT License.

## Developer

**Jithendra Pavuluri**
