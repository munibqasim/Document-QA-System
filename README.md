# Document QA System with RAG and Slack Integration

A production-ready document question-answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your document base through Slack. The system processes PDF documents, creates embeddings, and enables natural conversation through a Slack bot interface.

## Architecture

The system consists of three main Lambda functions that work together:

1. **PDF Processor**: Extracts text from PDF documents and saves processed content to S3
2. **Embedding Creator**: Generates embeddings from processed text using Amazon Bedrock
3. **Query Handler**: Manages Slack interactions and generates responses using RAG

```mermaid
graph TD
    A[PDF Upload] --> B[PDF Processor Lambda]
    B --> C[S3 Bucket/processed]
    C --> D[Embedding Creator Lambda]
    D --> E[FAISS Index]
    D --> F[Document Store]
    G[Slack Message] --> H[Query Handler Lambda]
    E --> H
    F --> H
    H --> I[Slack Response]
```

## Features

- **PDF Processing**
  - Extracts text from PDFs using pdf2image and Tesseract OCR
  - Handles document versioning
  - Maintains document metadata
  - Supports batch processing

- **Embedding Management**
  - Creates embeddings using Amazon Bedrock
  - Manages FAISS index for efficient similarity search
  - Handles document versioning and updates
  - Generates document summaries

- **Conversational Interface**
  - Slack bot integration
  - Natural language query processing
  - Conversation history management
  - Source document attribution
  - Document listing capability

## Prerequisites

- AWS Account with access to:
  - Lambda
  - S3
  - DynamoDB
  - Amazon Bedrock
- Slack Workspace with bot setup
- Python 3.8+

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── pdf_processor/
│   │   ├── __init__.py
│   │   └── handler.py
│   ├── embedding_creator/
│   │   ├── __init__.py
│   │   └── handler.py
│   └── query_handler/
│       ├── __init__.py
│       └── handler.py
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_embedding_creator.py
│   └── test_query_handler.py
└── infrastructure/
    └── terraform/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-qa-system.git
cd document-qa-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
aws configure
```

## Configuration

The system requires several environment variables:

```bash
# AWS Configuration
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1

# Slack Configuration
SLACK_OAUTH_TOKEN=xoxb-your-token
SLACK_BOT_USER_ID=your-bot-id

# Processing Configuration
BATCH_SIZE=50
```

## Deployment

1. Create required AWS resources:
```bash
cd infrastructure/terraform
terraform init
terraform apply
```

2. Deploy Lambda functions:
```bash
# You can use AWS SAM, Serverless Framework, or direct ZIP upload
aws lambda update-function-code --function-name pdf-processor --zip-file fileb://pdf_processor.zip
aws lambda update-function-code --function-name embedding-creator --zip-file fileb://embedding_creator.zip
aws lambda update-function-code --function-name query-handler --zip-file fileb://query_handler.zip
```

## Usage

1. **Processing PDFs**:
   - Upload PDF to the configured S3 bucket
   - The PDF processor Lambda will automatically process it
   - Embeddings will be created automatically

2. **Querying Documents**:
   - Mention the Slack bot in any channel
   - Ask questions about your documents
   - Use "what documents do you have" to see available documents

## Local Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.