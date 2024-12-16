import boto3
import json
import os
import faiss
import traceback
from typing import Dict, Any, List, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Import LangChain components
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

# Core LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation, LLMResult

import time
from datetime import datetime, timedelta

# Global variables
s3_client = boto3.client('s3')
qa_chain = None

# Custom LLM class that uses Bedrock's 'converse' method
class BedrockConverseLLM(LLM):
    model_id: str
    region_name: str = "us-east-1"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9

    @property
    def _client(self):
        if not hasattr(self, '__client'):
            self.__client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self.__client

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        conversation = [{"role": "user", "content": [{"text": prompt}]}]

        try:
            response = self._client.converse(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p,
                },
            )

            response_message = response.get("output", {}).get("message", {}).get("content", [])
            if response_message and isinstance(response_message[0], dict):
                response_text = response_message[0].get("text", "")

                if stop:
                    for token in stop:
                        response_text = response_text.split(token)[0]
                    response_text = response_text.strip()

                return response_text

            raise ValueError("No valid response from model.")

        except Exception as e:
            raise ValueError(f"Error during Bedrock Converse API call: {e}")

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generation = Generation(text=text)
            generations.append([generation])

        return LLMResult(generations=generations)

# Custom Embeddings class
class BedrockEmbeddings(Embeddings):
    def __init__(self, model_id: str = 'amazon.titan-embed-text-v1', region_name: str = 'us-east-1'):
        self.model_id = model_id
        self.region_name = region_name
        self.client = boto3.client('bedrock-runtime', region_name=self.region_name)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def _get_embedding(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType='application/json',
            accept='application/json',
            body=body
        )
        response_body = response['body'].read()
        embedding = json.loads(response_body)['embedding']
        return embedding

def setup_conversational_retrieval_qa(vector_store, bedrock_llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        memory_size=1
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant providing information from Nulogy's policies. Use the following context to answer the question in a natural, conversational way.

If the user specifically asks about the source of information, include the document details in your response using this format:
"This information comes from [document name] in the [category] section"

Context:
{context}

Question:
{question}

Answer:
""",
    )

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Given the following conversation and a follow-up question, rephrase the question to be a standalone question.
If the user is asking about the source of information, preserve that intent in the standalone question.

Conversation:
{chat_history}

Follow-up Question:
{question}

Standalone question:
""",
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    question_generator = LLMChain(
        llm=bedrock_llm,
        prompt=condense_question_prompt
    )

    combine_docs_chain = load_qa_chain(
        llm=bedrock_llm,
        chain_type="stuff",
        prompt=qa_prompt
    )

    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        memory=memory,
        verbose=False,
        return_source_documents=True,
        output_key='answer',
    )

    return qa_chain

def create_conversational_chain(bucket_name: str):
    print("Creating conversational chain...")

    s3_client.download_file(bucket_name, 'faiss/faiss.index', '/tmp/faiss.index')
    index = faiss.read_index('/tmp/faiss.index')
    print(f"FAISS index loaded with {index.ntotal} vectors")

    s3_client.download_file(bucket_name, 'faiss/docstore.json', '/tmp/docstore.json')
    with open('/tmp/docstore.json', 'r') as f:
        docstore_dict_raw = json.load(f)

    docstore_dict = {}
    for k, v in docstore_dict_raw.items():
        docstore_dict[k] = Document(page_content=v['page_content'], metadata=v.get('metadata', {}))

    docstore = InMemoryDocstore(docstore_dict)
    print(f"Docstore keys: {list(docstore_dict.keys())}")

    s3_client.download_file(bucket_name, 'faiss/index_to_docstore_id.json', '/tmp/index_to_docstore_id.json')
    with open('/tmp/index_to_docstore_id.json', 'r') as f:
        index_to_docstore_id = json.load(f)

    if isinstance(index_to_docstore_id, list):
        index_to_docstore_id = {i: item for i, item in enumerate(index_to_docstore_id)}
    elif isinstance(index_to_docstore_id, dict):
        index_to_docstore_id = {int(k): v for k, v in index_to_docstore_id.items()}
    else:
        raise ValueError("index_to_docstore_id must be a dictionary or list")

    print(f"index_to_docstore_id keys: {list(index_to_docstore_id.keys())}")

    bedrock_embeddings = BedrockEmbeddings()
    vector_store = FAISS(
        embedding_function=bedrock_embeddings.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    bedrock_llm = BedrockConverseLLM(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    qa_chain = setup_conversational_retrieval_qa(vector_store, bedrock_llm)
    return qa_chain

def initialize_qa_chain():
    global qa_chain
    if qa_chain is None:
        bucket = os.environ['S3_BUCKET']
        qa_chain = create_conversational_chain(bucket)
        print("qa_chain initialized successfully")

def process_event(slack_event: Dict[str, Any]):
    global qa_chain
    try:
        event_data = slack_event.get('event', {})
        
        # Basic validations
        if not event_data or event_data.get('type') != 'app_mention':
            print("Ignoring non-app_mention event")
            return
            
        bot_user_id = os.environ.get('SLACK_BOT_USER_ID')
        if not bot_user_id or event_data.get('user') == bot_user_id:
            return
            
        text = event_data.get('text', '').strip()
        channel = event_data.get('channel')
        
        # Remove bot mention and clean text
        text = text.replace(f'<@{bot_user_id}>', '').strip()
        
        if not text or not channel:
            print("Missing text or channel")
            return
            
        print(f"Processing query: {text}")
        
        slack_token = os.environ.get('SLACK_OAUTH_TOKEN')
        if not slack_token:
            print("Missing Slack token")
            return
            
        slack_client = WebClient(token=slack_token)
        
        # Use message timestamp as unique identifier
        message_ts = event_data.get('ts')
        thread_ts = event_data.get('thread_ts', message_ts)

        # Check if it's a request for available documents
        if any(phrase in text.lower() for phrase in [
            'what documents do you have',
            'which documents do you have access to',
            'what do you have in your database',
            'what information do you have'
        ]):
            # Get unique documents organized by category
            documents_by_category = {}
            for doc in qa_chain.retriever.vectorstore.docstore._dict.values():
                metadata = doc.metadata
                category = metadata.get('category', 'Uncategorized')
                filename = metadata.get('filename', '').replace('.txt', '')
                if filename:
                    if category not in documents_by_category:
                        documents_by_category[category] = set()
                    documents_by_category[category].add(filename)
            
            # Format response
            response = "I have access to the following documents:\n\n"
            for category, docs in documents_by_category.items():
                response += f"*{category}*:\n"
                for doc in sorted(docs):
                    response += f"• {doc}\n"
            
            slack_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=response
            )
            return

        # Process regular query
        response = qa_chain.invoke({"question": text})
        answer = response.get('answer', "I apologize, I couldn't find an answer to that question.")
        
        # Check if source documents are available and if the question asks about sources
        source_docs = response.get('source_documents', [])
        if source_docs and any(word in text.lower() for word in ['source', 'where', 'which document', 'which policy']):
            # Get unique document sources - simplified format
            unique_sources = set()
            for doc in source_docs:
                metadata = doc.metadata
                filename = metadata.get('filename', '').replace('.txt', '')
                if filename:
                    unique_sources.add(filename)
            
            if unique_sources:
                source_info = "\n\nThis information comes from: " + ", ".join(sorted(unique_sources))
                answer += source_info
        
        # Send response in thread
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=answer
        )
            
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']}")
    except Exception as e:
        print(f"Error in process_event: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ProcessedSlackEvents')

def is_event_processed(event_id):
    try:
        response = table.get_item(Key={'EventID': event_id})
        return 'Item' in response
    except Exception as e:
        print(f"Error checking DynamoDB: {str(e)}")
        return False

def mark_event_processed(event_id):
    try:
        # Set TTL for 24 hours from now
        ttl = int((datetime.now() + timedelta(hours=24)).timestamp())
        table.put_item(
            Item={
                'EventID': event_id,
                'ProcessedAt': int(time.time()),
                'TTL': ttl
            }
        )
    except Exception as e:
        print(f"Error writing to DynamoDB: {str(e)}")

def lambda_handler(event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    print("Lambda function invoked with event:", json.dumps(event))
    
    try:
        # Immediately create the success response
        response = {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok'})
        }

        # Parse the event
        if isinstance(event.get('body'), dict):
            slack_event = event['body']
        else:
            slack_event = json.loads(event.get('body', '{}'))

        # Handle URL verification
        if slack_event.get('type') == 'url_verification':
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'text/plain'},
                'body': slack_event.get('challenge', '')
            }

        event_id = slack_event.get('event_id')
        if not event_id:
            print("No event_id found")
            return response

        if is_event_processed(event_id):
            print(f"Event {event_id} already processed")
            return response

        # Mark as processed before processing
        mark_event_processed(event_id)

        # Process event
        if slack_event.get('type') == 'event_callback':
            if qa_chain is None:
                initialize_qa_chain()
            process_event(slack_event)

        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# Initialize QA chain when Lambda container starts
initialize_qa_chain()
