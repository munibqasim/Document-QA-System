import boto3
import faiss
import numpy as np
import json
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import uuid
import os
import re
from datetime import datetime
import logging
from botocore.exceptions import ClientError
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_header_or_toc_line(line: str) -> bool:
    """Helper function to identify headers or TOC lines"""
    header_patterns = [
        r'^\s*#{1,6}\s',           # Markdown headers
        r'^\s*\d+\.\s',            # Numbered headers
        r'.*?\.{2,}.*?\d+\s*$',    # TOC lines with dots and page numbers
        r'^\s*[A-Z\s]{10,}\s*$',   # All caps lines (likely headers)
        r'^\s*Chapter\s+\d+',      # Chapter headers
        r'^\s*Section\s+\d+',      # Section headers
    ]
    return any(re.match(pattern, line) for pattern in header_patterns)

def generate_document_summary(content: str, num_sentences: int = 3) -> str:
    """
    Extract the first few sentences from the content as a summary,
    excluding table of contents and headers.
    """
    try:
        # Common TOC indicators and patterns to skip
        toc_patterns = [
            r"(?i)table\s+of\s+contents?.*?(?=\n\n)",  # "Table of Contents" section
            r"(?i)contents?.*?(?=\n\n)",               # Just "Contents" section
            r"(?m)^\s*\d+\.\s+.*?(?=\n\n)",           # Numbered sections like "1. Introduction"
            r"(?m)^.*?\.{2,}.*?\d+\s*$\n*",           # Dots leading to page numbers
            r"(?im)^index.*?(?=\n\n)",                # Index section
        ]
        
        # Create a copy of content to work with
        processed_content = content
        
        # Remove TOC patterns
        for pattern in toc_patterns:
            processed_content = re.sub(pattern, '', processed_content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove multiple newlines and clean up
        processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
        processed_content = processed_content.strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', processed_content)
        
        # Filter out short lines that might be headers
        valid_sentences = [
            sent for sent in sentences
            if len(sent.split()) > 3  # Skip very short phrases
            and not all(c.isupper() for c in sent.replace(' ', ''))  # Skip ALL CAPS lines
            and not sent.strip().startswith('#')  # Skip markdown headers
            and not re.match(r'^\d+(\.\d+)*\s', sent.strip())  # Skip numbered headers
        ]
        
        # Get first few valid sentences
        summary_sentences = valid_sentences[:num_sentences]
        summary = ' '.join(summary_sentences).strip()
        
        logger.info(f"Generated summary with {len(summary_sentences)} sentences, {len(summary)} characters")
        
        return summary if summary else "No valid summary could be generated."

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."

def remove_old_versions(docstore_dict: Dict, base_name: str, latest_version: int) -> Dict:
    """Remove embeddings for old versions of a document"""
    if not docstore_dict:
        logger.info("Empty docstore, no versions to remove")
        return docstore_dict
        
    to_remove = []
    for doc_id, doc in docstore_dict.items():
        try:
            metadata = doc.get('metadata', {})
            if (metadata.get('base_name') == base_name and
                metadata.get('version') != latest_version):
                to_remove.append(doc_id)
        except Exception as e:
            logger.warning(f"Error checking document {doc_id}: {str(e)}")
            continue
    
    # Remove old versions from docstore
    for doc_id in to_remove:
        del docstore_dict[doc_id]
    
    logger.info(f"Removed {len(to_remove)} old version chunks for {base_name}")
    return docstore_dict

def load_existing_indices(s3: Any, bucket: str) -> tuple:
    """Load existing FAISS index and docstore with retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading indices attempt {attempt + 1}/{max_retries}")
            
            # Download existing index
            s3.download_file(bucket, 'faiss/faiss.index', '/tmp/faiss.index')
            index = faiss.read_index('/tmp/faiss.index')
            logger.info("Successfully loaded FAISS index")

            # Load docstore
            response = s3.get_object(Bucket=bucket, Key='faiss/docstore.json')
            docstore_dict = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded docstore with {len(docstore_dict)} entries")

            # Load index to docstore mapping
            response = s3.get_object(Bucket=bucket, Key='faiss/index_to_docstore_id.json')
            index_to_docstore_id = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded index mapping with {len(index_to_docstore_id)} entries")

            return index, docstore_dict, index_to_docstore_id

        except s3.exceptions.NoSuchKey:
            logger.info("No existing indices found, creating new ones")
            return None, {}, []
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to load indices after {max_retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            continue

def list_processed_files(s3: Any, bucket: str, prefix: str) -> List[str]:
    """List all processed text files"""
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.txt'):
                files.append(obj['Key'])
    return files

def group_files_by_base_name(files: List[str]) -> Dict[str, Dict[int, str]]:
    """Group files by their base name and track versions"""
    groups = {}
    for file in files:
        base_name = re.sub(r'_V\d+\.txt$', '', file)
        version = extract_version_from_key(file)
        if base_name not in groups:
            groups[base_name] = {}
        groups[base_name][version] = file
    return groups

def extract_version_from_key(key: str) -> int:
    """Extract version number from file key"""
    match = re.search(r'_V(\d+)', key)
    return int(match.group(1)) if match else 1

def get_latest_version(versions: Dict[int, str]) -> int:
    """Get the latest version number from a dict of versions"""
    return max(versions.keys())

def save_indices(s3: Any, bucket: str, index: Any, docstore_dict: Dict, index_to_docstore_id: List[str]) -> None:
    """Save FAISS index and docstore to S3"""
    # Save FAISS index
    faiss.write_index(index, '/tmp/faiss.index')
    s3.upload_file('/tmp/faiss.index', bucket, 'faiss/faiss.index')

    # Save docstore and mapping
    s3.put_object(
        Body=json.dumps(docstore_dict),
        Bucket=bucket,
        Key='faiss/docstore.json'
    )
    s3.put_object(
        Body=json.dumps(index_to_docstore_id),
        Bucket=bucket,
        Key='faiss/index_to_docstore_id.json'
    )

def create_embeddings(event: Dict, context: Any) -> Dict:
    logger.info("Starting embeddings creation process")
    try:
        # Validate environment variables
        bucket = os.environ.get('S3_BUCKET')
        if not bucket:
            raise ValueError("S3_BUCKET environment variable is required")

        s3 = boto3.client('s3')
        bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')
        
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )

        processed_prefix = 'processed/'

        try:
            index, docstore_dict, index_to_docstore_id = load_existing_indices(s3, bucket)
            logger.info("Successfully loaded existing indices")
            
            new_embeddings = []
            new_doc_ids = []
            processed_files = 0
            failed_files = 0

            all_files = list_processed_files(s3, bucket, processed_prefix)
            logger.info(f"Found {len(all_files)} processed files")

            if not all_files:
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": "No files found to process",
                        "processed_prefix": processed_prefix
                    })
                }

            file_groups = group_files_by_base_name(all_files)

            for base_name, versions in file_groups.items():
                try:
                    latest_version = get_latest_version(versions)
                    logger.info(f"Processing {base_name}: V{latest_version}")

                    if docstore_dict:
                        docstore_dict = remove_old_versions(docstore_dict, base_name, latest_version)

                    file_key = versions[latest_version]
                    response = s3.get_object(Bucket=bucket, Key=file_key)
                    content = response['Body'].read().decode('utf-8')
                    metadata = response.get('Metadata', {})
                    
                    category = metadata.get('category', file_key.split('/')[1] if '/' in file_key else 'uncategorized')
                    
                    # Generate summary using improved function
                    summary = generate_document_summary(content)
                    if summary in ["Error generating summary.", "No valid summary could be generated."]:
                        logger.warning(f"Could not generate proper summary for {file_key}")
                    
                    chunks = text_splitter.split_text(content)
                    logger.info(f"Created {len(chunks)} chunks for {file_key}")

                    chunk_success = 0
                    for i, chunk in enumerate(chunks):
                        try:
                            doc_id = str(uuid.uuid4())
                            embedding = bedrock_embeddings.embed_query(chunk)

                            chunk_metadata = {
                                "source": file_key,
                                "category": category,
                                "version": latest_version,
                                "base_name": base_name,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "processed_date": datetime.now().isoformat(),
                                "document_summary": summary
                            }

                            docstore_dict[doc_id] = {
                                "page_content": chunk,
                                "metadata": chunk_metadata
                            }

                            new_embeddings.append(embedding)
                            new_doc_ids.append(doc_id)
                            index_to_docstore_id.append(doc_id)
                            chunk_success += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {i} of {file_key}: {str(e)}")
                            continue

                    logger.info(f"Successfully processed {chunk_success}/{len(chunks)} chunks for {file_key}")
                    processed_files += 1

                except Exception as e:
                    logger.error(f"Error processing file {base_name}: {str(e)}")
                    failed_files += 1
                    continue

            if new_embeddings:
                embeddings_np = np.array(new_embeddings).astype('float32')
                if index is None:
                    index = faiss.IndexFlatL2(embeddings_np.shape[1])
                index.add(embeddings_np)
                logger.info(f"Added {len(new_embeddings)} new embeddings to index")

                save_indices(s3, bucket, index, docstore_dict, index_to_docstore_id)
                logger.info("Successfully saved updated indices to S3")

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Embeddings creation completed",
                    "new_embeddings_count": len(new_embeddings),
                    "total_embeddings": len(index_to_docstore_id),
                    "files_processed": processed_files,
                    "files_failed": failed_files
                })
            }

        except Exception as e:
            logger.error(f"Error in main processing loop: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Error creating embeddings",
                "details": str(e)
            })
        }
