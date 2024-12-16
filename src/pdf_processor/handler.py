import os
import boto3
from pdf2image import convert_from_path
import pytesseract
import json
import re
from datetime import datetime
import base64

def sanitize_metadata_value(value):
    """
    Sanitize metadata values to ensure they're valid for S3 headers.
    S3 metadata values must be valid ASCII and can't contain certain characters.
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Convert to ASCII, replacing non-ASCII characters
    try:
        # First encode to ASCII, replacing invalid chars with ?
        ascii_str = value.encode('ascii', 'replace').decode('ascii')
        # Remove newlines and other problematic characters
        sanitized = re.sub(r'[\n\r\t]', ' ', ascii_str)
        # Limit length (S3 metadata values have size limits)
        sanitized = sanitized[:1024]
        return sanitized
    except Exception:
        # If all else fails, base64 encode the value
        return base64.b64encode(value.encode('utf-8')).decode('ascii')

def extract_version(filename):
    """Extract version number from filename"""
    match = re.search(r'_V(\d+)', filename)
    return int(match.group(1)) if match else 1

def get_category_from_path(s3_key):
    """Extract category from S3 path"""
    parts = s3_key.split('/')
    if len(parts) > 1:
        return parts[1]  # Assuming structure: documents/category/filename
    return "uncategorized"

def extract_text_and_metadata(pdf_path, images):
    """Extract text and basic metadata from the first few pages"""
    text = ""
    description = ""
    first_page = True
    
    try:
        for image in images:
            page_text = pytesseract.image_to_string(image)
            if first_page:
                # Get first few sentences for description
                sentences = re.split(r'(?<=[.!?])\s+', page_text)
                description = ' '.join(sentences[:3])
                first_page = False
            text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Text extraction failed: {str(e)}")
    
    return text, description.strip()

def list_pdfs_in_folder(s3_client, bucket, prefix):
    """List all PDF files in the given S3 prefix"""
    pdfs = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith('.pdf'):
                        pdfs.append(obj['Key'])
        return pdfs
    except Exception as e:
        print(f"Error listing PDFs: {str(e)}")
        return []

def handler(event, context):
    print("Processing PDF documents")
    
    s3 = boto3.client('s3')
    processed_files = []
    failed_files = []
    
    try:
        # If event contains S3 trigger
        if 'Records' in event:
            s3_bucket = event['Records'][0]['s3']['bucket']['name']
            s3_key = event['Records'][0]['s3']['object']['key']
            pdfs_to_process = [s3_key]
            print(f"Processing single file from S3 trigger: {s3_key}")
            
        # If processing a single specified file
        elif event.get('file_path'):
            s3_bucket = os.environ.get('S3_BUCKET')
            pdfs_to_process = [event['file_path']]
            print(f"Processing single specified file: {event['file_path']}")
            
        # If processing a folder
        elif event.get('process_all') and event.get('folder_path'):
            s3_bucket = os.environ.get('S3_BUCKET')
            prefix = event['folder_path']
            BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 50))
            
            pdfs_to_process = list_pdfs_in_folder(s3, s3_bucket, prefix)
            pdfs_to_process = pdfs_to_process[:BATCH_SIZE]
            print(f"Processing folder {prefix} with batch size {BATCH_SIZE}")
            
        else:
            raise ValueError("Invalid input parameters. Provide either 'file_path' or both 'process_all' and 'folder_path'")
            
        print(f"Found {len(pdfs_to_process)} PDF files to process")
        
        for s3_key in pdfs_to_process:
            try:
                if not s3_key.lower().endswith('.pdf'):
                    continue

                # Extract metadata
                filename = s3_key.split('/')[-1]
                category = get_category_from_path(s3_key)
                version = extract_version(filename)
                
                # Download PDF from S3
                pdf_path = f"/tmp/{filename}"
                s3.download_file(s3_bucket, s3_key, pdf_path)
                
                # Convert PDF to images
                print(f"Converting PDF to images: {filename}")
                images = convert_from_path(pdf_path)
                
                # Extract text and description
                print("Extracting text using Tesseract")
                text, description = extract_text_and_metadata(pdf_path, images)
                
                if not text.strip():
                    raise Exception("No text extracted from document")
                    
                # Prepare metadata with sanitization
                metadata = {
                    "filename": sanitize_metadata_value(filename),
                    "category": sanitize_metadata_value(category),
                    "version": str(version),
                    "description": sanitize_metadata_value(description),
                    "processed_date": datetime.now().isoformat(),
                    "source_pdf": sanitize_metadata_value(s3_key)
                }
                
                # Upload the extracted text and metadata back to S3
                output_text_key = f"processed/{category}/{filename.replace('.pdf', '.txt')}"
                
                # Save text with sanitized metadata
                s3.put_object(
                    Body=text,
                    Bucket=s3_bucket,
                    Key=output_text_key,
                    Metadata=metadata
                )
                
                print(f"Successfully processed {filename}")
                print(f"Category: {category}, Version: {version}")
                print(f"Output saved to: {output_text_key}")
                
                # Clean up
                os.remove(pdf_path)
                
                processed_files.append(s3_key)
                
            except Exception as e:
                error_message = str(e)
                print(f"Error processing {s3_key}: {error_message}")
                failed_files.append({"file": s3_key, "error": error_message})
                
                # Clean up temp file if it exists
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.remove(pdf_path)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Processing completed. Processed {len(processed_files)} files",
                "processed_files": processed_files,
                "failed_files": failed_files
            })
        }
                
    except Exception as e:
        error_message = str(e)
        print(f"Critical error: {error_message}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Error processing documents",
                "details": error_message,
                "processed_files": processed_files,
                "failed_files": failed_files
            })
        }
