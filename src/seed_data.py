import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename (str): Tên file JSON cần đọc (ví dụ: 'data.json')
        directory (str): Thư mục chứa file (ví dụ: 'data_v3')
    Returns:
        tuple: Trả về (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm tạo và lưu vector embeddings vào Milvus từ dữ liệu local
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus để lưu dữ liệu
        filename (str): Tên file JSON chứa dữ liệu nguồn
        directory (str): Thư mục chứa file dữ liệu
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    """
    # Khởi tạo model embeddings tùy theo lựa chọn
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama2"  # hoặc model khác mà bạn đã cài đặt
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Determine file type and load accordingly
    file_extension = filename.split('.')[-1].lower()
    if file_extension == 'txt':
        local_data, doc_name = load_text_file(filename, directory)
    else:
        local_data, doc_name = load_data_from_local(filename, directory)
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=doc.get('page_content') or '',
            metadata={
                'source': doc['metadata'].get('source') or '',
                'content_type': doc['metadata'].get('content_type') or 'text/plain',
                'title': doc['metadata'].get('title') or '',
                'description': doc['metadata'].get('description') or '',
                'language': doc['metadata'].get('language') or 'en',
                'doc_name': doc_name,
                'start_index': doc['metadata'].get('start_index') or 0
            }
        )
        for doc in local_data
    ]

    print('documents: ', documents)

    # Tạo ID duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Khởi tạo và cấu hình Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    # Thêm documents vào Milvus
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def seed_milvus_live(URL: str, URI_link: str, collection_name: str, doc_name: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và tạo vector embeddings trong Milvus
    Args:
        URL (str): URL của trang web cần crawl dữ liệu
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus
        doc_name (str): Tên định danh cho tài liệu được crawl
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    """
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama2"  # hoặc model khác mà bạn đã cài đặt
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    documents = crawl_web(URL)

    # Cập nhật metadata cho mỗi document với giá trị mặc định
    for doc in documents:
        metadata = {
            'source': doc.metadata.get('source') or '',
            'content_type': doc.metadata.get('content_type') or 'text/plain',
            'title': doc.metadata.get('title') or '',
            'description': doc.metadata.get('description') or '',
            'language': doc.metadata.get('language') or 'en',
            'doc_name': doc_name,
            'start_index': doc.metadata.get('start_index') or 0
        }
        doc.metadata = metadata

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng model 'text-embedding-3-large' cho việc tạo embeddings khi truy vấn
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def load_text_file(filename: str, directory: str) -> list:
    """
    Load and process text file into documents
    Args:
        filename (str): Name of text file to read
        directory (str): Directory containing the file
    Returns:
        list: List of dictionaries with page_content and metadata
    """
    file_path = os.path.join(directory, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Create document format matching JSON structure
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            'page_content': chunk,
            'metadata': {
                'source': file_path,
                'content_type': 'text/plain',
                'title': filename,
                'description': f'Part {i+1} of {filename}',
                'language': 'en',
                'start_index': i
            }
        })
    
    return documents, filename.rsplit('.', 1)[0].replace('_', ' ')

def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    Thực hiện:
        1. Test seed_milvus với dữ liệu từ file local 'stack.json'
        2. Test seed_milvus với dữ liệu từ file text 'sample.txt'
        3. (Đã comment) Test seed_milvus_live với dữ liệu từ trang web stack-ai
    Chú ý:
        - Đảm bảo Milvus server đang chạy tại localhost:19530
        - Các biến môi trường cần thiết (như OPENAI_API_KEY) đã được cấu hình
    """
    # Test với file JSON
    print("\nTesting with JSON file...")
    seed_milvus('http://localhost:19530', 'stack_ai', 'stack.json', 'data', use_ollama=False)
    
    # Test với file text
    print("\nTesting with text file...")
    seed_milvus('http://localhost:19530', 'sample', 'sample.txt', 'data', use_ollama=False)
    
    # Test seed_milvus_live với URL trực tiếp (commented)
    # seed_milvus_live('https://www.stack-ai.com/docs', 'http://localhost:19530', 'data_test_live', 'stack-ai', use_ollama=False)

# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()
