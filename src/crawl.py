import os
import re
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
def bs4_extractor(html: str) -> str:
    """
    Hàm trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, "html.parser")  # Phân tích cú pháp HTML
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()  # Xóa khoảng trắng và dòng trống thừa

def crawl_web(url_data):
    """
    Hàm crawl dữ liệu từ URL với chế độ đệ quy
    Args:
        url_data (str): URL gốc để bắt đầu crawl
    Returns:
        list: Danh sách các Document object, mỗi object chứa nội dung đã được chia nhỏ
              và metadata tương ứng
    """
    # Tạo loader với độ sâu tối đa là 4 cấp
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()  # Tải nội dung
    print('length: ', len(docs))  # In số lượng tài liệu đã tải
    
    # Chia nhỏ văn bản thành các đoạn 10000 ký tự, với 500 ký tự chồng lấp
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)
    print('length_all_splits: ', len(all_splits))  # In số lượng đoạn văn bản sau khi chia
    return all_splits

def web_base_loader(url_data):
    """
    Hàm tải dữ liệu từ một URL đơn (không đệ quy)
    Args:
        url_data (str): URL cần tải dữ liệu
    Returns:
        list: Danh sách các Document object đã được chia nhỏ
    """
    loader = WebBaseLoader(url_data)  # Tạo loader cơ bản
    docs = loader.load()  # Tải nội dung
    print('length: ', len(docs))  # In số lượng tài liệu
    
    # Chia nhỏ văn bản tương tự như trên
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def save_data_locally(documents, filename, directory):
    """
    Lưu danh sách documents vào file JSON
    Args:
        documents (list): Danh sách các Document object cần lưu
        filename (str): Tên file JSON (ví dụ: 'data.json')
        directory (str): Đường dẫn thư mục lưu file
    Returns:
        None: Hàm không trả về giá trị, chỉ lưu file và in thông báo
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)  # Tạo đường dẫn đầy đủ

    # Chuyển đổi documents thành định dạng có thể serialize
    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    # Lưu vào file JSON
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f'Data saved to {file_path}')  # In thông báo lưu thành công

def main():
    """
    Hàm chính điều khiển luồng chương trình:
    1. Crawl dữ liệu từ trang web stack-ai
    2. Lưu dữ liệu đã crawl vào file JSON
    3. In kết quả crawl để kiểm tra
    """
    # Crawl dữ liệu từ trang docs của stack-ai
    data = crawl_web('https://www.stack-ai.com/docs')
    # Lưu dữ liệu vào thư mục data_v2
    save_data_locally(data, 'stack.json', 'data')
    print('data: ', data)  # In dữ liệu đã crawl

# Kiểm tra nếu file được chạy trực tiếp
if __name__ == "__main__":
    main()
