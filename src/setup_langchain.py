# Sẽ tự động tạo Project ở bên trên LangSmith dựa vào LANGCHAIN_PROJECT trong file .env

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")