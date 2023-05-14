import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAIChat
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your content")

    openai_api_key = st.sidebar.text_input('OPENAI_API_KEY:', value=os.getenv('OPENAI_API_KEY'))
    if openai_api_key is None or openai_api_key == '':
      st.sidebar.write("Check your .env file for OPENAI_API_KEY")
    else:
      st.sidebar.write("OPENAI_API_KEY is set")
    
    st.header("Ask your content")
    
    # upload file
    content = st.file_uploader("Upload your content", type=['pdf','txt'])
    
    # extract the text
    if content is not None:
      
      if not content.name.endswith(".pdf"):
        text = content.read()
      else:
        pdf_reader = PdfReader(content)
        text = ""
        for page in pdf_reader.pages:
          text += page.extract_text()
          
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your content:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAIChat(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()
