from fastapi import FastAPI,UploadFile,File,HTTPException
from typing import List
import os
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
import uuid

app=FastAPI()

UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

class QuestionRequest(BaseModel):
    user_question:str

class FileProcessor:
    def __init__(self):
        self.vector_store=None
        self.text_chunks=[]
        self.embedding_func=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def get_file_text(self,files):
        documents=[]
        try:
            for file_path in files:
                ext=os.path.splitext(file_path)[1].lower()
                if ext==".pdf":
                    loader=PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif ext==".txt":
                    loader=TextLoader(file_path,encoding="utf-8")
                    documents.extend(loader.load())
                else:
                    return f"Error:Unsupported file type for{file_path}"
            text=" ".join([doc.page_content for doc in documents if doc.page_content])
            if not text.strip():
                return "Error:No text found in files."
            return text
        except Exception as e:
            return f"Error reading file:{str(e)}"

    def get_text_chunks(self,text):
        try:
            text_splitter=SemanticChunker(
                embeddings=self.embedding_func,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )
            chunks=text_splitter.split_text(text)
            if not chunks:
                return "Error:Failed to split text into chunks."
            self.text_chunks=chunks
            return chunks
        except Exception as e:
            return f"Error splitting text:{str(e)}"

    def create_vector_store(self,text_chunks):
        try:
            documents=[Document(page_content=chunk) for chunk in text_chunks]
            self.vector_store=FAISS.from_documents(documents,embedding=self.embedding_func)
            return "Vector store created successfully."
        except Exception as e:
            return f"Error creating vector store: {str(e)}"

    def get_conversational_chain(self):
        try:
            prompt_template="""
            You are an expert at answering questions strictly based on provided documents.
            If the user asks to summarize/summary provide a detailed gist of the uploaded document/documents.
            Answer the question based strictly on the content of the uploaded document(s). 
            If the answer is not directly found,clearly state 'The answer is not in the provided context.'
            Context:{context}
            Question:{question}
            Detailed Answer:
            """
            llm=Ollama(model="llama3.2",temperature=0.5)
            prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
            chain=load_qa_chain(llm,chain_type="stuff",prompt=prompt)
            return chain
        except Exception as e:
            return f"Error creating conversational chain:{str(e)}"

    def user_input(self,user_question):
        try:
            if self.vector_store is None:
                return "Error:No vector store found.Please upload and process files first."
            docs=self.vector_store.similarity_search(user_question,k=5)
            if not docs:
                return "Error:No similar documents found."
            pairs=[[user_question,doc.page_content] for doc in docs]
            scores=self.cross_encoder.predict(pairs)
            sorted_docs=[doc for _,doc in sorted(zip(scores,docs),key=lambda x:x[0],reverse=True)][:3]
            chain=self.get_conversational_chain()
            if isinstance(chain,str) and chain.startswith("Error"):
                return chain
            response=chain({"input_documents":sorted_docs,"question":user_question})
            return response.get("output_text","The answer is not in the provided context.")
        except Exception as e:
            return f"Error processing user input:{str(e)}"

    def process_files(self,files):
        try:
            combined_text=self.get_file_text(files)
            if isinstance(combined_text,str) and combined_text.startswith("Error"):
                return combined_text
            text_chunks=self.get_text_chunks(combined_text)
            if isinstance(text_chunks,str) and text_chunks.startswith("Error"):
                return text_chunks
            vector_store_status=self.create_vector_store(text_chunks)
            if isinstance(vector_store_status,str) and vector_store_status.startswith("Error"):
                return vector_store_status
            return "Processing complete.You can now ask questions."
        except Exception as e:
            return f"Error processing files:{str(e)}"
        
file_processor=FileProcessor()

@app.post('/process_files')
async def process_files(files:List[UploadFile]=File(...)):
    try:
        file_paths=[]
        for file in files:
            if os.path.splitext(file.filename)[1].lower() not in [".pdf",".txt"]:
                raise HTTPException(status_code=400,detail=f"Unsupported file type:{file.filename}")
            file_path=os.path.join(UPLOAD_FOLDER,f"{uuid.uuid4()}_{file.filename}")
            with open(file_path,"wb") as f:
                f.write(file.file.read())
            file_paths.append(file_path)
        result=file_processor.process_files(file_paths)
        return {"message":result}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post('/ask_question')
async def ask_question(data:QuestionRequest):
    try:
        answer=file_processor.user_input(data.user_question)
        return {"answer":answer}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
