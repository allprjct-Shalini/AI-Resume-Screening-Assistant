%%writefile resume_app.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="AI Resume Screening Assistant", layout="wide")

st.title("📄 AI Resume Screening Assistant (LangChain + RAG)")
st.write("Upload candidate resumes and evaluate them against a Job Description.")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description (JD)")

if st.button("Evaluate Resumes"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.error("Please enter a Job Description.")
    else:
        with st.spinner("Processing resumes and evaluating against JD..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                all_docs.extend(docs)
                os.unlink(tmp_path)
            
            # Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            # Vector Store & Retriever
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # LLM & Prompt
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            system_prompt = (
                "You are an expert HR recruiter and AI Resume Screening Assistant.\n"
                "Analyze the candidate resume against the provided Job Description using only the retrieved context.\n"
                "Provide your evaluation in the following clear sections:\n"
                "1. **Candidate Name / File**\n"
                "2. **Match Score (0-100)**\n"
                "3. **Matching Skills**\n"
                "4. **Missing Skills**\n"
                "5. **Candidate Summary**\n"
                "6. **Strengths & Weaknesses**\n"
                "7. **Hiring Recommendation (Auto Approve / Reject / Human Review) with Justification**\n\n"
                "Context:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Job Description:\n{input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": job_description})
            
            st.success("Evaluation Complete!")
            st.markdown("### 📊 Evaluation Results")
            st.markdown(response["answer"])
