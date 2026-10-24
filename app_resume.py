import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import tempfile

# Streamlit Page Configuration
st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening Assistant (LangChain & RAG)")
st.markdown("Upload candidate resumes (PDF) and enter a Job Description to evaluate matches, skills, strengths, and hiring recommendations.")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Pydantic Output Schema for Structured JSON response
class ResumeEvaluation(BaseModel):
    match_score: int = Field(description="Match score between 0 and 100 based on JD alignment")
    matching_skills: list[str] = Field(description="List of skills found in the resume matching the JD")
    missing_skills: list[str] = Field(description="List of required skills from JD missing in the resume")
    candidate_summary: str = Field(description="Brief professional summary of the candidate")
    strengths: list[str] = Field(description="Key strengths of the candidate for this role")
    weaknesses: list[str] = Field(description="Potential weaknesses or gaps")
    hiring_recommendation: str = Field(description="Recommendation: Strong Hire, Hire, Hold, or Reject with justification")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area("Paste the Job Description here:", height=250, placeholder="Enter required skills, experience, and responsibilities...")

with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

if st.button("Evaluate Resumes", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not job_description:
        st.warning("Please provide a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        with st.spinner("Processing resumes and evaluating against JD..."):
            parser = JsonOutputParser(pydantic_object=ResumeEvaluation)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert HR recruiter and technical hiring manager. Analyze the provided resume context against the given Job Description and return your evaluation strictly in JSON format matching the schema instructions.\n{format_instructions}"),
                ("human", "Job Description:\n{job_description}\n\nCandidate Resume Context:\n{context}")
            ])
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            chain = prompt | llm | parser

            for uploaded_file in uploaded_files:
                st.markdown("---")
                st.markdown(f"### 📋 Evaluation Report for: `{uploaded_file.name}`")
                
                # Save uploaded PDF to a temporary file to load with PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Load and split PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)

                    # Create Vector Store & Retriever
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    
                    # Retrieve relevant resume text chunks
                    retrieved_docs = retriever.invoke(job_description)
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    # Invoke LLM Chain
                    result = chain.invoke({
                        "job_description": job_description,
                        "context": context_text,
                        "format_instructions": parser.get_format_instructions()
                    })

                    # Display Results in Streamlit
                    col_score, col_rec = st.columns(2)
                    with col_score:
                        st.metric(label="Match Score", value=f"{result.get('match_score', 0)} / 100")
                    with col_rec:
                        st.info(f"**Recommendation:** {result.get('hiring_recommendation', 'N/A')}")

                    st.markdown(f"**Candidate Summary:** {result.get('candidate_summary', '')}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**✅ Matching Skills:**")
                        for s in result.get('matching_skills', []):
                            st.markdown(f"- {s}")
                        st.markdown("**💪 Strengths:**")
                        for s in result.get('strengths', []):
                            st.markdown(f"- {s}")
                    with c2:
                        st.markdown("**❌ Missing Skills:**")
                        for s in result.get('missing_skills', []):
                            st.markdown(f"- {s}")
                        st.markdown("**⚠️ Weaknesses / Gaps:**")
                        for w in result.get('weaknesses', []):
                            st.markdown(f"- {w}")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)