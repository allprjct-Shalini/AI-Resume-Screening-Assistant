import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="Insurance Claim Processing Agent", layout="wide")

st.title("🛡️ Insurance Claim Processing Agent (LangGraph)")
st.write("Automate policy verification, document checks, fraud detection, and routing.")

# Define State
class ClaimState(TypedDict):
    claim_id: str
    claim_amount: float
    policy_status: str  # 'active' or 'expired'
    documents_submitted: List[str]
    doc_verified: bool
    eligibility_valid: bool
    fraud_score: float
    summary: str
    decision: str

# Node 1: Document Verification Agent
def verify_documents(state: ClaimState):
    required_docs = {"id_proof", "medical_bill", "incident_report"}
    submitted = set(state["documents_submitted"])
    return {"doc_verified": required_docs.issubset(submitted)}

# Node 2: Eligibility Check Agent
def check_eligibility(state: ClaimState):
    return {"eligibility_valid": state["policy_status"].lower() == "active"}

# Node 3: Fraud Detection Agent
def detect_fraud(state: ClaimState):
    # Simple rule: Claims over $10,000 flagged as potential fraud risk
    if state["claim_amount"] > 10000:
        fraud_score = 0.85  # High risk
    else:
        fraud_score = 0.10  # Low risk
    return {"fraud_score": fraud_score}

# Node 4: Claim Summary Agent
def generate_summary(state: ClaimState):
    doc_status = "Valid" if state["doc_verified"] else "Invalid/Missing Documents"
    elig_status = "Eligible" if state["eligibility_valid"] else "Ineligible/Expired Policy"
    risk_level = "High Risk" if state["fraud_score"] > 0.5 else "Low Risk"

    summary = (
        f"Claim ID: {state['claim_id']} | Amount: ${state['claim_amount']} | "
        f"Documents: {doc_status} | Policy: {elig_status} | Fraud Risk: {risk_level}"
    )
    return {"summary": summary}

# Node 5: Decision / Routing Logic
def make_decision(state: ClaimState):
    if not state["doc_verified"]:
        decision = "Reject (Missing Documents)"
    elif not state["eligibility_valid"]:
        decision = "Reject (Expired Policy)"
    elif state["fraud_score"] > 0.5 or state["claim_amount"] > 10000:
        decision = "Escalate for Human Review (High-Value / High Risk)"
    else:
        decision = "Auto Approve"
    return {"decision": decision}

# Build LangGraph Workflow
workflow = StateGraph(ClaimState)

workflow.add_node("verify_documents", verify_documents)
workflow.add_node("check_eligibility", check_eligibility)
workflow.add_node("detect_fraud", detect_fraud)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("make_decision", make_decision)

# Set entry point
workflow.set_entry_point("verify_documents")

# Parallel execution: documents, eligibility, and fraud run independently
workflow.add_edge("verify_documents", "check_eligibility")
workflow.add_edge("verify_documents", "detect_fraud")
workflow.add_edge("check_eligibility", "generate_summary")
workflow.add_edge("detect_fraud", "generate_summary")

workflow.add_edge("generate_summary", "make_decision")
workflow.add_edge("make_decision", END)

app = workflow.compile()

# Streamlit UI Inputs
st.sidebar.header("Claim Details Input")
claim_id = st.sidebar.text_input("Claim ID", value="CLM-98765")
claim_amount = st.sidebar.number_input("Claim Amount ($)", value=3500.0, step=500.0)
policy_status = st.sidebar.selectbox("Policy Status", ["active", "expired"])
docs = st.sidebar.multiselect(
    "Submitted Documents", 
    ["id_proof", "medical_bill", "incident_report"], 
    default=["id_proof", "medical_bill", "incident_report"]
)

if st.button("Process Claim"):
    initial_state = {
        "claim_id": claim_id,
        "claim_amount": claim_amount,
        "policy_status": policy_status,
        "documents_submitted": docs,
        "doc_verified": False,
        "eligibility_valid": False,
        "fraud_score": 0.0,
        "summary": "",
        "decision": ""
    }
    
    with st.spinner("Running Multi-Agent Workflow..."):
        final_state = app.invoke(initial_state)
        
        st.success("Claim Workflow Completed Successfully!")
        st.markdown("### 📋 Claim Processing Report")
        st.info(final_state["summary"])
        
        decision = final_state["decision"]
        if "Auto Approve" in decision:
            st.success(f"**Final Decision:** {decision}")
        elif "Reject" in decision:
            st.error(f"**Final Decision:** {decision}")
        else:
            st.warning(f"**Final Decision:** {decision}")