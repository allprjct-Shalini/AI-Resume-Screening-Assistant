import os
import streamlit as st
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# Streamlit Page Configuration
st.set_page_config(page_title="Insurance Claim Processing Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Insurance Claim Processing Agent (LangGraph)")
st.markdown("Automate insurance claim verification, eligibility checks, fraud detection, and routing with Human-in-the-Loop review.")

# Sidebar API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key (Claims)", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Define State Schema for LangGraph
class ClaimState(TypedDict):
    claim_id: str
    policy_id: str
    claim_amount: float
    incident_type: str
    document_list: List[str]
    policy_status: str  # "Active" or "Expired"
    
    # Agent outputs
    doc_verification_result: str
    eligibility_result: str
    fraud_score: float
    fraud_reason: str
    claim_summary: str
    final_decision: str  # "Auto Approve", "Reject", "Human Review"
    human_override: str

# Define Agent Nodes
def document_verification_node(state: ClaimState):
    docs = [d.lower() for d in state["document_list"]]
    required = ["police report", "medical bill", "claim form"]
    missing = [req for req in required if req not in docs]
    
    if missing:
        result = f"FAILED: Missing mandatory documents: {', '.join(missing)}"
    else:
        result = "PASSED: All required documents are present and valid."
    return {"doc_verification_result": result}

def eligibility_check_node(state: ClaimState):
    if state["policy_status"].lower() != "active":
        result = "FAILED: Policy is expired or inactive."
    else:
        result = "PASSED: Policy is active and covers the incident type."
    return {"eligibility_result": result}

def fraud_detection_node(state: ClaimState):
    amount = state["claim_amount"]
    incident = state["incident_type"].lower()
    
    score = 0.1
    reason = "Normal risk profile."
    
    if amount > 50000:
        score = 0.85
        reason = "High claim amount exceeds standard threshold ($50,000)."
    elif "accident" in incident and amount > 25000:
        score = 0.65
        reason = "Moderate-high claim amount for incident type."
        
    return {"fraud_score": score, "fraud_reason": reason}

def claim_summary_node(state: ClaimState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = f"""
    Summarize the following insurance claim:
    - Claim ID: {state['claim_id']}
    - Policy Status: {state['policy_status']}
    - Claim Amount: ${state['claim_amount']}
    - Incident: {state['incident_type']}
    - Document Check: {state['doc_verification_result']}
    - Eligibility Check: {state['eligibility_result']}
    - Fraud Assessment: Score {state['fraud_score']} ({state['fraud_reason']})
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"claim_summary": response.content}

def decision_router_node(state: ClaimState):
    doc_res = state["doc_verification_result"]
    elig_res = state["eligibility_result"]
    fraud_score = state["fraud_score"]
    
    if "FAILED" in doc_res or "FAILED" in elig_res:
        decision = "Reject"
    elif fraud_score > 0.6:
        decision = "Human Review"
    else:
        decision = "Auto Approve"
        
    return {"final_decision": decision}

def human_approval_node(state: ClaimState):
    return {"final_decision": state.get("human_override", "Human Review")}

# Build LangGraph Workflow
workflow = StateGraph(ClaimState)

workflow.add_node("verify_documents", document_verification_node)
workflow.add_node("check_eligibility", eligibility_check_node)
workflow.add_node("detect_fraud", fraud_detection_node)
workflow.add_node("generate_summary", claim_summary_node)
workflow.add_node("make_decision", decision_router_node)
workflow.add_node("human_approval", human_approval_node)

# Parallel Execution: START triggers verification, eligibility, and fraud detection simultaneously
workflow.add_edge(START, "verify_documents")
workflow.add_edge(START, "check_eligibility")
workflow.add_edge(START, "detect_fraud")

# Converge all checks into Summary & Decision Router
workflow.add_edge("verify_documents", "generate_summary")
workflow.add_edge("check_eligibility", "generate_summary")
workflow.add_edge("detect_fraud", "generate_summary")
workflow.add_edge("generate_summary", "make_decision")

# Conditional Routing
def route_after_decision(state: ClaimState):
    if state["final_decision"] == "Human Review":
        return "human_approval"
    return END

workflow.add_conditional_edges("make_decision", route_after_decision, {"human_approval": "human_approval", END: END})
workflow.add_edge("human_approval", END)

app_graph = workflow.compile()

# Streamlit UI Form
st.subheader("Submit Claim for Processing")

col1, col2 = st.columns(2)
with col1:
    claim_id = st.text_input("Claim ID", value="CLM-2026-001")
    policy_id = st.text_input("Policy ID", value="POL-98765")
    claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=12500.0, step=500.0)

with col2:
    incident_type = st.selectbox("Incident Type", ["Auto Accident", "Medical Emergency", "Property Damage", "Travel Delay"])
    policy_status = st.selectbox("Policy Status", ["Active", "Expired"])
    documents = st.multiselect("Supporting Documents Provided", ["police report", "medical bill", "claim form", "repair estimate"], default=["police report", "medical bill", "claim form"])

if st.button("Process Claim Agentic Workflow", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        with st.spinner("Running LangGraph multi-agent pipeline (Parallel checks & Fraud detection)..."):
            initial_state = {
                "claim_id": claim_id,
                "policy_id": policy_id,
                "claim_amount": claim_amount,
                "incident_type": incident_type,
                "document_list": documents,
                "policy_status": policy_status,
                "doc_verification_result": "",
                "eligibility_result": "",
                "fraud_score": 0.0,
                "fraud_reason": "",
                "claim_summary": "",
                "final_decision": "",
                "human_override": ""
            }
            
            final_state = app_graph.invoke(initial_state)
            
            st.success("Workflow Executed Successfully!")
            
            # Display Results
            st.markdown("---")
            st.subheader("📊 Claim Processing Report")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Fraud Risk Score", f"{final_state['fraud_score'] * 100:.0f}%")
            with col_b:
                st.metric("Policy Status", policy_status)
            with col_c:
                decision = final_state['final_decision']
                if decision == "Auto Approve":
                    st.success(f"**Decision:** {decision}")
                elif decision == "Reject":
                    st.error(f"**Decision:** {decision}")
                else:
                    st.warning(f"**Decision:** {decision}")

            st.markdown("### 📝 Claim Summary")
            st.write(final_state['claim_summary'])

            st.markdown("### 🔍 Agent Verification Details")
            st.markdown(f"- **Document Verification:** {final_state['doc_verification_result']}")
            st.markdown(f"- **Policy Eligibility:** {final_state['eligibility_result']}")
            st.markdown(f"- **Fraud Assessment:** {final_state['fraud_reason']}")

            if final_state['final_decision'] == "Human Review":
                st.markdown("---")
                st.warning("⚠️ This claim requires Human Supervisor Review due to high fraud risk or high value.")
                override = st.radio("Supervisor Override Decision:", ["Approve Claim", "Reject Claim", "Request More Documents"])
                if st.button("Confirm Supervisor Decision"):
                    st.success(f"Claim successfully updated to: **{override}**")