import streamlit as st
from src.orchestrator.graph import RAGOrchestrator
from src.core.logger import logger

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Enterprise RAG with Guardrails")
st.markdown("High-assurance retrieval system with input/output verification.")

# 2. Resource Caching (Prevents reloading the system on every click)
@st.cache_resource
def load_orchestrator():
    return RAGOrchestrator()

orchestrator = load_orchestrator()

# 3. Sidebar for Enterprise Identity
with st.sidebar:
    st.header("Organization Settings")
    org_id = st.text_input("Organization ID", value="default-org-123")
    user_id = st.text_input("User ID", value="user-456")
    st.divider()
    st.info("This system uses a Sentry -> Librarian -> Auditor loop to ensure faithfulness.")

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audit" in message:
            with st.expander("View Audit Trail"):
                st.json(message["audit"])

# 5. Chat Input and Processing
if prompt := st.chat_input("Ask a professional query..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Verifying and Generating response..."):
            try:
                # Call the RAG Orchestrator
                result = orchestrator.run(
                    query=prompt,
                    org_id=org_id,
                    user_id=user_id
                )

                if result["status"] == "success":
                    response_text = result["response"]
                    status_icon = "✅"
                else:
                    response_text = "❌ **Security Alert:** Your request was rejected by the Sentry or Auditor guardrails due to a policy violation or hallucination."
                    status_icon = "⚠️"

                full_response = f"{status_icon} {response_text}"
                st.markdown(full_response)

                # Show the technical audit trail for transparency
                with st.expander("View Verification Audit"):
                    st.json(result.get("audit_trail", []))

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "audit": result
                })

            except Exception as e:
                st.error(f"An internal error occurred: {str(e)}")
                logger.error(f"Streamlit App Error: {str(e)}")
