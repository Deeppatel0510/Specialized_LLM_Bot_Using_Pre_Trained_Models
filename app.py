import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import bs4

# Load Environment Variables
load_dotenv()

# Set API Keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Streamlit Config
st.set_page_config(page_title="TeroTAM AI", layout="centered")
st.title("TeroTAM AI")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vectors" not in st.session_state:
    st.session_state["vectors"], st.session_state["embeddings"] = None, None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat_session"

# File Paths
pdf_folder = "documents"
faiss_index_path = "faiss_index"
processed_file_path = "processed_files.pkl"
processed_urls_path = "processed_urls.pkl"
scraped_data_path = "scraped_data.txt"

# Load and Process Documents
def load_process_documents():
    model =  OllamaEmbeddings(model="mxbai-embed-large")
    vectors = None

    # Load existing FAISS vectors if available
    if os.path.exists(faiss_index_path):
        vectors = FAISS.load_local(faiss_index_path, model, allow_dangerous_deserialization=True)

    # Load processed files and URLs
    processed_files = pickle.load(open(processed_file_path, "rb")) if os.path.exists(processed_file_path) else set()
    processed_urls = pickle.load(open(processed_urls_path, "rb")) if os.path.exists(processed_urls_path) else set()

    all_pdfs = set(os.listdir(pdf_folder)) if os.path.exists(pdf_folder) else set()
    new_pdfs = all_pdfs - processed_files
    new_docs = []

    # Define URLs to be scraped
    urls = [
        "https://terotam.com/about-us", "https://terotam.com/asset-management-solution", 
        "https://terotam.com/inventory-management-solution", "https://terotam.com/staff-management-solution", 
        "https://terotam.com/facility-management-solution", "https://terotam.com/asset-tracking-solution", 
        "https://terotam.com/permit-to-work-solution", "https://terotam.com/enquiry-management-solution", 
        "https://terotam.com/preventive-maintenance-solution", "https://terotam.com/budget-expense-management-solution", 
        "https://terotam.com/property-management-solution", "https://terotam.com/customer-management-solution", 
        "https://terotam.com/task-management-solution", "https://terotam.com/project-management-solution", 
        "https://terotam.com/issue-tracking-management-solution", "https://terotam.com/workflow-management-solution", 
        "https://terotam.com/hrms-solution", "https://terotam.com/escalation-management-solution", 
        "https://terotam.com/store-condition-assessment-solution", "https://terotam.com/location-management-solution", 
        "https://terotam.com/iot-predictive-maintenance", "https://terotam.com/procurement-management-solution", 
        "https://terotam.com/document-management-solution", "https://terotam.com/vendor-management-solution", 
        "https://terotam.com/purchase-order-solution", "https://terotam.com/purchase-requisition-solution", 
        "https://terotam.com/request-for-quotation", "https://terotam.com/annual-rate-contract-management", 
        "https://terotam.com/food-beverage"
    ]
    
    # Automatically detect new URLs for scraping
    new_urls = [url for url in urls if url not in processed_urls]

    # Scrape new URLs if any
    if new_urls:
        web_loader = WebBaseLoader(
            new_urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=["wpb_wrapper", "main-head col-white", "banner-content col-white"])
            )
        )
        web_docs = web_loader.load()

        if web_docs:
            new_docs.extend(web_docs)
            with open(scraped_data_path, "a", encoding="utf-8") as f:  # Append new data
                f.write("\n\n".join([doc.page_content for doc in web_docs]))

            st.write(f"New web content saved for {len(new_urls)} URLs.")
            processed_urls.update(new_urls)  # Update only after successful scraping
        else:
            st.write("No new data scraped or URLs were inaccessible.")

    # Process new PDF documents
    if new_pdfs:
        loader = PyPDFDirectoryLoader(pdf_folder)
        all_docs = loader.load()
        new_docs.extend([doc for doc in all_docs if os.path.basename(doc.metadata.get("source", "")) in new_pdfs])

    # Process vectors if new data is found
    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        final_docs = splitter.split_documents(new_docs)
        doc_texts = [doc.page_content for doc in final_docs]
        new_vectors = FAISS.from_texts(doc_texts, model)

        if vectors:
            vectors.merge_from(new_vectors)
        else:
            vectors = new_vectors

        # Save the updated vector index and processed files
        vectors.save_local(faiss_index_path)
        pickle.dump(processed_files | new_pdfs, open(processed_file_path, "wb"))
        pickle.dump(processed_urls, open(processed_urls_path, "wb"))

    return vectors, model

# Load Vectors if Not Already Loaded
if st.session_state["vectors"] is None:
    st.session_state["vectors"], st.session_state["embeddings"] = load_process_documents()

memory = MemorySaver()

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.

    Ensure that for any variations of the question, the response is structured exactly the same way.

    Provide the response in this format:
    1.
    2. 
    3. 
    ...

    <context>
    {context}
    </context>

    User's Question (Corrected if necessary): 
    {input}

    Answer:
    """
)

def call_model(state: MessagesState):
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    # Greeting keywords
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

    # Sensitive or unethical keywords
    sensitive_keywords = {"hack", "bypass", "breach"}

    # Standard fallback response
    fallback_response = (
        "I'm sorry, but I couldn't find relevant information for your query. "
        "For more details, please contact TeroTAM Technical Support at +91 93281 35112, "
        "email contact@terotam.com, or visit our website at https://terotam.com."
    )

    # Ethical warning for sensitive queries
    ethical_warning = (
        "I'm sorry, but I can't assist with that. Attempting to hack or bypass security measures on any system "
        "is illegal and unethical. Always follow proper procedures and guidelines."
    )

    # Extract user query
    query = state['messages'][-1].content.lower() if state['messages'] else ""

    # Respond to greetings
    if any(greet in query for greet in greetings):
        response_text = (
            "Hello! Welcome to TeroTAM. I'm here to assist you with any queries or information you need. "
            "Feel free to ask your questions. For further assistance, contact TeroTAM Technical Support at "
            "+91 93281 35112, contact@terotam.com, or visit our website, https://terotam.com."
        )
    # Respond to sensitive or unethical queries
    elif any(sensitive in query for sensitive in sensitive_keywords):
        response_text = ethical_warning
    else:
        # Perform similarity search with FAISS
        retrieved_docs = st.session_state["vectors"].similarity_search_with_score(query, k=10)

        # Respond if similarity score is above threshold
        if retrieved_docs and retrieved_docs[0][1] > 0.3:
            context = retrieved_docs[0][0].page_content
            formatted_prompt = prompt.format(context=context, input=query)
            messages = [HumanMessage(content=formatted_prompt)]
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, "content") else fallback_response
        else:
            # Fallback for out-of-context queries
            response_text = fallback_response

    return {"messages": [AIMessage(content=response_text)]}

# Define LangGraph Workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
app = workflow.compile(checkpointer=memory)

# Sidebar with Predefined Questions
with st.sidebar:
    st.header("Predefined Questions")
    predefined_questions = [
        "What is TeroTAM?",
        "How does TeroTAM help with asset management?",
        "What are the features of TeroTAM's inventory management solution?",
        "Can you explain TeroTAM's facility management solution?",
        "How does TeroTAM handle preventive maintenance?",
        "How to create complaint?"
    ]

    # Display buttons for each predefined question
    for question in predefined_questions:
        if st.button(question):
            # Simulate user input by setting the question as the user_input
            user_input = question

# Display Chat Messages
for message in st.session_state["messages"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Handle Predefined Questions
if 'user_input' in locals():  # Check if a predefined question was clicked
    user_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        st.write(user_input)

    result = app.invoke(
        {"messages": st.session_state["messages"]},
        config={"configurable": {
            "thread_id": st.session_state["thread_id"],
            "checkpoint_ns": "chatbot_ns",
            "checkpoint_id": "chat_checkpoint"
        }}
    )

    bot_messages = result.get("messages", [])
    bot_response = bot_messages[-1].content if bot_messages else "I couldn't process your request."
    ai_msg = AIMessage(content=bot_response)
    st.session_state["messages"].append(ai_msg)

    with st.chat_message("assistant"):
        st.write(bot_response)

    st.rerun()

# Handle Manual User Input
user_input_manual = st.chat_input("Enter your message...")

if user_input_manual:
    user_msg = HumanMessage(content=user_input_manual)
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        st.write(user_input_manual)

    result = app.invoke(
        {"messages": st.session_state["messages"]},
        config={"configurable": {
            "thread_id": st.session_state["thread_id"],
            "checkpoint_ns": "chatbot_ns",
            "checkpoint_id": "chat_checkpoint"
        }}
    )

    bot_messages = result.get("messages", [])
    bot_response = bot_messages[-1].content if bot_messages else "I couldn't process your request."
    ai_msg = AIMessage(content=bot_response)
    st.session_state["messages"].append(ai_msg)

    with st.chat_message("assistant"):
        st.write(bot_response)

    st.rerun()