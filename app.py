"""
AI Data Analyst Agent - Streamlit Chat Interface
A sophisticated data analysis chatbot using Polars, DuckDB, and LLMs
"""

import streamlit as st
import polars as pl
import duckdb
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go

# Import our custom modules
from core.agent import DataAnalystAgent
from core.data_manager import DataManager
from core.llm_client import LLMClient
from utils.cache_manager import CacheManager
from utils.visualizer import Visualizer
from utils.report_generator import ReportGenerator

# Branding Configuration
LOGO_URL = "https://raw.githubusercontent.com/skappal7/TextAnalyser/refs/heads/main/logo.png"
FOOTER = "Developed with Streamlit with 💗 by CE Team Innovation Lab 2025"
APP_TITLE = "COGNiNSIGHTS"

# Page configuration
st.set_page_config(
    page_title="COGNiNSIGHTS - AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with Sutherland branding
st.markdown("""
<style>
    /* Sutherland Brand Colors */
    :root {
        --sutherland-navy: #2C3E50;
        --sutherland-pink: #E91E63;
        --sutherland-light: #F8F9FA;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        padding: 1rem 0 0.5rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    .logo-container img {
        max-width: 300px;
        height: auto;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--sutherland-navy);
        text-align: center;
        margin-bottom: 0.5rem;
        animation: slideDown 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: fadeIn 1.2s ease-in;
    }
    
    /* Button styling with Sutherland colors */
    .stButton>button {
        width: 100%;
        background-color: var(--sutherland-pink);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        margin: 0.2rem 0;
    }
    
    .stButton>button:hover {
        background-color: #C2185B;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(233, 30, 99, 0.3);
    }
    
    /* Fix button container alignment */
    .stButton {
        text-align: center;
    }
    
    /* Column alignment for buttons */
    [data-testid="column"] {
        display: flex;
        align-items: stretch;
    }
    
    [data-testid="column"] .stButton {
        width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--sutherland-light);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, var(--sutherland-navy) 0%, #34495E 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.6s ease-out;
    }
    
    /* Chat message bubbles - Modern ChatGPT/Chainlit style */
    .user-message {
        background: linear-gradient(135deg, var(--sutherland-pink) 0%, #D81B60 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.8rem 0 0.8rem auto;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(233, 30, 99, 0.2);
        animation: messageSlideLeft 0.4s ease-out;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: var(--sutherland-navy);
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.8rem auto 0.8rem 0;
        max-width: 75%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        animation: messageSlideRight 0.4s ease-out;
        word-wrap: break-word;
    }
    
    .message-container {
        display: flex;
        flex-direction: column;
        margin: 0.5rem 0;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0 0.5rem;
    }
    
    .user-avatar {
        background: var(--sutherland-pink);
        color: white;
    }
    
    .assistant-avatar {
        background: var(--sutherland-navy);
        color: white;
    }
    
    .message-row {
        display: flex;
        align-items: flex-end;
        margin: 0.5rem 0;
    }
    
    .user-row {
        justify-content: flex-end;
    }
    
    .assistant-row {
        justify-content: flex-start;
    }
    
    .message-timestamp {
        font-size: 0.75rem;
        color: #999;
        margin: 0.2rem 0.5rem;
    }
    
    /* Thinking indicator */
    .thinking-indicator {
        display: inline-block;
        padding: 0.8rem 1.2rem;
        background: #f0f2f6;
        border-radius: 18px;
        margin: 0.5rem 0;
    }
    
    .thinking-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--sutherland-pink);
        margin: 0 2px;
        animation: thinking 1.4s ease-in-out infinite;
    }
    
    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes thinking {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-10px); opacity: 1; }
    }
    
    @keyframes messageSlideLeft {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes messageSlideRight {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Input box styling */
    .stTextInput input {
        border-radius: 24px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem 1.2rem;
        font-size: 0.95rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: var(--sutherland-pink);
        box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.1);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--sutherland-pink);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #C2185B;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Action card buttons - Pink cards */
    .action-card-pink {
        background: linear-gradient(135deg, #E91E63 0%, #D81B60 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(233, 30, 99, 0.3);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .action-card-pink:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(233, 30, 99, 0.4);
    }
    
    /* Action card buttons - Navy cards */
    .action-card-navy {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(44, 62, 80, 0.3);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .action-card-navy:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(44, 62, 80, 0.4);
    }
    
    .action-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.8rem;
        letter-spacing: 0.5px;
    }
    
    .action-card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-nav {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #666;
        font-weight: 500;
    }
    
    .sidebar-nav-item:hover {
        background: white;
        color: var(--sutherland-navy);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .sidebar-nav-icon {
        margin-right: 0.8rem;
        font-size: 1.3rem;
    }
    
    /* App title styling */
    .app-title {
        font-size: 2rem;
        font-weight: 900;
        color: var(--sutherland-navy);
        letter-spacing: 1px;
        margin-bottom: 0;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #999;
        margin-top: 0.3rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--sutherland-navy);
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 2px solid var(--sutherland-pink);
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Cache info styling */
    .cache-info {
        font-size: 0.85rem;
        color: #888;
        font-style: italic;
        text-align: center;
        padding: 0.5rem;
    }
    
    /* Success/Error message animations */
    .stSuccess, .stError, .stWarning, .stInfo {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Dataframe styling */
    .dataframe {
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Selectbox with brand colors */
    .stSelectbox > div > div {
        border-color: var(--sutherland-pink);
    }
    
    /* Progress bar with brand color */
    .stProgress > div > div {
        background-color: var(--sutherland-pink);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'conversation_id' not in st.session_state:
        # Generate unique conversation ID
        st.session_state.conversation_id = hashlib.md5(
            str(datetime.now()).encode()
        ).hexdigest()[:8]
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'estimated_cost': 0.0
        }
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []

def sidebar_config():
    """Configure sidebar with LLM settings and file upload"""
    
    # LLM Configuration
    st.sidebar.markdown("### 🤖 LLM Configuration")
    
    # LLM Provider Selection
    llm_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ["OpenRouter (Free)", "Ollama (Local)", "LM Studio (Local)"],
        key="llm_provider"
    )
    
    # Provider-specific settings
    if "OpenRouter" in llm_provider:
        # Auto-load API key from secrets (hidden from user)
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        
        if not api_key:
            st.sidebar.error("⚠️ OPENROUTER_API_KEY not found in secrets. Add it in Streamlit Cloud settings.")
            return
        
        # Friendly display names for free models
        model_display_names = {
            "DeepSeek Chat V3": "deepseek/deepseek-chat-v3.1:free",
            "DeepSeek R1 70B": "deepseek/deepseek-r1-distill-llama-70b:free",
            "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
            "Qwen 32B Vision": "qwen/qwen2.5-vl-32b-instruct:free",
            "Qwen 235B": "qwen/qwen3-235b-a22b:free",
            "Mistral 7B": "mistralai/mistral-7b-instruct:free",
            "OpenChat 7B": "openchat/openchat-7b:free",
            "MythoMax 13B": "gryphe/mythomax-l2-13b:free",
            "GPT OSS 20B": "openai/gpt-oss-20b:free",
            "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
            "Kimi Vision 3B": "moonshotai/kimi-vl-a3b-thinking:free",
            "Kimi K2": "moonshotai/kimi-k2:free"
        }
        
        # Show friendly names in dropdown
        selected_display_name = st.sidebar.selectbox(
            "Select Model", 
            list(model_display_names.keys())
        )
        
        # Get actual model ID for API
        model = model_display_names[selected_display_name]
        base_url = "https://openrouter.ai/api/v1"
        
        # Auto-initialize LLM when model changes
        try:
            if st.session_state.llm_client is None or st.session_state.get('current_model') != model:
                st.session_state.llm_client = LLMClient(
                    provider=llm_provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
                st.session_state.agent = DataAnalystAgent(
                    llm_client=st.session_state.llm_client,
                    data_manager=st.session_state.data_manager,
                    cache_manager=st.session_state.cache_manager
                )
                st.session_state.current_model = model
                st.sidebar.success(f"✅ Ready: {selected_display_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
        
    else:  # Local models
        api_key = "local"
        base_url = st.sidebar.text_input(
            "Local API URL",
            value="http://localhost:11434" if "Ollama" in llm_provider else "http://localhost:1234/v1"
        )
        model = st.sidebar.text_input("Model Name", value="llama3.2:3b" if "Ollama" in llm_provider else "local-model")
        
        # Auto-initialize for local models
        if st.sidebar.button("Connect to Local Model", use_container_width=True):
            try:
                st.session_state.llm_client = LLMClient(
                    provider=llm_provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
                st.session_state.agent = DataAnalystAgent(
                    llm_client=st.session_state.llm_client,
                    data_manager=st.session_state.data_manager,
                    cache_manager=st.session_state.cache_manager
                )
                st.sidebar.success(f"✅ Connected: {model}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # File Upload Section
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload data files",
        type=['csv', 'parquet', 'xlsx', 'json'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.sidebar.spinner(f"📊 Generating summaries for {file.name}..."):
                    try:
                        df = st.session_state.data_manager.load_file(file)
                        st.session_state.uploaded_files[file.name] = {
                            'dataframe': df,
                            'upload_time': datetime.now(),
                            'file_size': file.size
                        }
                        st.sidebar.success(f"✅ {file.name} - Summaries ready!")
                    except Exception as e:
                        st.sidebar.error(f"❌ {file.name}: {str(e)}")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.sidebar.markdown("#### Loaded Files:")
        for filename, info in st.session_state.uploaded_files.items():
            df = info['dataframe']
            st.sidebar.markdown(
                f"**{filename}**\n"
                f"- Rows: {df.height:,}\n"
                f"- Columns: {df.width}\n"
                f"- Size: {info['file_size'] / 1024:.2f} KB"
            )
    
    st.sidebar.markdown("---")
    
    # Token Usage Display
    st.sidebar.markdown("### 💰 Token Usage")
    usage = st.session_state.token_usage
    st.sidebar.metric("Total Tokens", f"{usage['total_tokens']:,}")
    st.sidebar.metric("Estimated Cost", f"${usage['estimated_cost']:.4f}")
    
    # Cache Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Cache Management")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💾 Save Session", use_container_width=True):
            save_session()
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_cache()
    
    # Download session
    if st.sidebar.button("⬇️ Download Session", use_container_width=True):
        download_session()

def save_session():
    """Save current session to cache"""
    try:
        session_data = {
            'conversation_id': st.session_state.conversation_id,
            'messages': st.session_state.messages,
            'uploaded_files': {
                name: {
                    'shape': (info['dataframe'].height, info['dataframe'].width),
                    'columns': info['dataframe'].columns,
                    'upload_time': info['upload_time'].isoformat()
                }
                for name, info in st.session_state.uploaded_files.items()
            },
            'token_usage': st.session_state.token_usage,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.cache_manager.save_session(
            st.session_state.conversation_id,
            session_data
        )
        st.sidebar.success("✅ Session saved successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error saving session: {str(e)}")

def clear_cache():
    """Clear all cache"""
    st.session_state.cache_manager.clear_all()
    st.sidebar.success("✅ Cache cleared!")

def download_session():
    """Download session as JSON"""
    try:
        session_data = {
            'conversation_id': st.session_state.conversation_id,
            'messages': st.session_state.messages,
            'token_usage': st.session_state.token_usage,
            'timestamp': datetime.now().isoformat(),
            'analysis_results': st.session_state.analysis_results
        }
        
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json.dumps(session_data, indent=2),
            file_name=f"session_{st.session_state.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"❌ Error preparing download: {str(e)}")

def display_chat_history():
    """Display chat history with modern message bubbles like ChatGPT/Chainlit"""
    for idx, msg in enumerate(st.session_state.messages):
        is_user = msg['role'] == 'user'
        
        # Create message container
        if is_user:
            st.markdown(f"""
                <div class="message-row user-row">
                    <div class="user-message">
                        {msg['content']}
                    </div>
                    <div class="message-avatar user-avatar">U</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-row assistant-row">
                    <div class="message-avatar assistant-avatar">AI</div>
                    <div class="assistant-message">
                        {msg['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Display visualizations if present
        if 'visualizations' in msg and msg['visualizations']:
            for viz_idx, viz_data in enumerate(msg['visualizations']):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.plotly_chart(
                        viz_data['figure'],
                        use_container_width=True,
                        key=f"viz_{idx}_{viz_idx}"
                    )
                with col2:
                    st.download_button(
                        label="📥",
                        data=viz_data['figure'].to_html(),
                        file_name=f"{viz_data['title'].replace(' ', '_')}.html",
                        mime="text/html",
                        key=f"download_viz_{idx}_{viz_idx}",
                        help="Download visualization"
                    )
        
        # Display data tables if present
        if 'data_tables' in msg and msg['data_tables']:
            for table_idx, table_data in enumerate(msg['data_tables']):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.dataframe(
                        table_data['dataframe'],
                        use_container_width=True,
                        key=f"table_{idx}_{table_idx}"
                    )
                with col2:
                    csv_data = table_data['dataframe'].to_pandas().to_csv(index=False)
                    st.download_button(
                        label="📥",
                        data=csv_data,
                        file_name=f"{table_data.get('title', 'data')}.csv",
                        mime="text/csv",
                        key=f"download_table_{idx}_{table_idx}",
                        help="Download as CSV"
                    )

def process_user_input(user_input: str):
    """Process user input and generate response"""
    if not st.session_state.agent:
        st.error("⚠️ Please select a model from the sidebar!")
        return
    
    if not st.session_state.uploaded_files:
        st.warning("⚠️ No data files uploaded. Please upload data files to analyze.")
        return
    
    # Add user message to chat
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Show thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
        <div class="thinking-indicator">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # Process with agent
        response = st.session_state.agent.process_query(
            query=user_input,
            conversation_history=st.session_state.messages
        )
        
        # Remove thinking indicator
        thinking_placeholder.empty()
        
        # Add assistant response
        assistant_message = {
            'role': 'assistant',
            'content': response['narrative'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add visualizations if generated
        if response.get('visualizations'):
            assistant_message['visualizations'] = response['visualizations']
        
        # Add data tables if generated
        if response.get('data_tables'):
            assistant_message['data_tables'] = response['data_tables']
        
        st.session_state.messages.append(assistant_message)
        
        # Update token usage
        if 'token_usage' in response:
            for key in st.session_state.token_usage:
                st.session_state.token_usage[key] += response['token_usage'].get(key, 0)
        
        # Store analysis result for PDF generation
        st.session_state.analysis_results.append({
            'query': user_input,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-save session
        if len(st.session_state.messages) % 5 == 0:  # Save every 5 messages
            save_session()
        
    except Exception as e:
        thinking_placeholder.empty()
        st.error(f"❌ Error processing query: {str(e)}")
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            'timestamp': datetime.now().isoformat()
        })

def main():
    """Main application"""
    init_session_state()
    
    # Top bar with logo and title
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="app-subtitle">Your intelligent assistant for comprehensive data analysis and visualization</p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<img src="{LOGO_URL}" width="120">', unsafe_allow_html=True)
    
    # Sidebar
    sidebar_config()
    
    # Main chat area
    st.markdown("---")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    st.markdown("---")
    
    # Use columns for better layout
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me anything about your data...",
            key="user_input",
            placeholder="e.g., 'What are the top 10 customers by revenue?' or 'Show me sales trends over time'"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        process_user_input(user_input)
        st.rerun()
    
    # Action Cards (matching the image design)
    st.markdown("---")
    
    # Row 1 - Pink cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊\n\nDATA OVERVIEW", key="card_overview", use_container_width=True):
            process_user_input("Provide a comprehensive overview of all uploaded datasets")
            st.rerun()
    
    with col2:
        if st.button("📈\n\nTREND ANALYSIS", key="card_trends", use_container_width=True):
            process_user_input("Identify and visualize key trends in the data")
            st.rerun()
    
    with col3:
        if st.button("🔍\n\nANOMALY DETECTION", key="card_anomaly", use_container_width=True):
            process_user_input("Detect and explain any anomalies or outliers in the data")
            st.rerun()
    
    # Row 2 - Navy cards
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("💡\n\nINSIGHTS", key="card_insights", use_container_width=True):
            process_user_input("Generate key insights and recommendations based on the data")
            st.rerun()
    
    with col5:
        if st.button("📊\n\nSTATISTICAL ANALYSIS", key="card_stats", use_container_width=True):
            process_user_input("Perform detailed statistical analysis on the data")
            st.rerun()
    
    with col6:
        if st.button("💻\n\nCODING", key="card_coding", use_container_width=True):
            process_user_input("Generate Python code for data analysis and visualization")
            st.rerun()
    
    # PDF Report Download
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("#### 📄 Export Analysis")
        
        if st.button("📥 Download PDF Report", use_container_width=True, type="primary"):
            try:
                # Generate PDF report
                report_gen = ReportGenerator()
                
                # Get last analysis
                last_analysis = st.session_state.analysis_results[-1]
                
                # Get data summary
                data_summary = {}
                if st.session_state.uploaded_files:
                    first_file = list(st.session_state.uploaded_files.keys())[0]
                    df_info = st.session_state.uploaded_files[first_file]
                    data_summary = {
                        'row_count': df_info['dataframe'].height,
                        'column_count': df_info['dataframe'].width,
                        'memory_mb': df_info['file_size'] / (1024 * 1024)
                    }
                
                # Generate PDF
                pdf_bytes = report_gen.generate_report(
                    title="Data Analysis Report",
                    query=last_analysis['query'],
                    analysis_text=last_analysis['response']['narrative'],
                    data_summary=data_summary,
                    visualizations=last_analysis['response'].get('visualizations', []),
                    tables=last_analysis['response'].get('data_tables', [])
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=pdf_bytes,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p class="cache-info">💾 Session ID: {} | Auto-saved every 5 messages</p>'.format(
            st.session_state.conversation_id
        ),
        unsafe_allow_html=True
    )
    
    # Branded Footer
    st.markdown(f"""
        <div class="footer">
            {FOOTER}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
