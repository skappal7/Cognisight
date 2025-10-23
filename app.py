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
from streamlit_chat import message

# Import our custom modules
from core.agent import DataAnalystAgent
from core.data_manager import DataManager
from core.llm_client import LLMClient
from utils.cache_manager import CacheManager
from utils.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="COGNISIGHT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cache-info {
        font-size: 0.85rem;
        color: #888;
        font-style: italic;
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
    st.sidebar.markdown("### 🤖 LLM Configuration")
    
    # LLM Provider Selection
    llm_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ["OpenRouter (Free)", "OpenRouter (Paid)", "Ollama (Local)", "LM Studio (Local)"],
        key="llm_provider"
    )
    
    # Provider-specific settings
    if "OpenRouter" in llm_provider:
        api_key = st.sidebar.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai"
        )
        
        if "Free" in llm_provider:
            model_options = [
                "google/gemini-flash-1.5-8b",
                "meta-llama/llama-3.2-3b-instruct",
                "qwen/qwen-2-7b-instruct"
            ]
        else:
            model_options = [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o",
                "google/gemini-pro-1.5",
                "deepseek/deepseek-chat"
            ]
        
        model = st.sidebar.selectbox("Select Model", model_options)
        base_url = "https://openrouter.ai/api/v1"
        
    else:  # Local models
        api_key = "local"
        base_url = st.sidebar.text_input(
            "Local API URL",
            value="http://localhost:11434" if "Ollama" in llm_provider else "http://localhost:1234/v1"
        )
        model = st.sidebar.text_input("Model Name", value="llama3.2:3b" if "Ollama" in llm_provider else "local-model")
    
    # Initialize LLM Client
    if st.sidebar.button("Initialize LLM", use_container_width=True):
        if not api_key and "OpenRouter" in llm_provider:
            st.sidebar.error("Please provide an API key")
        else:
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
                st.sidebar.success(f"✅ LLM initialized: {model}")
            except Exception as e:
                st.sidebar.error(f"Error initializing LLM: {str(e)}")
    
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
                try:
                    # Load data using DataManager
                    df = st.session_state.data_manager.load_file(file)
                    st.session_state.uploaded_files[file.name] = {
                        'dataframe': df,
                        'upload_time': datetime.now(),
                        'file_size': file.size
                    }
                    st.sidebar.success(f"✅ Loaded: {file.name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading {file.name}: {str(e)}")
    
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
    """Display chat history with proper formatting"""
    for idx, msg in enumerate(st.session_state.messages):
        is_user = msg['role'] == 'user'
        
        # Display message
        message(
            msg['content'],
            is_user=is_user,
            key=f"msg_{idx}",
            avatar_style="avataaars" if is_user else "bottts"
        )
        
        # Display visualizations if present
        if 'visualizations' in msg and msg['visualizations']:
            for viz_idx, viz_data in enumerate(msg['visualizations']):
                st.plotly_chart(
                    viz_data['figure'],
                    use_container_width=True,
                    key=f"viz_{idx}_{viz_idx}"
                )
                
                # Add download button for visualization
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.download_button(
                        label="⬇️ Download",
                        data=viz_data['figure'].to_html(),
                        file_name=f"{viz_data['title'].replace(' ', '_')}.html",
                        mime="text/html",
                        key=f"download_viz_{idx}_{viz_idx}"
                    )
        
        # Display data tables if present
        if 'data_tables' in msg and msg['data_tables']:
            for table_idx, table_data in enumerate(msg['data_tables']):
                st.dataframe(
                    table_data['dataframe'],
                    use_container_width=True,
                    key=f"table_{idx}_{table_idx}"
                )
                
                # Add download button for data
                col1, col2 = st.columns([3, 1])
                with col2:
                    csv_data = table_data['dataframe'].to_pandas().to_csv(index=False)
                    st.download_button(
                        label="⬇️ CSV",
                        data=csv_data,
                        file_name=f"{table_data.get('title', 'data')}.csv",
                        mime="text/csv",
                        key=f"download_table_{idx}_{table_idx}"
                    )

def process_user_input(user_input: str):
    """Process user input and generate response"""
    if not st.session_state.agent:
        st.error("⚠️ Please initialize the LLM first from the sidebar!")
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
    with st.spinner("🤔 Analyzing..."):
        try:
            # Process with agent
            response = st.session_state.agent.process_query(
                query=user_input,
                conversation_history=st.session_state.messages
            )
            
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
            
            # Store analysis result
            st.session_state.analysis_results.append({
                'query': user_input,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Auto-save session
            if len(st.session_state.messages) % 5 == 0:  # Save every 5 messages
                save_session()
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                'timestamp': datetime.now().isoformat()
            })

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 AI Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Your intelligent assistant for comprehensive data analysis and visualization</p>',
        unsafe_allow_html=True
    )
    
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
    
    # Quick action buttons
    st.markdown("#### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Data Overview", use_container_width=True):
            process_user_input("Provide a comprehensive overview of all uploaded datasets")
            st.rerun()
    
    with col2:
        if st.button("📈 Trend Analysis", use_container_width=True):
            process_user_input("Identify and visualize key trends in the data")
            st.rerun()
    
    with col3:
        if st.button("🔍 Anomaly Detection", use_container_width=True):
            process_user_input("Detect and explain any anomalies or outliers in the data")
            st.rerun()
    
    with col4:
        if st.button("💡 Insights", use_container_width=True):
            process_user_input("Generate key insights and recommendations based on the data")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p class="cache-info">💾 Session ID: {} | Auto-saved every 5 messages</p>'.format(
            st.session_state.conversation_id
        ),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
