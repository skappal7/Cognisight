# 🤖 AI Data Analyst Agent

A sophisticated, AI-powered data analysis chatbot built with Streamlit, Polars, DuckDB, and multiple LLM providers. This agent acts like a seasoned data analyst, providing intricate analysis, professional visualizations, and actionable insights through a conversational interface.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31+-red.svg)
![Polars](https://img.shields.io/badge/polars-0.20+-orange.svg)
![DuckDB](https://img.shields.io/badge/duckdb-0.10+-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**[🚀 Live Demo](https://share.streamlit.io)** | **[📖 Documentation](./GETTING_STARTED.md)** | **[💬 Discussions](#)** | **[🐛 Report Bug](#)** | **[✨ Request Feature](#)**

## ✨ Features

### 🎯 Core Capabilities
- **Conversational Data Analysis**: Chat with your data like you're talking to a senior analyst
- **Autonomous Agentic Reasoning**: ReAct-style agent loop for complex multi-step analysis
- **Smart Context Management**: Uses data summaries and pivots instead of raw data for token efficiency
- **Professional Visualizations**: Publication-ready charts and graphs using Plotly
- **Multi-File Support**: Analyze multiple datasets simultaneously with relationship discovery
- **Session Management**: Save, load, and download analysis sessions
- **Intelligent Caching**: Reduces redundant LLM calls and speeds up analysis

### 🔧 Technical Features
- **High-Performance Data Processing**: Polars + DuckDB for lightning-fast analytics
- **Flexible LLM Support**:
  - OpenRouter (Free models: Gemini Flash, Llama 3.2, Qwen)
  - OpenRouter (Paid models: Claude Sonnet, GPT-4o, Gemini Pro, DeepSeek)
  - Local models via Ollama
  - Local models via LM Studio
- **Token & Cost Tracking**: Real-time monitoring of token usage and costs
- **Downloadable Results**: Export visualizations and data as HTML, CSV, or JSON
- **Parquet Support**: Efficient data storage and processing

## 📁 Project Structure

```
data_analyst_agent/
├── app.py                      # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── agent.py               # Agentic reasoning engine (ReAct loop)
│   ├── data_manager.py        # Polars + DuckDB data management
│   ├── llm_client.py          # Unified LLM client (OpenRouter, Ollama, LM Studio)
│   └── tools.py               # Tool registry (analysis & visualization tools)
├── utils/
│   ├── __init__.py
│   ├── cache_manager.py       # Smart caching and session management
│   └── visualizer.py          # Plotly visualization generator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .cache/                    # Cache directory (auto-created)
    ├── queries/               # Cached query results
    ├── sessions/              # Saved sessions
    └── data/                  # Cached data summaries
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone or download this repository**:
```bash
# If you have the code
cd data_analyst_agent
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Configuration

#### Option 1: OpenRouter (Easiest for beginners)

1. Sign up at [OpenRouter](https://openrouter.ai)
2. Get your API key from the dashboard
3. Choose between:
   - **Free models**: No cost, great for testing
   - **Paid models**: Better quality, costs vary by model

#### Option 2: Local Models (Free, but requires setup)

**Using Ollama:**
1. Install [Ollama](https://ollama.ai)
2. Pull a model:
```bash
ollama pull llama3.2:3b
# or
ollama pull deepseek-r1:7b
```
3. Ollama will run on `http://localhost:11434`

**Using LM Studio:**
1. Download [LM Studio](https://lmstudio.ai)
2. Download a model from the interface
3. Start the local server (usually `http://localhost:1234`)

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Usage Guide

### 1. Configure LLM

In the sidebar:
1. Select your LLM provider
2. Enter API key (if using OpenRouter)
3. Choose a model
4. Click "Initialize LLM"

### 2. Upload Data

Supported formats:
- CSV (`.csv`)
- Parquet (`.parquet`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)

Simply drag and drop files into the upload area in the sidebar.

### 3. Start Analyzing

Type your questions in natural language:

**Example Queries:**
```
"Give me an overview of all the data"
"What are the top 10 customers by revenue?"
"Show me sales trends over the last 6 months"
"Identify any anomalies in the transaction amounts"
"Create a correlation heatmap for numeric columns"
"Compare average order values across different regions"
"What insights can you provide about customer churn?"
```

### 4. Quick Actions

Use the quick action buttons for common analyses:
- 📊 **Data Overview**: Comprehensive summary of all datasets
- 📈 **Trend Analysis**: Identify and visualize trends
- 🔍 **Anomaly Detection**: Find outliers and anomalies
- 💡 **Insights**: Generate key insights and recommendations

### 5. Download Results

- **Visualizations**: Click the download button below each chart to save as HTML
- **Data Tables**: Export to CSV with one click
- **Session**: Download entire session including chat history and results as JSON

## 🛠️ Advanced Features

### Intelligent Caching

The agent automatically caches:
- Query results (24-hour expiration)
- Data summaries
- Session state

Benefits:
- Faster responses for repeated queries
- Reduced API costs
- Offline access to previous analyses

### Session Management

**Save Session**:
```
Sidebar → Cache Management → Save Session
```

**Download Session**:
```
Sidebar → Cache Management → Download Session
```

Session includes:
- Full chat history
- Token usage
- Analysis results
- Metadata

### Token Tracking

Monitor your usage in real-time:
- Total tokens used
- Estimated cost (for paid models)
- Per-query breakdown

## 📊 Available Analysis Tools

The agent has access to these tools:

### Data Analysis
- `get_data_summary`: Statistical summaries
- `execute_sql`: Run SQL queries via DuckDB
- `filter_data`: Apply filters and conditions
- `aggregate_data`: Group and aggregate
- `get_column_stats`: Detailed column analysis
- `get_correlation`: Correlation matrices
- `detect_outliers`: IQR-based outlier detection

### Visualizations
- `create_line_chart`: Time series and trends
- `create_bar_chart`: Categorical comparisons
- `create_scatter_plot`: Relationship analysis
- `create_histogram`: Distribution analysis
- `create_box_plot`: Statistical distribution
- `create_heatmap`: Correlation visualization
- `create_pie_chart`: Composition analysis

## 🎨 Customization

### Modify System Prompt

Edit `core/agent.py` to customize the agent's behavior:
```python
self.system_prompt = """Your custom instructions here..."""
```

### Add Custom Tools

Add new tools in `core/tools.py`:
```python
def my_custom_tool(self, ...):
    """Tool description"""
    # Implementation
    return result
```

Register in `_register_tools()`:
```python
'my_custom_tool': {
    'function': self.my_custom_tool,
    'description': '...',
    'parameters': {...}
}
```

### Adjust Cache Settings

Modify cache behavior in `utils/cache_manager.py`:
```python
# Change cache expiration (default: 24 hours)
max_age_hours=48  # 2 days
```

## 🐛 Troubleshooting

### Common Issues

**"Please initialize the LLM first"**
- Click "Initialize LLM" in the sidebar after entering your API key

**"Error loading file"**
- Ensure file format is supported (CSV, Parquet, Excel, JSON)
- Check for encoding issues (use UTF-8)

**"Error executing query"**
- Verify column names match your data
- Check SQL syntax if using execute_sql

**Ollama connection error**
- Ensure Ollama is running: `ollama serve`
- Check the port (default: 11434)

**Slow performance**
- Use Parquet format for large datasets
- Enable caching (it's on by default)
- Consider using local models for frequent queries

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization types
- More sophisticated anomaly detection
- Natural language to SQL conversion
- Report generation (PDF/DOCX)
- Multi-language support

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) - Web framework
- [Polars](https://www.pola.rs) - Fast DataFrame library
- [DuckDB](https://duckdb.org) - In-memory analytics
- [Plotly](https://plotly.com) - Interactive visualizations
- [OpenRouter](https://openrouter.ai) - LLM API gateway

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Analyzing! 📊🤖**
