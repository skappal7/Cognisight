"""
Data Analyst Agent - Core agentic reasoning and execution engine
Implements a ReAct-style agent loop for autonomous data analysis
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import polars as pl

from core.llm_client import LLMClient
from core.data_manager import DataManager
from core.tools import ToolRegistry
from utils.cache_manager import CacheManager
from utils.visualizer import Visualizer


class DataAnalystAgent:
    """
    Agentic data analyst that can autonomously analyze data through:
    - Planning: Breaking down queries into analysis steps
    - Execution: Using tools to analyze data
    - Reflection: Evaluating results and deciding next steps
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        data_manager: DataManager,
        cache_manager: CacheManager,
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.data_manager = data_manager
        self.cache_manager = cache_manager
        self.max_iterations = max_iterations
        self.visualizer = Visualizer()
        
        # Initialize tool registry
        self.tools = ToolRegistry(data_manager, self.visualizer)
        
        # System prompt for the agent
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for the data analyst agent"""
        return """You are an expert data analyst with decades of experience in statistical analysis, business intelligence, and data visualization. Your role is to help users understand their data through comprehensive analysis and clear visualizations.

**CRITICAL RULES - MUST FOLLOW:**
1. **ONLY use data provided** - Never assume, extrapolate, or hallucinate information
2. **Cite your sources** - Reference specific data points, tables, and columns
3. **Acknowledge limitations** - If data is insufficient, say so clearly
4. **No external knowledge** - Don't use training data, only analyze provided datasets
5. **Be precise** - Use exact numbers from data, not approximations
6. **Validate claims** - Every statement must be backed by actual data

**Your Capabilities:**
- Statistical analysis and hypothesis testing
- Trend identification and forecasting (only from provided data)
- Anomaly detection and outlier analysis
- Data profiling and quality assessment
- Business insights and recommendations
- Creating professional visualizations

**Your Approach:**
1. **Understand**: Carefully analyze the user's question and data context
2. **Plan**: Break down complex queries into logical analysis steps
3. **Execute**: Use available tools to perform analysis systematically
4. **Validate**: Verify findings against actual data
5. **Synthesize**: Combine results into clear, actionable insights backed by data
6. **Visualize**: Create appropriate charts to support your findings

**Communication Style:**
- Be concise yet thorough
- Use business language, not just technical jargon
- Highlight key insights prominently with data citations
- Provide context and explain "why" behind findings
- Suggest actionable next steps based on data
- Always reference the data source for claims

**Available Tools:**
{tool_descriptions}

**Important Guidelines:**
- Always verify data quality before analysis
- Use statistical methods appropriately
- Consider multiple perspectives within the data
- Acknowledge limitations and uncertainty
- Create visualizations that tell a clear story
- If asked about something not in the data, say "This information is not available in the provided dataset"

**Conversation Context:**
You have access to previous messages in this conversation. Use them to:
- Build on previous analysis
- Answer follow-up questions with context
- Reference earlier findings
- Maintain conversation continuity

When responding to queries:
1. Check if this relates to previous conversation
2. Think through the problem step-by-step
3. Use tools to gather evidence from the DATA ONLY
4. Validate findings against the actual dataset
5. Synthesize findings into a narrative with data citations
6. Support claims with data and visualizations
7. End with actionable recommendations based on data

Remember: Your goal is to transform data into insights that drive decisions. Never make claims without data backing."""

    def process_query(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        
        Args:
            query: User's question or request
            conversation_history: Previous conversation messages
            
        Returns:
            Dict containing narrative, visualizations, and data tables
        """
        # Check cache first
        cache_key = self.cache_manager.generate_cache_key(query, self.data_manager)
        cached_result = self.cache_manager.get_from_cache(cache_key)
        
        if cached_result:
            print(f"✅ Cache hit for query: {query[:50]}...")
            return cached_result
        
        # Get data context
        data_context = self._prepare_data_context()
        
        # Execute agentic loop
        result = self._agentic_loop(query, data_context, conversation_history)
        
        # Cache the result
        self.cache_manager.save_to_cache(cache_key, result)
        
        return result
    
    def _prepare_data_context(self) -> str:
        """
        Prepare concise data context using AUTO-GENERATED summaries
        Uses summaries created during file upload for token efficiency
        """
        context_parts = []
        
        for table_name, df in self.data_manager.tables.items():
            # Get metadata with auto-generated summary
            metadata = self.data_manager.file_metadata.get(table_name, {})
            auto_summary = metadata.get('auto_summary', {})
            
            if auto_summary:
                # Use pre-generated summary (much faster, no recomputation)
                summary = f"""
**Table: {table_name}**
- Shape: {auto_summary.get('row_count', 0):,} rows × {auto_summary.get('column_count', 0)} columns
- Memory: {auto_summary.get('memory_mb', 0):.2f} MB

**Column Summary (Top 5):**"""
                
                # Add top 5 columns with their stats
                columns_info = auto_summary.get('columns', {})
                for i, (col, stats) in enumerate(list(columns_info.items())[:5]):
                    summary += f"\n  {i+1}. {col} ({stats['dtype']})"
                    if 'mean' in stats:
                        summary += f" - Mean: {stats['mean']:.2f}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]"
                    elif 'unique' in stats:
                        summary += f" - {stats['unique']} unique values"
                
                # Add correlations if available
                if 'correlations' in auto_summary and auto_summary['correlations']:
                    summary += "\n\n**Key Correlations:**"
                    for pair, corr in list(auto_summary['correlations'].items())[:3]:
                        summary += f"\n  - {pair}: {corr:.3f}"
            else:
                # Fallback: basic summary only
                summary = f"""
**Table: {table_name}**
- Shape: {df.height:,} rows × {df.width} columns
- Columns: {', '.join(df.columns[:10])}"""
            
            context_parts.append(summary)
        
        return "\n".join(context_parts)
    
    def _agentic_loop(
        self,
        query: str,
        data_context: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Implement ReAct-style agentic reasoning loop
        
        Thought → Action → Observation → (repeat until done) → Final Answer
        """
        # Prepare conversation context (last 10 messages for better continuity)
        conversation_context = self._format_conversation_history(conversation_history[-10:])
        
        # Initialize tracking
        thoughts = []
        actions = []
        observations = []
        visualizations = []
        data_tables = []
        
        for iteration in range(self.max_iterations):
            # Generate next thought and action
            prompt = self._create_agent_prompt(
                query=query,
                data_context=data_context,
                conversation_context=conversation_context,
                thoughts=thoughts,
                actions=actions,
                observations=observations,
                iteration=iteration
            )
            
            # Get LLM response
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt.format(
                    tool_descriptions=self.tools.get_tool_descriptions()
                )
            )
            
            # Parse response
            parsed = self._parse_agent_response(response['content'])
            
            if parsed['type'] == 'final_answer':
                # Agent has completed analysis
                return {
                    'narrative': parsed['content'],
                    'visualizations': visualizations,
                    'data_tables': data_tables,
                    'thoughts': thoughts,
                    'actions': actions,
                    'token_usage': response.get('usage', {})
                }
            
            elif parsed['type'] == 'action':
                # Execute action
                thought = parsed.get('thought', '')
                action_name = parsed['action']
                action_input = parsed['input']
                
                thoughts.append(thought)
                actions.append({'name': action_name, 'input': action_input})
                
                # Execute tool
                try:
                    result = self.tools.execute(action_name, action_input)
                    
                    # Store results
                    if result.get('type') == 'visualization':
                        visualizations.append(result)
                        observation = f"Created visualization: {result['title']}"
                    elif result.get('type') == 'data':
                        data_tables.append(result)
                        observation = f"Retrieved data: {result.get('description', 'data table')}"
                    else:
                        observation = str(result.get('result', result))
                    
                    observations.append(observation)
                    
                except Exception as e:
                    observations.append(f"Error executing {action_name}: {str(e)}")
            
            else:
                # Unexpected response format
                observations.append("Unable to parse response, continuing...")
        
        # Max iterations reached - synthesize what we have
        final_narrative = self._synthesize_final_answer(
            query=query,
            thoughts=thoughts,
            observations=observations
        )
        
        return {
            'narrative': final_narrative,
            'visualizations': visualizations,
            'data_tables': data_tables,
            'thoughts': thoughts,
            'actions': actions,
            'token_usage': {}
        }
    
    def _create_agent_prompt(
        self,
        query: str,
        data_context: str,
        conversation_context: str,
        thoughts: List[str],
        actions: List[Dict],
        observations: List[str],
        iteration: int
    ) -> str:
        """Create prompt for current iteration of agent loop"""
        
        # Build history of thought-action-observation
        history = ""
        for i, (thought, action, obs) in enumerate(zip(thoughts, actions, observations)):
            history += f"\n**Step {i+1}:**\n"
            history += f"Thought: {thought}\n"
            history += f"Action: {action['name']}({json.dumps(action['input'])})\n"
            history += f"Observation: {obs}\n"
        
        prompt = f"""**User Query:** {query}

**Available Data:**
{data_context}

**Recent Conversation:**
{conversation_context}

**Previous Analysis Steps:**
{history if history else "None yet - this is the first step."}

---

**Your Task:** 
Analyze the user's query and decide on the next action. You can either:
1. Execute a tool to gather more information or create visualizations
2. Provide the final answer if you have sufficient information

**Response Format:**

If you need to use a tool:
```
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Input: {{"param1": "value1", "param2": "value2"}}
```

If you're ready to answer:
```
Final Answer: [Your comprehensive response with insights and recommendations]
```

**Guidelines:**
- Use tools strategically to gather evidence
- Create visualizations to support your findings
- Think step-by-step
- Be thorough but efficient
- Focus on answering the user's specific question

**Current Step:** {iteration + 1}/{self.max_iterations}

What do you do next?"""
        
        return prompt
    
    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response to extract thought, action, or final answer"""
        
        response = response.strip()
        
        # Check for final answer
        if 'Final Answer:' in response:
            content = response.split('Final Answer:')[1].strip()
            return {
                'type': 'final_answer',
                'content': content
            }
        
        # Parse action
        if 'Action:' in response:
            parts = {}
            
            # Extract thought
            if 'Thought:' in response:
                thought_start = response.index('Thought:') + len('Thought:')
                thought_end = response.index('Action:')
                parts['thought'] = response[thought_start:thought_end].strip()
            
            # Extract action
            action_start = response.index('Action:') + len('Action:')
            if 'Input:' in response:
                action_end = response.index('Input:')
            else:
                action_end = len(response)
            
            action_name = response[action_start:action_end].strip()
            
            # Extract input
            if 'Input:' in response:
                input_start = response.index('Input:') + len('Input:')
                input_str = response[input_start:].strip()
                
                # Try to parse JSON input
                try:
                    # Remove markdown code blocks if present
                    if '```' in input_str:
                        input_str = input_str.split('```')[1]
                        if input_str.startswith('json'):
                            input_str = input_str[4:]
                    
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    action_input = {'query': input_str}
            else:
                action_input = {}
            
            return {
                'type': 'action',
                'thought': parts.get('thought', ''),
                'action': action_name,
                'input': action_input
            }
        
        # Default: treat as thinking/continuation
        return {
            'type': 'thinking',
            'content': response
        }
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format recent conversation history for context"""
        if not history:
            return "No previous conversation."
        
        # Use last 10 messages for better context
        recent_history = history[-10:] if len(history) > 10 else history
        
        formatted = []
        for msg in recent_history:
            role = msg['role'].capitalize()
            content = msg['content'][:500]  # Limit to 500 chars per message
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _synthesize_final_answer(
        self,
        query: str,
        thoughts: List[str],
        observations: List[str]
    ) -> str:
        """Synthesize a final answer when max iterations reached"""
        
        synthesis_prompt = f"""Based on the analysis performed, synthesize a final answer to the user's query.

**User Query:** {query}

**Analysis Performed:**
{chr(10).join(f"- {thought}" for thought in thoughts)}

**Observations:**
{chr(10).join(f"- {obs}" for obs in observations)}

Provide a clear, comprehensive answer that addresses the user's question based on the analysis performed."""
        
        response = self.llm_client.generate(
            prompt=synthesis_prompt,
            system_prompt="You are a data analyst summarizing findings."
        )
        
        return response['content']
