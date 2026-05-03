import json
import math
import re
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import the memory manager
import sys
sys.path.append(str(Path(__file__).parent))
from scout_agent_binary import BinaryScoutAgent, NODE_TYPE, EDGE_TYPE

class NavigatorAgent:
    def __init__(self, index_path: str, content_path: str, llm_endpoint: str = "http://localhost:1234/v1/chat/completions", llm_model: str = "qwen3.5-0.8b"):
        """
        Initialize the Navigator Agent.
        
        Args:
            index_path: Path to memory_index.bin
            content_path: Path to memory_content.bin
            llm_endpoint: API endpoint for the LLM (e.g., Ollama, vLLM)
            llm_model: Name of the model to use (e.g., qwen3.5-08b)
        """
        self.memory = BinaryScoutAgent(index_path, content_path)
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.capsules = {} # (node_id, task_type) -> summary
        self.feedback_log = []

        # System Prompt for the Navigator
        self.system_prompt = """
You are the Navigator — a small, fast AI agent that operates over a binary memory map of a large codebase and SQL/database schema.

## Your world

The memory map contains two binary files:
- memory_index.bin: a graph index of every file, class, function, SQL table, column, view, schema, and ETL job in the codebase.
- memory_content.bin: the raw source text for each node.

The graph has these edge types:
- contains: parent → child
- calls: function A calls function B
- imports: file A imports file B
- references: table A references table B via foreign key
- feeds: view or ETL job reads from base table(s)

## Your tools

You have access to these functions. Call them by outputting a JSON tool call block.

1. rank_nodes(query, top_k)
   → Returns top_k nodes ranked by semantic + structural relevance to query string.

2. get_content(node_id)
   → Returns the raw source text for a specific node.

3. plan_path(start_id, target_id)
   → Returns BFS shortest path between two nodes.

4. get_callers(node_id, max_depth)
   → Returns all nodes that call or reference the given node, up to max_depth hops.

5. get_children(node_id)
   → Returns direct child nodes (e.g. columns of a table).

6. capsule_get(node_id, task_type)
   → Returns a cached latent summary for this node+task combination.

7. capsule_put(node_id, task_type, summary_text)
   → Stores a compressed summary for future reuse.

## Your task

Given a user's natural-language prompt, you must:

1. DECOMPOSE the prompt into 1–5 specific sub-questions.
2. PLAN which tools to call and in what order.
3. EXECUTE the tool calls in order.
4. ASSEMBLE a compact context package for the big LLM.

## Rules

- Never return raw content for more than 8 nodes to the big LLM — be selective.
- Always prefer capsule_get over re-reading content for entities you've seen before.
- If rank_nodes returns nodes with scores below 0.3, say so.
- For lineage questions (upstream/downstream tables), always trace and show the full path.
- For "what is this?" questions about tables or classes, always call get_children.
- Never invent relationships that are not in the graph.

## Output format

Return a JSON object:
{
  "sub_questions": ["...", "..."],
  "tool_calls_executed": [{"tool": "rank_nodes", "args": ["query", 5]}],
  "context_for_big_llm": {
    "relevant_nodes": [{"id": "...", "type": "...", "snippet": "..."}],
    "key_paths": [["nodeA", "nodeB"]],
    "structural_summary": "...",
    "caveats": "..."
  },
  "capsules_stored": [{"node_id": "...", "task_type": "..."}]
}
"""

    def invoke_llm(self, query: str) -> Dict:
        """Invoke the LLM to get the navigation plan."""
        # Construct the prompt for the chat API
        user_content = f"User Query: {query}\n\nNavigator Agent State:\n- Total Nodes: {len(self.memory.nodes)}\n\nPlease execute your plan:"
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }

        try:
            response = requests.post(self.llm_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Extract JSON from the response (handling potential markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
        except Exception as e:
            print(f"[Navigator] LLM invocation error: {e}")
            return self._fallback_logic(query)

    def _fallback_logic(self, query: str) -> Dict:
        """Deterministic fallback if LLM is unavailable."""
        ranked = self.memory.rank_nodes(query, top_k=5)
        relevant_nodes = []
        for nid, score in ranked[:3]:
            content = self.memory.get_node_content(nid)
            relevant_nodes.append({
                "id": nid,
                "type": self.memory.node_map[nid].get("type"),
                "snippet": content[:500] if content else ""
            })
        
        return {
            "sub_questions": [query],
            "tool_calls_executed": [{"tool": "rank_nodes", "args": [query, 5]}],
            "context_for_big_llm": {
                "relevant_nodes": relevant_nodes,
                "key_paths": [],
                "structural_summary": f"Found {len(relevant_nodes)} nodes via deterministic fallback.",
                "caveats": "LLM unavailable, using deterministic scoring."
            },
            "capsules_stored": []
        }

    def rank_nodes(self, query: str, top_k: int = 8) -> List[Dict]:
        """Rank nodes by semantic + structural relevance."""
        ranked = self.memory.rank_nodes(query, top_k)
        result = []
        for nid, score in ranked:
            node = self.memory.node_map.get(nid)
            if node:
                # Normalize caller_count/reference_count for SQL nodes
                node["caller_count"] = node.get("caller_count") or node.get("reference_count", 0)
                result.append({
                    "id": nid,
                    "type": node.get("type"),
                    "score": score,
                    "snippet": self.memory.get_node_content(nid) or ""
                })
        return result

    def get_content(self, node_id: str) -> Optional[str]:
        """Get raw source text for a node."""
        return self.memory.get_node_content(node_id)

    def plan_path(self, start_id: str, target_id: str) -> Optional[List[str]]:
        """BFS shortest path."""
        return self.memory.plan_path(start_id, target_id)

    def get_callers(self, node_id: str, max_depth: int = 3) -> List[str]:
        """Get callers up to max_depth."""
        return self.memory.get_callers(node_id, max_depth)

    def get_children(self, node_id: str) -> List[Dict]:
        """Get direct children."""
        children_ids = self.memory.adj.get(node_id, [])
        return [self.memory.node_map[cid] for cid in children_ids if cid in self.memory.node_map]

    def capsule_get(self, node_id: str, task_type: str) -> Optional[str]:
        """Get cached summary."""
        return self.capsules.get((node_id, task_type))

    def capsule_put(self, node_id: str, task_type: str, summary: str):
        """Store cached summary."""
        self.capsules[(node_id, task_type)] = summary

    def _column_lineage(self, table_name: str, column_name: str, direction: str = "both") -> List[Dict]:
        """
        Find lineage for a specific column in a table.
        Walk REFERENCES and FEEDS edges in both directions.
        """
        # This is a placeholder implementation - in a real system this would
        # traverse the actual graph structure to find column lineage
        return []

    def navigate(self, query: str) -> Dict:
        """Main navigation logic using the LLM."""
        # 1. Ask LLM to plan
        llm_result = self.invoke_llm(query)
        
        sub_questions = llm_result.get("sub_questions", [query])
        tool_calls = llm_result.get("tool_calls_executed", [])
        capsules_stored = llm_result.get("capsules_stored", [])
        
        # 2. Execute tool calls (if any were generated by the LLM)
        # Note: In a real implementation, you might parse the LLM output to actually call the tools
        # Here we assume the LLM *simulated* the tool calls and returned the context directly
        # to save latency, but we can also execute them if we want to be strict.
        
        # For this implementation, we trust the LLM's `context_for_big_llm` if it provided it.
        # If not, we generate it deterministically.
        
        # Handle specific query types that we know about
        if "tool_calls_executed" in llm_result:
            for call in llm_result["tool_calls_executed"]:
                tool = call.get("tool", "")
                if tool == "column_lineage":
                    # This would be implemented in a real version
                    pass
        
        if "context_for_big_llm" in llm_result:
            return llm_result
        
        # Fallback if LLM didn't provide context (shouldn't happen with good prompting)
        return self._fallback_logic(query)

# Usage
if __name__ == "__main__":
    # Example: Assuming LM Studio is running on localhost:1234
    # agent = NavigatorAgent(
    #     index_path=".context-tree/memory_index.bin",
    #     content_path=".context-tree/memory_content.bin",
    #     llm_endpoint="http://localhost:1234/v1/chat/completions",
    #     llm_model="qwen3.5-08b"
    # )
    
    # For testing without LLM, use fallback:
    agent = NavigatorAgent(".context-tree/memory_index.bin", ".context-tree/memory_content.bin")
    result = agent.navigate("Where is add defined?")
    print(json.dumps(result, indent=2))