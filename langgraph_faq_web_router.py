import os
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from ddgs import DDGS
from langgraph.graph import StateGraph, END
import json


from qa_agent import get_qa_agent

load_dotenv()

class SupervisorState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    next_action: str
    final_answer: str

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.faq_agent = get_qa_agent()
        self.web_search = DDGS()
        
    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """Supervisor determines which agent to use"""
        
        last_message = state["messages"][-1].content.lower()

        # Use LLM for complex cases
        supervisor_prompt = f"""
        You are a supervisor who must choose between two agents:
        1. FAQ - for questions about Anzara loan app, online loans, credit, financial services
        2. WEB_search - for general questions, news, weather, facts that are NOT related to loans
        
        User asked: "{state['messages'][-1].content}"
        
        Analyze the query:
        - If question is about loans, credit, financial services, Anzara - choose FAQ
        - If question is about weather, news, general facts - choose WEB_search
        
        Answer with ONLY one word: FAQ or WEB_search
        """
        
        messages = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=state["messages"][-1].content)
        ]
        
        response = self.llm.invoke(messages)
        next_action = response.content.strip()
        
        # Check if response is valid
        if next_action not in ["FAQ", "WEB_search"]:
            next_action = "FAQ"  # Default for loan-related questions
        
        print(f"🧠 Supervisor chose: {next_action}")
        
        return {
            "messages": state["messages"],
            "next_action": next_action,
            "final_answer": ""
        }
    
    def faq_node(self, state: SupervisorState) -> SupervisorState:
        """FAQ agent handles questions about Anzara"""
        
        last_message = state["messages"][-1].content
        
        try:
            result = self.faq_agent.invoke(last_message)
            answer = result["result"]
        except Exception as e:
            answer = f"Sorry, an error occurred while processing your Anzara query: {str(e)}"
        
        return {
            "messages": state["messages"] + [AIMessage(content=answer)],
            "next_action": "END",
            "final_answer": answer
        }
    
    def web_search_node(self, state: SupervisorState) -> SupervisorState:
        """Web search for general questions"""
        
        last_message = state["messages"][-1].content
        
        try:
            # Use ddgs for search
            search_results = list(self.web_search.text(last_message, max_results=3))
            
            # Format text from results
            results_text = "\n".join([f"- {result['title']}: {result['body']}" for result in search_results])
            
            # Create response based on search results
            response_prompt = f"""
            User asked: "{last_message}"
            
            Search results:
            {results_text}
            
            Provide a helpful and structured answer based on the found information.
            """
            
            messages = [
                SystemMessage(content="You are an assistant that provides information based on web search results."),
                HumanMessage(content=response_prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
        except Exception as e:
            answer = f"Sorry, an error occurred during web search: {str(e)}"
        
        return {
            "messages": state["messages"] + [AIMessage(content=answer)],
            "next_action": "END", 
            "final_answer": answer
        }

def create_supervisor_graph():
    """Creates LangGraph with supervisor"""
    
    supervisor = SupervisorAgent()
    
    # Create graph
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor.supervisor_node)
    workflow.add_node("faq", supervisor.faq_node)
    workflow.add_node("web_search", supervisor.web_search_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_action"].lower(),
        {
            "faq": "faq",
            "web_search": "web_search"
        }
    )
    
    # Final transitions
    workflow.add_edge("faq", END)
    workflow.add_edge("web_search", END)
    
    return workflow.compile()

def main():
    """Main function for testing"""
    
    # Setup tracing (optional)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "anzara-supervisor"
    
    # Create graph
    app = create_supervisor_graph()
    
    print("=== Anzara Supervisor Agent ===")
    print("Available query types:")
    print("1. Questions about Anzara loan app (FAQ)")
    print("2. General questions (Web Search)")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("Your question: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        try:
            # Initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "next_action": "",
                "final_answer": ""
            }
            
            # Run graph
            result = app.invoke(initial_state)
            
            # Print result
            print(f"\n🤖 Answer: {result['final_answer']}")
            print(f"📊 Selected agent: {result['next_action']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()