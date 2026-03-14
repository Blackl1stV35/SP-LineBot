import os
from google import genai
from google.genai import types

def tool_scan_drive(user_id: str):
    """Trigger this when the user asks to scan, read, or sync their Google Drive."""
    return {"status": "success", "action_taken": "Triggered Drive Scan sequence"}

def tool_check_inventory(item_name: str):
    """Trigger this when the user asks about stock levels or who drew an item."""
    return {"status": "success", "search_term": item_name, "action_taken": "Searched ChromaDB"}

def tool_general_chat():
    """Trigger this for general conversation, greetings, or questions not related to the database."""
    return {"status": "success", "action_taken": "Standard LLM response generation"}

def agentic_intent_parser(user_message: str):
    """
    Replaces FIXED_COMMANDS and SentenceTransformers.
    Lets the LLM dynamically route the intent based on context.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # We pass the functions to the LLM so it knows what it is capable of doing
    model = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_message,
        config=types.GenerateContentConfig(
            tools=[tool_scan_drive, tool_check_inventory, tool_general_chat],
            temperature=0.0 # Force it to be strictly logical
        )
    )
    
    # Check if the AI decided to call a function
    if model.function_calls:
        function_call = model.function_calls[0]
        called_tool_name = function_call.name
        args = function_call.args
        
        print(f"🤖 AGENT DECISION: Executing {called_tool_name} with args {args}")
        return called_tool_name, args
        
    return "tool_general_chat", {}

if __name__ == "__main__":
    # Test it!
    agentic_intent_parser("สแกนไดรฟ์ให้หน่อยครับ") # Will route to tool_scan_drive
    agentic_intent_parser("ดารารัตน์เบิกสีดำเงาไปกี่กระป๋อง") # Will route to tool_check_inventory