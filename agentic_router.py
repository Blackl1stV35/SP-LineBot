import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def tool_scan_drive():
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ สแกนไดรฟ์, อัพเดทไฟล์, หรือดึงข้อมูลจาก Google Drive"""
    pass

def tool_check_inventory(search_query: str):
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ ค้นหาข้อมูลการเบิก, เช็คสต็อก, หรือถามว่าใครเบิกอะไรไปบ้าง
    Args:
        search_query: สิ่งที่ผู้ใช้ต้องการค้นหา เช่น 'สีดำเงา', 'โจ้เบิกอะไร', 'เดือน 2'
    """
    pass

def tool_add_memory(note: str):
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ บันทึกข้อมูลใหม่, เพิ่มสต็อก, ลดสต็อก, หรือจดบันทึก
    Args:
        note: สรุปข้อความที่ต้องการบันทึก เช่น 'โจ้เบิกสี 2 ป๋อง', 'รับของเข้า 5 ลัง'
    """
    pass

def tool_general_chat():
    """ใช้ฟังก์ชันนี้สำหรับการพูดคุยทั่วไป ทักทาย ถามวิธีซ่อมแซม หรือคำถามที่ไม่เกี่ยวกับสต็อก"""
    pass

def analyze_intent(user_message: str):
    """
    Evaluates user message and decides the action.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Analyze this request and call the right tool: {user_message}",
            config=types.GenerateContentConfig(
                tools=[tool_scan_drive, tool_check_inventory, tool_add_memory, tool_general_chat],
                temperature=0.0 # Force strict logical routing
            )
        )
        
        if response.function_calls:
            call = response.function_calls[0]
            args_dict = {k: v for k, v in call.args.items()} if call.args else {}
            return call.name, args_dict
            
        return "tool_general_chat", {}
    except Exception as e:
        print(f"Agentic Router Error: {e}")
        return "tool_general_chat", {}

if __name__ == "__main__":
    print("Test 1 (Scan):", analyze_intent("สแกนไดรฟ์ให้หน่อยครับ"))
    print("Test 2 (Search):", analyze_intent("ดารารัตน์เบิกสีดำเงาไปกี่กระป๋อง"))
    print("Test 3 (Write-Back):", analyze_intent("จดไว้หน่อย เดือน 3 โจ้เบิกทินเนอร์ 2 แกลลอน"))
    print("Test 4 (Chat):", analyze_intent("สวัสดีครับ วันนี้อากาศดีไหม"))