import streamlit as st
import chromadb
import pandas as pd
import os
import re

st.set_page_config(page_title="Factory Analytics Dashboard", layout="wide")

st.title("Factory Inventory Analytics")
st.markdown("Real-time insights extracted from vector database memory.")

def load_data():
    if not os.path.exists("./chroma_data"):
        st.error("No database found. Please run a Drive Scan from the Line Bot first.")
        return []
        
    client = chromadb.PersistentClient(path="./chroma_data")
    
    all_docs = []
    for collection in client.list_collections():
        data = collection.get()
        if data and data['documents']:
            all_docs.extend(data['documents'])
            
    return all_docs

docs = load_data()

if docs:
    st.success(f"Loaded {len(docs)} inventory records from memory.")
    
    parsed_data = []
    for text in docs:
        month_match = re.search(r"ข้อมูลเดือน:\s*(.*)", text)
        emp_match = re.search(r"ชื่อพนักงาน:\s*(.*)", text)
        items_match = re.search(r"รายการเบิกวัสดุสิ้นเปลือง:\s*(.*)", text)
        
        if emp_match and items_match:
            month = month_match.group(1).strip() if month_match else "Unknown"
            employee = emp_match.group(1).strip()
            items = items_match.group(1).strip()
            
            if items and items.lower() != "none" and items != "":
                parsed_data.append({"Month": month, "Employee": employee, "Items Drawn": items})

    df = pd.DataFrame(parsed_data)
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Activity Log")
            st.dataframe(df)
            
        with col2:
            st.subheader("Top Employees by Requisition Frequency")
            emp_counts = df['Employee'].value_counts()
            st.bar_chart(emp_counts)
    else:
        st.warning("Database records found, but none matched the parsing format. Ensure files are standard CSVs.")
        with st.expander("View Raw Database Chunks (For Debugging)"):
            st.write(docs[:5])
else:
    st.warning("Database is empty.")