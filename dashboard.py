import streamlit as st
import chromadb
import pandas as pd
import os

st.set_page_config(page_title="Factory Analytics Dashboard", layout="wide")

st.title("🏭 Factory Inventory Analytics")
st.markdown("Real-time insights extracted from vector database memory.")

def load_data():
    if not os.path.exists("./chroma_data"):
        st.error("No database found. Please run a Drive Scan from the Line Bot first.")
        return []
        
    client = chromadb.PersistentClient(path="./chroma_data")
    
    # Fetch all collections (users)
    all_docs = []
    for collection in client.list_collections():
        data = collection.get()
        if data and data['documents']:
            all_docs.extend(data['documents'])
            
    return all_docs

docs = load_data()

if docs:
    st.success(f"Loaded {len(docs)} inventory records from memory.")
    
    # Process unstructured text into a DataFrame
    parsed_data = []
    for text in docs:
        if "ชื่อพนักงาน" in text:
            try:
                # Basic extraction logic for your specific CSV text chunks
                parts = text.split("\n")
                month = parts[0].split(":")[1].strip()
                employee = parts[1].split(":")[1].strip()
                items = parts[2].split(":")[1].strip()
                parsed_data.append({"Month": month, "Employee": employee, "Items Drawn": items})
            except:
                pass

    df = pd.DataFrame(parsed_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Activity Log")
        st.dataframe(df)
        
    with col2:
        st.subheader("Top Employees by Requisition Frequency")
        if not df.empty:
            emp_counts = df['Employee'].value_counts()
            st.bar_chart(emp_counts)
            
else:
    st.warning("Database is empty or formatting is unrecognized.")