import csv
import logging
import openpyxl

logger = logging.getLogger(__name__)

def parse_dense_inventory_csv(filepath: str):
    """
    Transforms a 2D empty matrix (CSV or Excel) into dense, highly-searchable semantic text chunks.
    Formats exactly to match dashboard.py Regex.
    """
    chunks = []
    try:
        rows = []
        
        # 1. Read Data (Handle both CSV and XLSX)
        if filepath.lower().endswith('.csv'):
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            wb = openpyxl.load_workbook(filepath, data_only=True)
            sheet = wb.active
            for row in sheet.iter_rows(values_only=True):
                # Convert None to empty string
                rows.append([str(cell).strip() if cell is not None else "" for cell in row])

        if len(rows) < 2:
            return chunks

        # 2. Extract Headers and Month Info
        headers = rows[0]
        # Usually row 1, col 0 contains the month (e.g., "แผ่นที่1 เดือน1/2569")
        month_info = rows[1][0].strip() if len(rows) > 1 and len(rows[1]) > 0 else "ไม่ระบุเดือน"

        # 3. Unroll Employees (Row 2 downwards)
        for row in rows[2:]:
            if not row or not row[0].strip():
                continue 
                
            employee_name = row[0].strip()
            
            # Skip junk rows or headers that bled over
            if employee_name == '|' or 'Unnamed' in employee_name or 'แผ่นที่' in employee_name:
                continue
                
            items_drawn = []
            
            # Loop through the columns for this specific employee
            for col_idx in range(1, len(row)):
                if col_idx < len(headers):
                    qty = row[col_idx].strip()
                    item_name = headers[col_idx].strip()
                    
                    # ONLY record if the quantity is not empty, not a pipe, and not NaN
                    if qty and qty != "" and qty != "|" and qty.lower() != "nan":
                        items_drawn.append(f"{item_name} จำนวน {qty}")

            # 4. Format strictly for the Dashboard Regex and ChromaDB
            if items_drawn:
                chunk_text = (
                    f"ข้อมูลเดือน: {month_info}\n"
                    f"ชื่อพนักงาน: {employee_name}\n"
                    f"รายการเบิกวัสดุสิ้นเปลือง: " + ", ".join(items_drawn)
                )
                chunks.append(chunk_text)

    except Exception as e:
        logger.error(f"Error parsing semantic file {filepath}: {str(e)}")

    return chunks