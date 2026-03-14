import csv
import logging

logger = logging.getLogger(__name__)

def parse_dense_inventory_csv(filepath: str):
    """
    Transforms a 2D empty matrix into dense, highly-searchable semantic text chunks.
    """
    chunks = []
    try:
        # utf-8-sig handles Thai characters and Excel BOM perfectly
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return chunks

        # Row 0 contains our Items (Headers)
        headers = rows[0]
        
        # Row 1, Col 0 usually contains the Month (e.g., "เดือน1/2569")
        month_info = rows[1][0].strip() if len(rows[1]) > 0 else "Unknown Month"

        # Loop through employees (Row 2 downwards)
        for row in rows[2:]:
            if not row or not row[0].strip():
                continue # Skip empty rows
                
            employee_name = row[0].strip()
            items_drawn = []
            
            # Loop through the columns for this specific employee
            for col_idx in range(1, len(row)):
                if col_idx < len(headers):
                    qty = row[col_idx].strip()
                    item_name = headers[col_idx].strip()
                    
                    # ONLY record if the quantity is not empty
                    if qty and qty != "":
                        items_drawn.append(f"{item_name} จำนวน {qty}")

            # If the employee actually drew something, create a rich memory chunk
            if items_drawn:
                chunk_text = (
                    f"ข้อมูลเดือน: {month_info}\n"
                    f"ชื่อพนักงาน: {employee_name}\n"
                    f"รายการเบิกวัสดุสิ้นเปลือง: " + ", ".join(items_drawn)
                )
                chunks.append(chunk_text)

    except Exception as e:
        logger.error(f"Error parsing semantic CSV {filepath}: {str(e)}")

    return chunks