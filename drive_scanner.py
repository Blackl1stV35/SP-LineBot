import csv
import logging

logger = logging.getLogger(__name__)

def parse_dense_inventory_csv(filepath):
    """
    Parses a 2D matrix CSV where:
    Row 0: Headers (Item names)
    Row 1: Month info in Col 0
    Row 2+: Employee names in Col 0, quantities in subsequent columns
    Returns pre-formatted semantic text chunks ready for vector database insertion.
    """
    chunks = []
    try:
        # utf-8-sig handles potential BOM characters common in Excel-exported CSVs
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            logger.warning(f"File {filepath} does not have enough rows.")
            return chunks

        headers = rows[0]
        # Month string usually sits at row 1, col 0
        month_info = rows[1][0].strip() if len(rows[1]) > 0 else "Unknown Month"

        for row in rows[2:]:
            if not row or not row[0].strip():
                continue
            
            employee_name = row[0].strip()
            items_drawn = []
            
            # Iterate through columns to find which items this employee drew
            for col_idx in range(1, len(row)):
                if col_idx < len(headers):
                    item_name = headers[col_idx].strip()
                    qty = row[col_idx].strip()
                    
                    # If quantity exists, pair it with the header name
                    if qty and item_name:
                        items_drawn.append(f"{item_name} จำนวน {qty}")

            # Only create a chunk if the employee actually drew items
            if items_drawn:
                chunk_text = f"ข้อมูลเดือน: {month_info}\nชื่อพนักงาน: {employee_name}\nรายการเบิกวัสดุสิ้นเปลือง: " + ", ".join(items_drawn)
                chunks.append(chunk_text)

    except Exception as e:
        logger.error(f"Error parsing CSV {filepath}: {str(e)}")

    return chunks