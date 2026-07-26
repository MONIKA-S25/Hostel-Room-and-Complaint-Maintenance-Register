import pandas as pd
import random
from datetime import datetime, timedelta

blocks = ['A', 'B', 'C']
complaint_types = ['Plumbing', 'Electrical', 'Carpentry', 'Cleaning', 'None']
statuses = ['Pending', 'In Progress', 'Resolved']

data = []
start_date = datetime(2026, 1, 1)

for i in range(1, 101):
    rec_id = i
    room_no = 100 + (i % 20) + 1
    block = random.choice(blocks)
    name = f"Student_{i}"
    
    # Intentional missing value case for testing edge cases
    if i == 15:
        name = ""
        
    c_type = random.choice(complaint_types)
    
    if c_type == 'None':
        status = 'None'
        rep_date = ""
        res_date = ""
        is_delayed = 0
    else:
        status = random.choice(statuses)
        days_ago = random.randint(1, 40)
        rep_dt = start_date + timedelta(days=days_ago)
        rep_date = rep_dt.strftime('%Y-%m-%d')
        
        if status == 'Resolved':
            res_dt = rep_dt + timedelta(days=random.randint(1, 5))
            res_date = res_dt.strftime('%Y-%m-%d')
            is_delayed = 1 if (res_dt - rep_dt).days > 3 else 0
        else:
            res_date = ""
            is_delayed = 1 if (datetime.now() - rep_dt).days > 3 else 0

    data.append([rec_id, room_no, block, name, c_type, rep_date, status, res_date, is_delayed])

df = pd.DataFrame(data, columns=[
    'record_id', 'room_no', 'block', 'occupant_name', 
    'complaint_type', 'reported_date', 'status', 'resolved_date', 'is_delayed'
])

df.to_csv('dataset.csv', index=False)
print("Success! Created dataset.csv with 100 records.")