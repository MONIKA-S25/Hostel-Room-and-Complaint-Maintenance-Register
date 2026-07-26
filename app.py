from flask import Flask, render_template_string, request
import pandas as pd
import datetime
import joblib

app = Flask(__name__)

# Load initial dataset and trained ML model
df = pd.read_csv('dataset.csv')

try:
    model = joblib.load('delay_model.pkl')
except Exception:
    model = None

# Single page HTML layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hostel Room Allocation & Maintenance Register</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f6f9; color: #333; }
        h1, h2 { color: #1e293b; }
        .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; border: 1px solid #cbd5e1; text-align: left; }
        th { background: #2563eb; color: white; }
        tr:nth-child(even) { background-color: #f8fafc; }
        input, select, button { padding: 8px; margin: 5px 0; width: 100%; box-sizing: border-box; }
        button { background: #16a34a; color: white; border: none; cursor: pointer; font-weight: bold; border-radius: 4px; }
        button:hover { background: #15803d; }
        .badge-high { background-color: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
        .badge-low { background-color: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>🏢 Hostel Room Allocation & Complaint Register</h1>

    <!-- Task 2 Form: Add Record / Complaint -->
    <div class="card">
        <h2>Allocate Room / Log Maintenance Complaint</h2>
        <form action="/add" method="POST">
            <input type="number" name="room_no" placeholder="Room Number (e.g. 105)" required>
            <input type="text" name="block" placeholder="Block (A, B, or C)" required>
            <input type="text" name="occupant_name" placeholder="Student Occupant Name">
            <select name="complaint_type">
                <option value="None">None (Allocate Room Only)</option>
                <option value="Plumbing">Plumbing</option>
                <option value="Electrical">Electrical</option>
                <option value="Carpentry">Carpentry</option>
                <option value="Cleaning">Cleaning</option>
            </select>
            <button type="submit">Save Record</button>
        </form>
    </div>

    <!-- Task 4 Form: NLP Chatbot Assistant -->
    <div class="card">
        <h2>🤖 Ask Assistant</h2>
        <form action="/ask" method="POST">
            <input type="text" name="question" placeholder="Try asking: 'which rooms are vacant?' or 'show pending complaints'" required>
            <button type="submit" style="background: #0284c7;">Submit Query</button>
        </form>
        {% if bot_response %}
            <div style="margin-top: 10px; padding: 10px; background: #e0f2fe; border-radius: 4px;">
                <strong>Assistant Answer:</strong> {{ bot_response }}
            </div>
        {% endif %}
    </div>

    <!-- Live Records Table -->
    <div class="card">
        <h2>📋 Live Register (Recent Records)</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Room</th>
                <th>Block</th>
                <th>Occupant Name</th>
                <th>Complaint</th>
                <th>Reported Date</th>
                <th>Days Open</th>
                <th>Status</th>
                <th>Predicted Delay Risk</th>
            </tr>
            {% for row in records %}
            <tr>
                <td>{{ row['record_id'] }}</td>
                <td>{{ row['room_no'] }}</td>
                <td>{{ row['block'] }}</td>
                <td>{{ row['occupant_name'] if row['occupant_name'] else '<i>Unassigned</i>' }}</td>
                <td>{{ row['complaint_type'] }}</td>
                <td>{{ row['reported_date'] }}</td>
                <td>{{ row['days_open'] }}</td>
                <td>{{ row['status'] }}</td>
                <td>
                    {% if row['delay_risk'] == 'High' %}
                        <span class="badge-high">High Risk</span>
                    {% elif row['delay_risk'] == 'Low' %}
                        <span class="badge-low">Low Risk</span>
                    {% else %}
                        N/A
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

def calculate_derived_fields(dataframe):
    """Calculates days_open dynamically on the server."""
    df_copy = dataframe.copy()
    today = datetime.date.today()
    days_open_list = []
    
    for _, row in df_copy.iterrows():
        if pd.notna(row['reported_date']) and str(row['reported_date']).strip() != "":
            rep_dt = datetime.datetime.strptime(str(row['reported_date']), '%Y-%m-%d').date()
            if row['status'] == 'Resolved' and pd.notna(row['resolved_date']) and str(row['resolved_date']).strip() != "":
                res_dt = datetime.datetime.strptime(str(row['resolved_date']), '%Y-%m-%d').date()
                days_open_list.append((res_dt - rep_dt).days)
            else:
                days_open_list.append((today - rep_dt).days)
        else:
            days_open_list.append(0)
            
    df_copy['days_open'] = days_open_list
    return df_copy

def get_delay_prediction(complaint_type, block):
    """Uses trained ML model to output delay risk prediction."""
    if model is None or complaint_type == 'None':
        return 'N/A'
    try:
        complaint_map = {'Plumbing': 0, 'Electrical': 1, 'Carpentry': 2, 'Cleaning': 3}
        block_map = {'A': 0, 'B': 1, 'C': 2}
        
        c_code = complaint_map.get(complaint_type, 0)
        b_code = block_map.get(block, 0)
        
        pred = model.predict([[c_code, b_code]])[0]
        return 'High' if pred == 1 else 'Low'
    except Exception:
        return 'N/A'

@app.route('/')
def home():
    df_calc = calculate_derived_fields(df)
    
    predictions = []
    for _, row in df_calc.head(15).iterrows():
        predictions.append(get_delay_prediction(row['complaint_type'], row['block']))
    
    df_calc_subset = df_calc.head(15).copy()
    df_calc_subset['delay_risk'] = predictions
    
    records = df_calc_subset.to_dict(orient='records')
    return render_template_string(HTML_TEMPLATE, records=records)

@app.route('/add', methods=['POST'])
def add_record():
    global df
    room_no = request.form.get('room_no')
    block = request.form.get('block', '').strip().upper()
    occupant_name = request.form.get('occupant_name', '').strip()
    complaint_type = request.form.get('complaint_type')
    
    # Server-side validation & record creation
    new_id = len(df) + 1
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    status = 'Pending' if complaint_type != 'None' else 'None'

    new_row = {
        'record_id': new_id,
        'room_no': int(room_no) if room_no else 0,
        'block': block,
        'occupant_name': occupant_name,
        'complaint_type': complaint_type,
        'reported_date': today_str if complaint_type != 'None' else "",
        'status': status,
        'resolved_date': "",
        'is_delayed': 0
    }

    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    return home()

@app.route('/ask', methods=['POST'])
def ask_assistant():
    q = request.form.get('question', '').strip().lower()
    
    # NLP Normalization & intent matching
    if 'vacant' in q or 'free' in q or 'empty' in q:
        vacant_rooms = df[df['occupant_name'].isna() | (df['occupant_name'] == "")]['room_no'].unique().tolist()
        ans = f"Vacant rooms available: {vacant_rooms[:5]}" if vacant_rooms else "No vacant rooms currently available."
    elif 'pending' in q or 'outstanding' in q:
        pending_count = len(df[df['status'] == 'Pending'])
        ans = f"There are currently {pending_count} outstanding pending maintenance complaints."
    elif 'count' in q or 'total' in q:
        ans = f"Total registered records in system: {len(df)}"
    elif 'plumbing' in q:
        count = len(df[df['complaint_type'] == 'Plumbing'])
        ans = f"Total plumbing issues logged: {count}"
    else:
        ans = "Sorry, I do not know the answer to that question. You can ask me about vacant rooms, pending complaints, or record counts."
        
    df_calc = calculate_derived_fields(df)
    predictions = [get_delay_prediction(row['complaint_type'], row['block']) for _, row in df_calc.head(15).iterrows()]
    df_calc_subset = df_calc.head(15).copy()
    df_calc_subset['delay_risk'] = predictions
    
    records = df_calc_subset.to_dict(orient='records')
    return render_template_string(HTML_TEMPLATE, records=records, bot_response=ans)

if __name__ == '__main__':
    app.run(debug=True)