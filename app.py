from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from PIL import Image
import torch
import pandas as pd
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipImageProcessor
from datetime import datetime
from collections import Counter
from database import VQAHistory, get_db
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ground truth data
try:
    df_ground_truth = pd.read_excel('test_pred_30.xlsx')
    print("Ground truth data loaded successfully!")
except Exception as e:
    print(f"Error loading ground truth data: {e}")
    df_ground_truth = None

def check_ground_truth(question, predicted_answer):
    if df_ground_truth is None:
        return None, None
    
    # Find the matching question in the dataset
    matching_row = df_ground_truth[df_ground_truth['Question'].str.lower() == question.lower()]
    
    if not matching_row.empty:
        actual_answer = matching_row['Actual'].iloc[0]
        is_correct = predicted_answer.lower() == actual_answer.lower()
        return actual_answer, is_correct
    return None, None

# Khởi tạo model và processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "Salesforce/blip-vqa-base"

# Load processors
text_processor = BlipProcessor.from_pretrained(checkpoint, cache_dir=f'./cache/processor/{checkpoint}')
image_processor = BlipImageProcessor.from_pretrained(checkpoint, cache_dir=f'./cache/Iprocessor/{checkpoint}')

# Load model giống như trong file path-vqa.py
model = BlipForQuestionAnswering.from_pretrained(checkpoint, cache_dir=f'./cache/model/{checkpoint}')
model.to(device)

# Load state dict từ file đã train
# model_path = os.path.join('model_save', 'model_30.pth')  # Điều chỉnh đường dẫn nếu cần
# print(model_path)
# assert os.path.exists(model_path), f"Model file not found at {model_path}"
model_path = 'model_30.pth'

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.eval()

# Tạo class để lưu lịch sử
class History:
    def add_item(self, image_path, question, predicted_answer, actual_answer, timestamp):
        db = next(get_db())
        history_item = VQAHistory(
            image_path=image_path,
            question=question,
            predicted_answer=predicted_answer,
            actual_answer=actual_answer,
            is_correct=(predicted_answer.lower() == actual_answer.lower()) if actual_answer else None,
            timestamp=timestamp
        )
        db.add(history_item)
        db.commit()

    def get_items(self, skip=0, limit=10):
        db = next(get_db())
        return db.query(VQAHistory).order_by(VQAHistory.timestamp.desc()).offset(skip).limit(limit).all()

    def clear_items(self):
        db = next(get_db())
        db.query(VQAHistory).delete()
        db.commit()

    def get_statistics(self):
        db = next(get_db())
        items = db.query(VQAHistory).all()
        total = len(items)
        if total == 0:
            return None
            
        correct = sum(1 for item in items if item.is_correct)
        accuracy = (correct / total) * 100

        predicted_counts = Counter(item.predicted_answer.lower() for item in items)
    
        try:
            # Tạo biểu đồ với backend Agg
            plt.switch_backend('Agg')
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(predicted_counts.keys(), predicted_counts.values())
            ax.set_title('Distribution of Predicted Answers')
            
            # Chuyển biểu đồ thành base64 string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close(fig)
            
            return {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'chart': img_base64,
                'predicted_counts': dict(predicted_counts)
            }
        except Exception as e:
            print(f"Error generating statistics: {e}")
            return {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'predicted_counts': dict(predicted_counts)
            }
        
# Khởi tạo history
history = History()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    question = request.form['question']
    
    # Lưu và xử lý ảnh
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    image = Image.open(img_path).convert('RGB')
    
    # Xử lý ảnh và câu hỏi
    image_encoding = image_processor(image, return_tensors="pt").to(device)
    text_encoding = text_processor(None, question, return_tensors="pt").to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=image_encoding['pixel_values'],
            input_ids=text_encoding['input_ids']
        )
    
    predicted_answer = text_processor.decode(outputs[0], skip_special_tokens=True)
    
    # Check ground truth
    actual_answer, is_correct = check_ground_truth(question, predicted_answer)
    
    history.add_item(
        image_path=os.path.join('uploads', file.filename),
        question=question,
        predicted_answer=predicted_answer,
        actual_answer=actual_answer,
        timestamp=datetime.now()
    )
    
    return render_template('result.html', 
                         image_path=os.path.join('uploads', file.filename),
                         question=question,
                         answer=predicted_answer,
                         actual_answer=actual_answer,
                         is_correct=is_correct)

@app.route('/history')  # Thêm route decorator này
def view_history():
    page = request.args.get('page', 1, type=int)
    per_page = 6
    
    db = next(get_db())
    total_items = db.query(VQAHistory).count()
    total_pages = (total_items + per_page - 1) // per_page
    
    items = db.query(VQAHistory)\
        .order_by(VQAHistory.timestamp.desc())\
        .offset((page - 1) * per_page)\
        .limit(per_page)\
        .all()
    
    return render_template('history.html', 
                         items=items,
                         page=page,
                         total_pages=total_pages)
@app.route('/clear_history', methods=['POST'])
def clear_history():
    history.clear_items()  # Thay vì history.items.clear()
    return redirect(url_for('view_history'))  # Thay vì url_for('history')
    return redirect(url_for('history'))

@app.route('/statistics')
def view_statistics():
    stats = history.get_statistics()
    return render_template('statistics.html', stats=stats)

@app.route('/export_history')
def export_history():
    db = next(get_db())
    items = db.query(VQAHistory).all()
    if not items:
        return "No history to export", 400
        
    # Convert SQLAlchemy objects to dictionaries
    items_dict = [{
        'image_path': item.image_path,
        'question': item.question,
        'predicted_answer': item.predicted_answer,
        'actual_answer': item.actual_answer,
        'timestamp': item.timestamp
    } for item in items]
    
    df = pd.DataFrame(items_dict)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='vqa_history.xlsx'
    )

@app.route('/search_history')
def search_history():
    query = request.args.get('q', '').lower()
    if not query:
        return redirect(url_for('view_history'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 6
    
    db = next(get_db())
    # Get filtered items
    filtered_items = db.query(VQAHistory).filter(
        (VQAHistory.question.ilike(f'%{query}%')) |
        (VQAHistory.predicted_answer.ilike(f'%{query}%')) |
        (VQAHistory.actual_answer.ilike(f'%{query}%'))
    ).all()
    
    # Calculate total pages for pagination
    total_items = len(filtered_items)
    total_pages = (total_items + per_page - 1) // per_page
    
    # Get items for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    items = filtered_items[start_idx:end_idx]
    
    return render_template('history.html', 
                         items=items,
                         page=page,
                         total_pages=total_pages,
                         search_query=query)

@app.route('/filter_history')
def filter_history():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    filtered_items = history.items
    
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        filtered_items = [item for item in filtered_items 
                         if item['timestamp'].date() >= start_date.date()]
    
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        filtered_items = [item for item in filtered_items 
                         if item['timestamp'].date() <= end_date.date()]
    
    return render_template('history.html', 
                         items=filtered_items,
                         start_date=start_date,
                         end_date=end_date)

def get_time_series_stats():
    if not history.items:
        return None
        
    # Nhóm theo ngày
    daily_stats = {}
    for item in history.items:
        date = item['timestamp'].date()
        if date not in daily_stats:
            daily_stats[date] = {'total': 0, 'correct': 0}
        
        daily_stats[date]['total'] += 1
        if item.get('actual') and item['predicted'].lower() == item['actual'].lower():
            daily_stats[date]['correct'] += 1
    
    # Tạo biểu đồ
    dates = list(daily_stats.keys())
    accuracies = [daily_stats[d]['correct']/daily_stats[d]['total']*100 for d in dates]
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, accuracies, marker='o')
    plt.title('Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    time_series_chart = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return time_series_chart

if __name__ == '__main__':
    app.run(debug=True)
