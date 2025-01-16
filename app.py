from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import torch
import pandas as pd
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipImageProcessor
from datetime import datetime

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
    def __init__(self):
        self.items = []
    
    def add_item(self, image_path, question, predicted_answer, actual_answer, timestamp):
        self.items.append({
            'image_path': image_path,
            'question': question,
            'predicted': predicted_answer,
            'actual': actual_answer,
            'timestamp': timestamp
        })
    
    def get_items(self, limit=10):
        return list(reversed(self.items[-limit:]))

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

@app.route('/history')
def view_history():
    return render_template('history.html', items=history.get_items())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    history.items.clear()
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
