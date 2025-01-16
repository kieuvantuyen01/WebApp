from flask import Flask, render_template, request
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipImageProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    
    answer = text_processor.decode(outputs[0], skip_special_tokens=True)
    
    return render_template('result.html', 
                         image_path=os.path.join('uploads', file.filename),
                         question=question,
                         answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
