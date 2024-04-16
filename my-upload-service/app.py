from flask import Flask, request, jsonify,send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from ultralytics import YOLO
import os


app = Flask(__name__)
CORS(app)

# 配置上传文件夹和允许的扩展名
UPLOAD_FOLDER = '/Users/smc/Downloads/flower/my-upload-service/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('runs/detect/train3/weights/best.pt')  # pretrained YOLOv8n model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 检查文件是否为图片格式（这里假设支持的图片格式为'.jpg', '.jpeg', '.png', '.gif', '.tif）
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif', 'tif')):
            # 删除图片文件
            os.remove(file_path)


@app.route('/upload', methods=['POST'])
def upload_file():
    clear_images('/Users/smc/Downloads/flower/my-upload-service/uploads')
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    # 如果用户没有选择文件，浏览器也会提交一个空的文件无文件名。
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        file.save('/Users/smc/Downloads/flower/my-upload-service/uploads/' + secure_filename(file.filename))
        
        results = model('/Users/smc/Downloads/flower/my-upload-service/uploads')  # return a list of Results objects
        # Process results list
        for i,result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            result.show()  # display to screen
            result.save(filename=f'/Users/smc/Downloads/flower/my-upload-service/downloads/0.jpg')  # save to disk

        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})
    
    return jsonify({'error': 'File not allowed'})

@app.route('/downloads/<filename>', methods=['GET'])
def download_file(filename):
    return send_file('/Users/smc/Downloads/flower/my-upload-service/downloads/0.jpg', as_attachment=True)



if __name__ == '__main__':
    # 确保上传文件夹存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
