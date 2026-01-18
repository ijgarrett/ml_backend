from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from sudoku_scanner_advanced import image_to_board
from detect_polynomial import extract_polynomial_boxes

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to ML Backend API',
        'status': 'running'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/sudoku/scan', methods=['POST'])
def scan_sudoku():
    """
    Scans a puzzle grid image and returns the extracted grid.
    Expects an image file in the request.
    Optional: grid_size parameter (default: 9) to specify grid dimensions (e.g., 4, 9, 16).
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

        grid_size = request.form.get('grid_size', 9, type=int)

        if grid_size <= 0 or grid_size > 25:
            return jsonify({'error': 'Grid size must be between 1 and 25'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            grid = image_to_board(filepath, grid_size=grid_size)

            os.remove(filepath)

            return jsonify({
                'success': True,
                'grid_size': grid_size,
                'grid': grid
            }), 200

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/polynomial/scan', methods=['POST'])
def scan_polynomial():
    """
    Scans a worksheet image and extracts polynomial boxes.
    Expects an image file in the request.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed types: png, jpg, jpeg'
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            result = extract_polynomial_boxes(filepath)

            os.remove(filepath)

            return jsonify({
                'success': True,
                'num_boxes': result['num_boxes'],
                'detections': result['detections'],
                'polynomial': result['polynomial']
            }), 200

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': f'Error processing image: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
