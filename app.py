from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
import os
import numpy as np
from skimage import io
import supervision as sv
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import base64
from io import BytesIO
import cv2
from PIL import Image
import tempfile
from roboflow import Roboflow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from skimage.metrics import structural_similarity as ssim

# Import the model class from the provided code
class Model:
    def __init__(self, api_key: str, workspace: str, project: str, version: int):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
        self.model = self.project.version(version).model

    def predict_cheque_quality(
        self,
        image_path: str,
        confidence: int = 40,
        overlap: int = 30
    ) -> str:
        try:
            print(f"📂 Checking image: {image_path}")
            result = self.predict(image_path, confidence=confidence, overlap=overlap)
            detections = sv.Detections.from_inference(result)
            classes = detections.data["class_name"]
            print(f"🔍 Detected: {classes}")

            required = {'Account no','Amount','date','Bank name','cheque no'}
            if required.issubset(set(classes)):
                print("✅ Good cheque")
                return "good"
            else:
                print("❌ Bad cheque")
                return "bad"

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return "bad"
        
    def predict(self, image_path: str, confidence: int = 40, overlap: int = 30) -> dict:
        """Run inference and return the raw JSON."""
        return self.model.predict(image_path, confidence=confidence, overlap=overlap).json()

    def detect(self, image_path: str, confidence: int = 40, overlap: int = 30) -> sv.Detections:
        """Run inference and convert to a Supervision Detections object."""
        result = self.predict(image_path, confidence, overlap)
        image = io.imread(image_path)
        return image, sv.Detections.from_inference(result)

    def crop_fields(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        field_names: list[str],
    ) -> dict[str, np.ndarray | None]:
        """
        Safely crop each field in field_names from the detection results.
        Returns a dict mapping field_name -> cropped_image or None.
        """
        class_list = detections.data["class_name"]
        out = {}
        for name in field_names:
            det = detections[class_list == name]
            if len(det) > 0:
                out[name] = sv.crop_image(image, det.xyxy[0].tolist())
            else:
                print(f"⚠️ Field not detected: {name}")
                out[name] = None
        return out

    @staticmethod
    def run_trocr(
        image: np.ndarray | Image.Image,
        model_name: str = "microsoft/trocr-large-handwritten",
    ) -> str:
        """
        Run Microsoft TrOCR on one cropped image and return the decoded text.
        """
        # to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return texts[0]

class SignatureMatching:
    """
    Provides methods to compare a test signature image against a reference
    (or a directory of stored signatures) using Structural Similarity Index.
    """

    def __init__(self, signature_dir: str):
        self.signature_dir = signature_dir

    @staticmethod
    def compare_signatures(ref_path: str, test_image: np.ndarray) -> float:
        """
        Read the reference image from disk and the test image (RGB or grayscale NumPy array),
        resize both to a common shape, then compute SSIM score.
        """
        # load reference as grayscale
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        # ensure test is grayscale
        if test_image.ndim == 3:
            test_img = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        else:
            test_img = test_image
        # resize to fixed size
        ref_img = cv2.resize(ref_img, (200, 100))
        test_img = cv2.resize(test_img, (200, 100))
        # compute SSIM
        score, _ = ssim(ref_img, test_img, full=True)
        return score

    def find_best_signature_match(
        self,
        test_image: np.ndarray,
        uploaded_signature_path: str | None = None
    ) -> tuple[float, str | None]:
        """
        If `uploaded_signature_path` is provided, compare only against that file.
        Otherwise iterate over all image files in `self.signature_dir`,
        return the highest SSIM score and matching filename.
        """
        # compare against uploaded file first
        if uploaded_signature_path:
            score = self.compare_signatures(uploaded_signature_path, test_image)
            return score, uploaded_signature_path

        # otherwise scan the directory
        best_score = 0.0
        best_match = None
        for fname in os.listdir(self.signature_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(self.signature_dir, fname)
            score = self.compare_signatures(path, test_image)
            if score > best_score:
                best_score, best_match = score, fname

        return best_score, best_match

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SIGNATURE_FOLDER'] = 'signatures'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SIGNATURE_FOLDER'], exist_ok=True)

# Global to store evaluation results
evaluation_results = {
    'true_labels': [],
    'predicted_labels': [],
    'images': []
}

# Initialize model with your API key
# You will need to set this before running
model = None  # Will be initialized in a route

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def plot_confusion_matrix():
    """Generate a confusion matrix as a base64 encoded image"""
    if not evaluation_results['true_labels'] or not evaluation_results['predicted_labels']:
        return None
    
    cm = confusion_matrix(evaluation_results['true_labels'], 
                          evaluation_results['predicted_labels'], 
                          labels=["good", "bad"])
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Bad"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    global model
    
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        workspace = request.form.get('workspace')
        project = request.form.get('project')
        version = request.form.get('version')
        
        try:
            # Initialize the model with provided credentials
            model = Model(api_key=api_key, 
                          workspace=workspace, 
                          project=project, 
                          version=int(version))
            
            flash('Model initialized successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error initializing model: {str(e)}', 'error')
    
    return render_template('setup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not model:
        flash('Please set up the model first!', 'error')
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            prediction = model.predict_cheque_quality(filepath)
            
            # Get detected fields and visualization
            try:
                image, detections = model.detect(filepath)
                # Create annotated image
                box_annotator = sv.BoxAnnotator()
                annotated_image = box_annotator.annotate(
                    scene=image, 
                    detections=detections
                )
                
                # Convert numpy array to PIL Image
                annotated_pil = Image.fromarray(annotated_image)
                
                # Save to BytesIO
                buf = BytesIO()
                annotated_pil.save(buf, format='PNG')
                buf.seek(0)
                
                # Convert to base64
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                return render_template('prediction.html', 
                                       filename=filename,
                                       prediction=prediction,
                                       image_data=img_str,
                                       classes=list(detections.data["class_name"]))
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return render_template('prediction.html', 
                                       filename=filename,
                                       prediction=prediction,
                                       error=str(e))
    
    return render_template('predict.html')

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    global evaluation_results
    
    if not model:
        flash('Please set up the model first!', 'error')
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Clear previous evaluation results
        evaluation_results = {
            'true_labels': [],
            'predicted_labels': [],
            'images': []
        }
        
        # Get the files
        good_files = request.files.getlist('good_files')
        bad_files = request.files.getlist('bad_files')
        
        temp_files = []
        
        # Process 'good' cheques
        for file in good_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                temp_files.append(filepath)
                
                # Add to evaluation data
                evaluation_results['true_labels'].append("good")
                predicted = model.predict_cheque_quality(filepath)
                evaluation_results['predicted_labels'].append(predicted)
                
                # Save image info
                img = Image.open(filepath)
                img.thumbnail((150, 150))
                buf = BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                evaluation_results['images'].append({
                    'path': filepath,
                    'true_label': 'good',
                    'predicted_label': predicted,
                    'thumbnail': img_str
                })
        
        # Process 'bad' cheques
        for file in bad_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                temp_files.append(filepath)
                
                # Add to evaluation data
                evaluation_results['true_labels'].append("bad")
                predicted = model.predict_cheque_quality(filepath)
                evaluation_results['predicted_labels'].append(predicted)
                
                # Save image info
                img = Image.open(filepath)
                img.thumbnail((150, 150))
                buf = BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                evaluation_results['images'].append({
                    'path': filepath,
                    'true_label': 'bad',
                    'predicted_label': predicted,
                    'thumbnail': img_str
                })
        
        # Calculate accuracy
        if evaluation_results['true_labels']:
            accuracy = accuracy_score(evaluation_results['true_labels'], 
                                      evaluation_results['predicted_labels'])
            conf_matrix_img = plot_confusion_matrix()
            
            return render_template('evaluation_results.html',
                                  accuracy=accuracy * 100,
                                  results=evaluation_results,
                                  conf_matrix=conf_matrix_img)
        else:
            flash('No valid files were uploaded', 'error')
    
    return render_template('evaluate.html')

@app.route('/signature', methods=['GET', 'POST'])
def signature():
    if not model:
        flash('Please set up the model first!', 'error')
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Check if the post request has both files
        if 'reference_file' not in request.files or 'test_file' not in request.files:
            flash('Both reference and test files are required', 'error')
            return redirect(request.url)
        
        ref_file = request.files['reference_file']
        test_file = request.files['test_file']
        
        # Validate files
        if ref_file.filename == '' or test_file.filename == '':
            flash('Both files must be selected', 'error')
            return redirect(request.url)
        
        if (ref_file and allowed_file(ref_file.filename) and 
            test_file and allowed_file(test_file.filename)):
            
            # Save reference signature
            ref_filename = secure_filename(ref_file.filename)
            ref_filepath = os.path.join(app.config['SIGNATURE_FOLDER'], ref_filename)
            ref_file.save(ref_filepath)
            
            # Save test file
            test_filename = secure_filename(test_file.filename)
            test_filepath = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
            test_file.save(test_filepath)
            
            # Initialize signature matcher
            sig_matcher = SignatureMatching(app.config['SIGNATURE_FOLDER'])
            
            # Read test image
            test_img = cv2.imread(test_filepath)
            
            # Compare signatures
            score, _ = sig_matcher.find_best_signature_match(
                test_img, uploaded_signature_path=ref_filepath)
            
            # Load images for display
            ref_img = Image.open(ref_filepath)
            test_img = Image.open(test_filepath)
            
            # Convert to base64 for display
            ref_buf = BytesIO()
            ref_img.save(ref_buf, format='PNG')
            ref_buf.seek(0)
            ref_img_str = base64.b64encode(ref_buf.getvalue()).decode('utf-8')
            
            test_buf = BytesIO()
            test_img.save(test_buf, format='PNG')
            test_buf.seek(0)
            test_img_str = base64.b64encode(test_buf.getvalue()).decode('utf-8')
            
            # Result
            match_result = "Match" if score > 0.9 else "No Match"
            
            return render_template('signature_result.html',
                                  score=score * 100,
                                  result=match_result,
                                  ref_image=ref_img_str,
                                  test_image=test_img_str)
    
    return render_template('signature.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Add these imports at the top if not already present
import os
import re

# Add a new folder configuration for storing account-based signatures
app.config['ORIGINAL_SIGNATURE_FOLDER'] = 'original-signatures'
os.makedirs(app.config['ORIGINAL_SIGNATURE_FOLDER'], exist_ok=True)

# Add these routes after your existing routes

@app.route('/process-cheque', methods=['GET', 'POST'])
def process_cheque():
    if not model:
        flash('Please set up the model first!', 'error')
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'cheque_file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['cheque_file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Step 1: Check if the cheque is good or bad
                cheque_quality = model.predict_cheque_quality(filepath)
                
                # Step 2: Get detected fields and visualization
                image, detections = model.detect(filepath)
                
                # Create annotated image
                box_annotator = sv.BoxAnnotator()
                annotated_image = box_annotator.annotate(
                    scene=image, 
                    detections=detections
                )
                
                # Convert numpy array to PIL Image
                annotated_pil = Image.fromarray(annotated_image)
                
                # Save to BytesIO for display
                buf = BytesIO()
                annotated_pil.save(buf, format='PNG')
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                # Step 3: Extract field information
                field_names = ['Account no', 'Amount', 'date', 'Bank name', 'cheque no', 'Signature']
                cropped_fields = model.crop_fields(image, detections, field_names)
                
                # Step 4: Process each field with OCR if needed
                extracted_fields = {}
                for field_name, field_img in cropped_fields.items():
                    if field_img is not None:
                        # Use OCR for text fields
                        if field_name != 'Signature':
                            text = model.run_trocr(field_img)
                            extracted_fields[field_name] = text
                
                # Step 5: Check for account number and clean it
                account_number = None
                if 'Account no' in extracted_fields:
                    # Clean up account number (remove spaces and non-numeric chars)
                    account_number = re.sub(r'[^0-9]', '', extracted_fields['Account no'])
                    extracted_fields['Account no'] = account_number
                
                # Step 6: Process signature if found
                signature_match = False
                signature_score = 0
                extracted_signature_str = None
                reference_signature_str = None
                
                if 'Signature' in cropped_fields and cropped_fields['Signature'] is not None:
                    # Initialize signature matcher
                    sig_matcher = SignatureMatching(app.config['ORIGINAL_SIGNATURE_FOLDER'])
                    
                    # Convert signature to image for display
                    signature_img = Image.fromarray(cropped_fields['Signature'])
                    sig_buf = BytesIO()
                    signature_img.save(sig_buf, format='PNG')
                    sig_buf.seek(0)
                    extracted_signature_str = base64.b64encode(sig_buf.getvalue()).decode('utf-8')
                    
                    # Look for reference signature based on account number
                    if account_number:
                        ref_sig_path = None
                        # Check if we have a reference signature for this account
                        for fname in os.listdir(app.config['ORIGINAL_SIGNATURE_FOLDER']):
                            if fname.startswith(account_number) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                                ref_sig_path = os.path.join(app.config['ORIGINAL_SIGNATURE_FOLDER'], fname)
                                break
                        
                        if ref_sig_path:
                            # Compare signatures
                            signature_score, _ = sig_matcher.find_best_signature_match(
                                cropped_fields['Signature'], uploaded_signature_path=ref_sig_path)
                            signature_score *= 100  # Convert to percentage
                            signature_match = signature_score > 80  # Set threshold
                            
                            # Load reference signature for display
                            ref_img = Image.open(ref_sig_path)
                            ref_buf = BytesIO()
                            ref_img.save(ref_buf, format='PNG')
                            ref_buf.seek(0)
                            reference_signature_str = base64.b64encode(ref_buf.getvalue()).decode('utf-8')
                
                return render_template('cheque_results.html',
                                    cheque_quality=cheque_quality,
                                    image_data=img_str,
                                    extracted_fields=extracted_fields,
                                    extracted_signature=extracted_signature_str,
                                    reference_signature=reference_signature_str,
                                    signature_match=signature_match,
                                    signature_score=signature_score)
            
            except Exception as e:
                flash(f'Error processing cheque: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template('process_cheque.html')

@app.route('/upload-signature', methods=['GET', 'POST'])
def upload_signature():
    if request.method == 'POST':
        # Check if the post request has account number and file
        if 'account_number' not in request.form or 'signature_file' not in request.files:
            flash('Both account number and signature file are required', 'error')
            return redirect(request.url)
        
        account_number = request.form['account_number']
        file = request.files['signature_file']
        
        # Basic validation
        if not account_number or account_number.strip() == '':
            flash('Account number is required', 'error')
            return redirect(request.url)
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        # Clean account number (remove non-numeric chars)
        account_number = re.sub(r'[^0-9]', '', account_number)
        
        if account_number == '':
            flash('Account number must contain numeric digits', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Create filename with account number prefix
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{account_number}_signature.{extension}"
            filepath = os.path.join(app.config['ORIGINAL_SIGNATURE_FOLDER'], filename)
            
            # Remove any existing signature for this account
            for existing_file in os.listdir(app.config['ORIGINAL_SIGNATURE_FOLDER']):
                if existing_file.startswith(f"{account_number}_"):
                    os.remove(os.path.join(app.config['ORIGINAL_SIGNATURE_FOLDER'], existing_file))
            
            # Save the new signature
            file.save(filepath)
            
            flash(f'Signature for account {account_number} saved successfully!', 'success')
            return redirect(url_for('index'))
    
    return render_template('upload_signature.html')

if __name__ == '__main__':
    app.run(debug=True)
