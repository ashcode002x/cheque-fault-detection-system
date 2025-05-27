from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import re
import cv2
import base64
from io import BytesIO
from PIL import Image
import supervision as sv
from sklearn.metrics import accuracy_score

from models import ChequeModel, SignatureMatching
from database import get_signature_path_by_account, insert_signature
from utils import allowed_file, plot_confusion_matrix

# Global variables
model = None
evaluation_results = {
    'true_labels': [],
    'predicted_labels': [],
    'images': []
}

def init_routes(app):
    """Initialize all routes for the Flask app."""
    
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
                model = ChequeModel(
                    api_key=api_key,
                    workspace=workspace,
                    project=project,
                    version=int(version)
                )
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
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                prediction = model.predict_cheque_quality(filepath)
                
                try:
                    image, detections = model.detect(filepath)
                    box_annotator = sv.BoxAnnotator()
                    annotated_image = box_annotator.annotate(scene=image, detections=detections)
                    
                    annotated_pil = Image.fromarray(annotated_image)
                    buf = BytesIO()
                    annotated_pil.save(buf, format='PNG')
                    buf.seek(0)
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
            evaluation_results = {'true_labels': [], 'predicted_labels': [], 'images': []}
            
            good_files = request.files.getlist('good_files')
            bad_files = request.files.getlist('bad_files')
            
            # Process good cheques
            for file in good_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    evaluation_results['true_labels'].append("good")
                    predicted = model.predict_cheque_quality(filepath)
                    evaluation_results['predicted_labels'].append(predicted)
                    
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
            
            # Process bad cheques
            for file in bad_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    evaluation_results['true_labels'].append("bad")
                    predicted = model.predict_cheque_quality(filepath)
                    evaluation_results['predicted_labels'].append(predicted)
                    
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
            
            if evaluation_results['true_labels']:
                accuracy = accuracy_score(evaluation_results['true_labels'],
                                        evaluation_results['predicted_labels'])
                conf_matrix_img = plot_confusion_matrix(evaluation_results)
                
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
            if 'reference_file' not in request.files or 'test_file' not in request.files:
                flash('Both reference and test files are required', 'error')
                return redirect(request.url)
            
            ref_file = request.files['reference_file']
            test_file = request.files['test_file']
            
            if ref_file.filename == '' or test_file.filename == '':
                flash('Both files must be selected', 'error')
                return redirect(request.url)
            
            if (ref_file and allowed_file(ref_file.filename) and
                test_file and allowed_file(test_file.filename)):
                
                ref_filename = secure_filename(ref_file.filename)
                ref_filepath = os.path.join(app.config['SIGNATURE_FOLDER'], ref_filename)
                ref_file.save(ref_filepath)
                
                test_filename = secure_filename(test_file.filename)
                test_filepath = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
                test_file.save(test_filepath)
                
                sig_matcher = SignatureMatching(app.config['SIGNATURE_FOLDER'])
                test_img = cv2.imread(test_filepath)
                
                score, _ = sig_matcher.find_best_signature_match(
                    test_img, uploaded_signature_path=ref_filepath)
                
                ref_img = Image.open(ref_filepath)
                test_img = Image.open(test_filepath)
                
                ref_buf = BytesIO()
                ref_img.save(ref_buf, format='PNG')
                ref_buf.seek(0)
                ref_img_str = base64.b64encode(ref_buf.getvalue()).decode('utf-8')
                
                test_buf = BytesIO()
                test_img.save(test_buf, format='PNG')
                test_buf.seek(0)
                test_img_str = base64.b64encode(test_buf.getvalue()).decode('utf-8')
                
                match_result = "Match" if score > 0.9 else "No Match"
                
                return render_template('signature_result.html',
                                     score=score * 100,
                                     result=match_result,
                                     ref_image=ref_img_str,
                                     test_image=test_img_str)
        
        return render_template('signature.html')

    @app.route('/process-cheque', methods=['GET', 'POST'])
    def process_cheque():
        if not model:
            flash('Please set up the model first!', 'error')
            return redirect(url_for('setup'))
        
        if request.method == 'POST':
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
                    cheque_quality = model.predict_cheque_quality(filepath)
                    image, detections = model.detect(filepath)
                    
                    box_annotator = sv.BoxAnnotator()
                    annotated_image = box_annotator.annotate(scene=image, detections=detections)
                    
                    annotated_pil = Image.fromarray(annotated_image)
                    buf = BytesIO()
                    annotated_pil.save(buf, format='PNG')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    field_names = ["micr", "Date", "Account number", "ifsc", "Amount",
                                 "Signature", "from name", "payname", "Bank Name"]
                    cropped_fields = model.crop_fields(image, detections, field_names)
                    
                    extracted_fields = {}
                    for field_name, field_img in cropped_fields.items():
                        if field_img is not None and field_name != 'Signature':
                            text = model.run_trocr(field_img)
                            extracted_fields[field_name] = text
                    
                    account_number = None
                    if 'Account number' in extracted_fields:
                        acct = re.sub(r'[^0-9]', '', extracted_fields['Account number'])
                        account_number = acct
                        extracted_fields['Account number'] = acct
                    
                    signature_match = False
                    signature_score = 0
                    extracted_signature_str = None
                    reference_signature_str = None
                    
                    if 'Signature' in cropped_fields and cropped_fields['Signature'] is not None:
                        sig_matcher = SignatureMatching(app.config['ORIGINAL_SIGNATURE_FOLDER'])
                        
                        signature_img = Image.fromarray(cropped_fields['Signature'])
                        sig_buf = BytesIO()
                        signature_img.save(sig_buf, format='PNG')
                        sig_buf.seek(0)
                        extracted_signature_str = base64.b64encode(sig_buf.getvalue()).decode('utf-8')
                        
                        if account_number:
                            ref_sig_path = get_signature_path_by_account(account_number)
                            
                            if ref_sig_path:
                                signature_score, _ = sig_matcher.find_best_signature_match(
                                    cropped_fields['Signature'], uploaded_signature_path=ref_sig_path)
                                signature_score *= 100
                                signature_match = signature_score > 80
                                
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
            if 'account_number' not in request.form or 'signature_file' not in request.files:
                flash('Both account number and signature file are required', 'error')
                return redirect(request.url)
            
            account_number = request.form['account_number']
            file = request.files['signature_file']
            
            if not account_number or account_number.strip() == '':
                flash('Account number is required', 'error')
                return redirect(request.url)
            
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            
            account_number = re.sub(r'[^0-9]', '', account_number)
            
            if account_number == '':
                flash('Account number must contain numeric digits', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{account_number}.{ext}"
                sig_dir = app.config['ORIGINAL_SIGNATURE_FOLDER']
                
                for existing in os.listdir(sig_dir):
                    if existing.startswith(f"{account_number}."):
                        os.remove(os.path.join(sig_dir, existing))
                
                filepath = os.path.join(sig_dir, filename)
                file.save(filepath)
                insert_signature(account_number, filepath)
                flash(f'Signature for account {account_number} saved!', 'success')
                return redirect(url_for('index'))
        
        return render_template('upload_signature.html')

    @app.route('/about')
    def about():
        return render_template('about.html')
    # Add this import at the top of routes.py
    from models import ChequeModel, SignatureMatching, BankingSystem, ChequeData
    from datetime import datetime
    from decimal import Decimal

    # Add this new route to your existing routes in routes.py
    from utils import clean_extracted_fields

    # Replace the field extraction section in the route:
    @app.route('/process-cheque-with-transaction', methods=['GET', 'POST'])
    def process_cheque_with_transaction():
        if not model:
            flash('Please set up the model first!', 'error')
            return redirect(url_for('setup'))
        
        if request.method == 'POST':
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
                    # Step 1: Basic cheque processing
                    cheque_quality = model.predict_cheque_quality(filepath)
                    image, detections = model.detect(filepath)
                    
                    box_annotator = sv.BoxAnnotator()
                    annotated_image = box_annotator.annotate(scene=image, detections=detections)
                    
                    annotated_pil = Image.fromarray(annotated_image)
                    buf = BytesIO()
                    annotated_pil.save(buf, format='PNG')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    # Step 2: Extract fields
                    field_names = ["micr", "Date", "Account number", "ifsc", "Amount",
                                "Signature", "from name", "payname", "Bank Name"]
                    cropped_fields = model.crop_fields(image, detections, field_names)
                    
                    # Extract text from fields (raw OCR)
                    raw_extracted_fields = {}
                    for field_name, field_img in cropped_fields.items():
                        if field_img is not None and field_name != 'Signature':
                            text = model.run_trocr(field_img)
                            raw_extracted_fields[field_name] = text
                    
                    # Clean the extracted fields
                    extracted_fields = clean_extracted_fields(raw_extracted_fields)
                    
                    # Step 3: Signature verification
                    signature_verified = False
                    signature_score = 0
                    extracted_signature_str = None
                    reference_signature_str = None
                    
                    # Use cleaned account number
                    account_number = extracted_fields.get('Account number', '')
                    
                    if 'Signature' in cropped_fields and cropped_fields['Signature'] is not None:
                        sig_matcher = SignatureMatching(app.config['ORIGINAL_SIGNATURE_FOLDER'])
                        
                        signature_img = Image.fromarray(cropped_fields['Signature'])
                        sig_buf = BytesIO()
                        signature_img.save(sig_buf, format='PNG')
                        sig_buf.seek(0)
                        extracted_signature_str = base64.b64encode(sig_buf.getvalue()).decode('utf-8')
                        
                        if account_number:
                            ref_sig_path = get_signature_path_by_account(account_number)
                            
                            if ref_sig_path:
                                signature_score, _ = sig_matcher.find_best_signature_match(
                                    cropped_fields['Signature'], uploaded_signature_path=ref_sig_path)
                                signature_score *= 100
                                signature_verified = signature_score > 80
                                
                                ref_img = Image.open(ref_sig_path)
                                ref_buf = BytesIO()
                                ref_img.save(ref_buf, format='PNG')
                                ref_buf.seek(0)
                                reference_signature_str = base64.b64encode(ref_buf.getvalue()).decode('utf-8')
                    
                    # Step 4: Transaction simulation
                    transaction_result = None
                    if cheque_quality == "good" and extracted_fields:
                        try:
                            # Parse cleaned data
                            amount_str = extracted_fields.get('Amount', '0')
                            amount = Decimal(amount_str) if amount_str else Decimal('0')
                            
                            # Parse date
                            date_str = extracted_fields.get('Date', '')
                            try:
                                # Try different date formats
                                if '/' in date_str:
                                    cheque_date = datetime.strptime(date_str, '%d/%m/%Y')
                                else:
                                    cheque_date = datetime.now()
                            except:
                                cheque_date = datetime.now()
                            
                            # Create cheque data object with cleaned fields
                            cheque_data = ChequeData(
                                cheque_number=f"CHQ{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                account_number=account_number,
                                payee_name=extracted_fields.get('payname', 'Unknown'),
                                amount=amount,
                                date=cheque_date,
                                micr_code=extracted_fields.get('micr', ''),
                                ifsc_code=extracted_fields.get('ifsc', ''),
                                bank_name=extracted_fields.get('Bank Name', ''),
                                from_name=extracted_fields.get('from name', '')
                            )
                            
                            # Simulate transaction
                            banking_system = BankingSystem()
                            transaction_result = banking_system.simulate_transaction(
                                cheque_data, signature_verified
                            )
                            
                        except Exception as e:
                            transaction_result = {
                                'success': False,
                                'errors': [f'Transaction simulation error: {str(e)}']
                            }
                    
                    # Show both raw and cleaned fields for debugging
                    return render_template('cheque_transaction_results.html',
                                        cheque_quality=cheque_quality,
                                        image_data=img_str,
                                        raw_extracted_fields=raw_extracted_fields,  # For debugging
                                        extracted_fields=extracted_fields,  # Cleaned fields
                                        extracted_signature=extracted_signature_str,
                                        reference_signature=reference_signature_str,
                                        signature_match=signature_verified,
                                        signature_score=signature_score,
                                        transaction_result=transaction_result)
                
                except Exception as e:
                    flash(f'Error processing cheque: {str(e)}', 'error')
                    return redirect(request.url)
        
        return render_template('process_cheque_transaction.html')

    @app.route('/setup-banking', methods=['GET', 'POST'])
    def setup_banking():
        if request.method == 'POST':
            try:
                from database import create_banking_tables, insert_sample_accounts
                create_banking_tables()
                insert_sample_accounts()
                flash('Banking system setup completed successfully!', 'success')
            except Exception as e:
                flash(f'Error setting up banking system: {str(e)}', 'error')
        
        return render_template('setup_banking.html')