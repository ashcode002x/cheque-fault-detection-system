import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import base64
from io import BytesIO
from flask import current_app
import re
from decimal import Decimal

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def plot_confusion_matrix(evaluation_results):
    """Generate a confusion matrix as a base64 encoded image."""
    if not evaluation_results['true_labels'] or not evaluation_results['predicted_labels']:
        return None
    
    cm = confusion_matrix(
        evaluation_results['true_labels'], 
        evaluation_results['predicted_labels'], 
        labels=["good", "bad"]
    )
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Bad"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

def clean_micr_code(micr_text: str) -> str:
    """Clean and normalize MICR code by removing spaces and non-numeric characters."""
    if not micr_text:
        return ""
    
    # Remove all spaces and non-numeric characters except letters
    cleaned = re.sub(r'[^\d]', '', micr_text.strip())
    
    # MICR codes are typically 9 digits
    # If we have more than 9 digits, take the first 9
    if len(cleaned) > 9:
        cleaned = cleaned[:9]
    
    return cleaned

def clean_account_number(account_text: str) -> str:
    """Extract and clean account number from OCR text."""
    if not account_text:
        return ""
    
    # Remove common prefixes and text
    cleaned = account_text.upper().strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        'AVCNO.', 'ACC NO.', 'ACCOUNT NO.', 'A/C NO.', 
        'ACCOUNT NUMBER', 'ACC NUMBER', 'NO.'
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Extract only numeric characters
    account_number = re.sub(r'[^\d]', '', cleaned)
    
    return account_number

def clean_ifsc_code(ifsc_text: str) -> str:
    """Clean and normalize IFSC code."""
    if not ifsc_text:
        return ""
    
    # Remove common prefixes
    cleaned = ifsc_text.upper().strip()
    
    prefixes_to_remove = ['IFS CODE', 'IFSC CODE', 'IFSC', 'IFS', '-']
    for prefix in prefixes_to_remove:
        cleaned = cleaned.replace(prefix, '').strip()
    
    # Remove extra spaces and special characters, keep only alphanumeric
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    
    # IFSC codes are 11 characters (4 letters + 7 characters)
    if len(cleaned) >= 11:
        cleaned = cleaned[:11]
    
    return cleaned

def clean_amount(amount_text: str) -> str:
    """Extract and clean amount from OCR text."""
    if not amount_text:
        return "0"
    
    # Remove currency symbols and common prefixes
    cleaned = amount_text.strip()
    cleaned = re.sub(r'[₹$#]', '', cleaned)
    
    # Extract numbers, decimal points, and commas
    amount_match = re.search(r'[\d,]+\.?\d*', cleaned)
    
    if amount_match:
        amount_str = amount_match.group()
        # Remove commas
        amount_str = amount_str.replace(',', '')
        
        try:
            # Validate it's a valid number
            float(amount_str)
            return amount_str
        except ValueError:
            pass
    
    # If no valid amount found, try to extract just numbers
    numbers_only = re.sub(r'[^\d.]', '', cleaned)
    return numbers_only if numbers_only else "0"

def clean_date(date_text: str) -> str:
    """Clean and normalize date from OCR text."""
    if not date_text:
        return ""
    
    # Remove hash symbols and extra characters
    cleaned = date_text.strip().replace('#', '').replace('.', '/')
    
    # Look for date patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)
    date_patterns = [
        r'\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2})\b',   # DD/MM/YY
        r'\b(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})\b',   # YYYY/MM/DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, cleaned)
        if match:
            if len(match.group(3)) == 2:  # If year is 2 digits
                year = "20" + match.group(3)
                return f"{match.group(1)}/{match.group(2)}/{year}"
            else:
                return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    
    return cleaned

def clean_name(name_text: str) -> str:
    """Clean name fields from OCR text."""
    if not name_text:
        return ""
    
    # Remove common prefixes and clean up
    cleaned = name_text.strip()
    
    # Remove leading dashes and dots
    cleaned = re.sub(r'^[-.\s]+', '', cleaned)
    
    # Remove trailing dots and spaces
    cleaned = re.sub(r'[.\s]+$', '', cleaned)
    
    # Clean up extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.title()  # Convert to title case

def clean_bank_name(bank_text: str) -> str:
    """Clean bank name from OCR text."""
    if not bank_text:
        return ""
    
    cleaned = bank_text.strip()
    
    # Remove trailing dots and extra spaces
    cleaned = re.sub(r'[.\s]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.title()

def clean_extracted_fields(extracted_fields: dict) -> dict:
    """Apply cleaning functions to all extracted fields."""
    cleaned_fields = {}
    
    for field_name, field_value in extracted_fields.items():
        if not field_value:
            cleaned_fields[field_name] = ""
            continue
            
        field_lower = field_name.lower()
        
        if 'micr' in field_lower:
            cleaned_fields[field_name] = clean_micr_code(field_value)
        elif 'account' in field_lower:
            cleaned_fields[field_name] = clean_account_number(field_value)
        elif 'ifsc' in field_lower:
            cleaned_fields[field_name] = clean_ifsc_code(field_value)
        elif 'amount' in field_lower:
            cleaned_fields[field_name] = clean_amount(field_value)
        elif 'date' in field_lower:
            cleaned_fields[field_name] = clean_date(field_value)
        elif 'name' in field_lower:
            cleaned_fields[field_name] = clean_name(field_value)
        elif 'bank' in field_lower:
            cleaned_fields[field_name] = clean_bank_name(field_value)
        else:
            # For other fields, just clean whitespace
            cleaned_fields[field_name] = field_value.strip()
    
    return cleaned_fields