import numpy as np
from skimage import io
import supervision as sv
from roboflow import Roboflow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
import os
import re
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
import psycopg2
from config import Config

class AccountStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    DORMANT = "dormant"

class TransactionType(Enum):
    DEBIT = "debit"
    CREDIT = "credit"

class TransactionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSED = "processed"

@dataclass
class Account:
    account_number: str
    account_holder: str
    balance: Decimal
    status: AccountStatus
    daily_limit: Decimal
    monthly_limit: Decimal
    last_transaction_date: Optional[datetime]
    micr_code: str
    ifsc_code: str
    frozen_reason: Optional[str] = None
    kyc_status: str = "pending"

@dataclass
class ChequeData:
    cheque_number: str
    account_number: str
    payee_name: str
    amount: Decimal
    date: datetime
    micr_code: str
    ifsc_code: str
    bank_name: str
    from_name: str

class ChequeModel:
    def __init__(self, api_key: str, workspace: str, project: str, version: int):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
        self.model = self.project.version(version).model

    def predict_cheque_quality(self, image_path: str, confidence: int = 40, overlap: int = 30) -> str:
        try:
            print(f"📂 Checking image: {image_path}")
            result = self.predict(image_path, confidence=confidence, overlap=overlap)
            detections = sv.Detections.from_inference(result)
            classes = detections.data["class_name"]
            print(f"🔍 Detected: {classes}")

            required = {
                "micr", "Date", "Account number", "ifsc", "Amount",
                "Signature", "from name", "payname", "Bank Name"
            }
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

    def detect(self, image_path: str, confidence: int = 40, overlap: int = 30):
        """Run inference and convert to a Supervision Detections object."""
        result = self.predict(image_path, confidence, overlap)
        image = io.imread(image_path)
        return image, sv.Detections.from_inference(result)

    def crop_fields(self, image: np.ndarray, detections: sv.Detections, field_names: list[str]) -> dict[str, np.ndarray | None]:
        """Safely crop each field in field_names from the detection results."""
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
    def run_trocr(image: np.ndarray | Image.Image, model_name: str = "microsoft/trocr-large-handwritten") -> str:
        """Run Microsoft TrOCR on one cropped image and return the decoded text."""
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
    """Provides methods to compare signatures using Structural Similarity Index."""
    
    def __init__(self, signature_dir: str):
        self.signature_dir = signature_dir

    @staticmethod
    def compare_signatures(ref_path: str, test_image: np.ndarray) -> float:
        """Compare reference and test signature images using SSIM."""
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if test_image.ndim == 3:
            test_img = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        else:
            test_img = test_image
            
        ref_img = cv2.resize(ref_img, (200, 100))
        test_img = cv2.resize(test_img, (200, 100))
        score, _ = ssim(ref_img, test_img, full=True)
        return score

    def find_best_signature_match(self, test_image: np.ndarray, uploaded_signature_path: str | None = None) -> tuple[float, str | None]:
        """Find best matching signature."""
        if uploaded_signature_path:
            score = self.compare_signatures(uploaded_signature_path, test_image)
            return score, uploaded_signature_path

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


class BankingSystem:
    """Comprehensive banking system for transaction simulation and validation."""
    
    def __init__(self):
        self.db_config = Config.DATABASE
    
    def get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config['host'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            dbname=self.db_config['dbname']
        )
    
    def get_account_by_micr(self, micr_code: str) -> Optional[Account]:
        """Retrieve account details using MICR code."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean MICR code
                clean_micr = re.sub(r'[^0-9]', '', micr_code)
                
                cursor.execute("""
                    SELECT account_number, account_holder, balance, status, 
                           daily_limit, monthly_limit, last_transaction_date,
                           micr_code, ifsc_code, frozen_reason, kyc_status
                    FROM banking_accounts 
                    WHERE micr_code = %s OR SIMILARITY(micr_code, %s) > 0.8
                    ORDER BY SIMILARITY(micr_code, %s) DESC
                    LIMIT 1
                """, (clean_micr, clean_micr, clean_micr))
                
                result = cursor.fetchone()
                
                if result:
                    return Account(
                        account_number=result[0],
                        account_holder=result[1],
                        balance=Decimal(str(result[2])),
                        status=AccountStatus(result[3]),
                        daily_limit=Decimal(str(result[4])),
                        monthly_limit=Decimal(str(result[5])),
                        last_transaction_date=result[6],
                        micr_code=result[7],
                        ifsc_code=result[8],
                        frozen_reason=result[9],
                        kyc_status=result[10]
                    )
                return None
                
        except Exception as e:
            print(f"Error retrieving account by MICR: {e}")
            return None
    
    def validate_account_status(self, account: Account) -> Dict:
        """Validate account status for transaction eligibility."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        if account.status == AccountStatus.FROZEN:
            validation_result['errors'].append(f"Account is frozen. Reason: {account.frozen_reason}")
        elif account.status == AccountStatus.SUSPENDED:
            validation_result['errors'].append("Account is suspended")
        elif account.status == AccountStatus.CLOSED:
            validation_result['errors'].append("Account is closed")
        elif account.status == AccountStatus.DORMANT:
            validation_result['warnings'].append("Account is dormant - may need reactivation")
        
        if account.kyc_status != 'verified':
            validation_result['errors'].append("KYC verification required")
        
        if account.last_transaction_date:
            days_inactive = (datetime.now() - account.last_transaction_date).days
            if days_inactive > 365:
                validation_result['warnings'].append(f"Account inactive for {days_inactive} days")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        return validation_result
    
    def check_balance_sufficiency(self, account: Account, amount: Decimal) -> Dict:
        """Check if account has sufficient balance."""
        balance_check = {
            'sufficient': False,
            'available_balance': account.balance,
            'required_amount': amount,
            'shortfall': Decimal('0')
        }
        
        if account.balance >= amount:
            balance_check['sufficient'] = True
        else:
            balance_check['shortfall'] = amount - account.balance
        
        return balance_check
    
    def check_transaction_limits(self, account: Account, amount: Decimal) -> Dict:
        """Check daily and monthly transaction limits."""
        limit_check = {
            'within_limits': True,
            'daily_remaining': Decimal('0'),
            'monthly_remaining': Decimal('0'),
            'errors': []
        }
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check daily limit
                today = datetime.now().date()
                cursor.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM transactions 
                    WHERE account_number = %s 
                    AND transaction_date::date = %s 
                    AND transaction_type = %s
                    AND status = %s
                """, (account.account_number, today, TransactionType.DEBIT.value, 
                      TransactionStatus.PROCESSED.value))
                
                daily_spent = Decimal(str(cursor.fetchone()[0]))
                daily_remaining = account.daily_limit - daily_spent
                
                if amount > daily_remaining:
                    limit_check['within_limits'] = False
                    limit_check['errors'].append(f"Daily limit exceeded. Remaining: ₹{daily_remaining}")
                
                # Check monthly limit
                month_start = datetime.now().replace(day=1).date()
                cursor.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM transactions 
                    WHERE account_number = %s 
                    AND transaction_date >= %s 
                    AND transaction_type = %s
                    AND status = %s
                """, (account.account_number, month_start, TransactionType.DEBIT.value,
                      TransactionStatus.PROCESSED.value))
                
                monthly_spent = Decimal(str(cursor.fetchone()[0]))
                monthly_remaining = account.monthly_limit - monthly_spent
                
                if amount > monthly_remaining:
                    limit_check['within_limits'] = False
                    limit_check['errors'].append(f"Monthly limit exceeded. Remaining: ₹{monthly_remaining}")
                
                limit_check['daily_remaining'] = daily_remaining
                limit_check['monthly_remaining'] = monthly_remaining
                
        except Exception as e:
            print(f"Error checking limits: {e}")
            limit_check['errors'].append("Limit validation failed")
            limit_check['within_limits'] = False
        
        return limit_check
    
    def simulate_transaction(self, cheque_data: ChequeData, signature_verified: bool) -> Dict:
        """Simulate the complete transaction process."""
        transaction_result = {
            'success': False,
            'transaction_id': None,
            'status': TransactionStatus.REJECTED,
            'errors': [],
            'warnings': [],
            'processing_time': datetime.now(),
            'account_info': None,
            'balance_info': None,
            'limit_info': None
        }
        
        try:
            # Step 1: Get account by MICR code
            account = self.get_account_by_micr(cheque_data.micr_code)
            
            if not account:
                transaction_result['errors'].append("Account not found with provided MICR code")
                return transaction_result
            
            transaction_result['account_info'] = {
                'account_number': account.account_number,
                'account_holder': account.account_holder,
                'status': account.status.value,
                'balance': float(account.balance)
            }
            
            # Step 2: Validate account status
            status_validation = self.validate_account_status(account)
            if not status_validation['valid']:
                transaction_result['errors'].extend(status_validation['errors'])
                transaction_result['warnings'].extend(status_validation['warnings'])
                return transaction_result
            
            # Step 3: Signature verification check
            if not signature_verified:
                transaction_result['errors'].append("Signature verification failed")
                return transaction_result
            
            # Step 4: Check balance sufficiency
            balance_check = self.check_balance_sufficiency(account, cheque_data.amount)
            transaction_result['balance_info'] = balance_check
            
            if not balance_check['sufficient']:
                transaction_result['errors'].append(
                    f"Insufficient balance. Available: ₹{balance_check['available_balance']}, "
                    f"Required: ₹{balance_check['required_amount']}, "
                    f"Shortfall: ₹{balance_check['shortfall']}"
                )
                return transaction_result
            
            # Step 5: Check transaction limits
            limit_check = self.check_transaction_limits(account, cheque_data.amount)
            transaction_result['limit_info'] = limit_check
            
            if not limit_check['within_limits']:
                transaction_result['errors'].extend(limit_check['errors'])
                return transaction_result
            
            # Step 6: Date validation
            cheque_date = cheque_data.date.date()
            current_date = datetime.now().date()
            
            if cheque_date > current_date:
                transaction_result['errors'].append("Post-dated cheque not allowed")
                return transaction_result
            
            if (current_date - cheque_date).days > 180:
                transaction_result['errors'].append("Stale cheque (older than 6 months)")
                return transaction_result
            
            # Step 7: Process transaction
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert transaction record
                cursor.execute("""
                    INSERT INTO transactions 
                    (cheque_number, account_number, payee_name, amount, 
                     transaction_type, transaction_date, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING transaction_id
                """, (cheque_data.cheque_number, cheque_data.account_number,
                      cheque_data.payee_name, cheque_data.amount,
                      TransactionType.DEBIT.value, datetime.now(),
                      TransactionStatus.APPROVED.value))
                
                transaction_id = cursor.fetchone()[0]
                transaction_result['transaction_id'] = transaction_id
                
                # Update account balance
                new_balance = account.balance - cheque_data.amount
                cursor.execute("""
                    UPDATE banking_accounts 
                    SET balance = %s, last_transaction_date = %s
                    WHERE account_number = %s
                """, (new_balance, datetime.now(), account.account_number))
                
                conn.commit()
                
                transaction_result['success'] = True
                transaction_result['status'] = TransactionStatus.PROCESSED
                transaction_result['balance_info']['new_balance'] = float(new_balance)
                
        except Exception as e:
            transaction_result['errors'].append(f"Transaction processing error: {str(e)}")
            print(f"Transaction error: {e}")
        
        return transaction_result


