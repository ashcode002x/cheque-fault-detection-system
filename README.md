
https://github.com/user-attachments/assets/ea4f2245-3a82-4de9-82a9-8837d543d8e3

# 🏦 Cheque Fault Detection System

A comprehensive AI-powered banking solution for automated cheque processing, validation, and transaction simulation with real-time fraud detection capabilities.

![image](https://github.com/user-attachments/assets/91cb8291-1c00-4f7d-9470-c6fa6ace711c)
![image](https://github.com/user-attachments/assets/af407593-ee1a-4307-a62a-e9dc5e43d875)
![image](https://github.com/user-attachments/assets/08792d89-2244-42f2-b86e-cad58716a256)
![image](https://github.com/user-attachments/assets/6acc81d5-0a26-4bfd-98bb-964808149d71)
![image](https://github.com/user-attachments/assets/f4cc3f43-f897-411d-9367-36284e189432)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)


## 🌟 Features

### 🔍 AI-Powered Cheque Analysis
- **Computer Vision Detection**: Advanced object detection using Roboflow AI models
- **OCR Text Extraction**: Microsoft TrOCR for handwritten and printed text recognition
- **Quality Assessment**: Automated good/bad cheque classification
- **Field Extraction**: Automatic detection of MICR code, account number, amount, date, signatures, and bank details

### ✍️ Signature Verification
- **SSIM-based Matching**: Structural Similarity Index for signature comparison
- **Reference Management**: Upload and manage account holder signatures
- **Real-time Verification**: Instant signature matching with confidence scores
- **Security Threshold**: Configurable matching thresholds for fraud prevention

### 💰 Banking Transaction Simulation
- **Account Validation**: Real-time account status and KYC verification
- **Balance Management**: Sufficient funds checking and balance updates
- **Transaction Limits**: Daily and monthly limit enforcement
- **MICR-based Lookup**: Account retrieval using MICR codes
- **Risk Assessment**: Automated fraud detection and payee validation

### 🛡️ Security & Compliance
- **Multi-layer Validation**: Account status, signature, balance, and limit checks
- **Date Validation**: Post-dated and stale cheque detection
- **Blacklist Management**: Payee risk assessment and blacklisting
- **Audit Trail**: Complete transaction logging and history

### 🎯 Model Evaluation
- **Batch Processing**: Evaluate model performance on multiple cheques
- **Accuracy Metrics**: Precision, recall, and F1-score calculations
- **Confusion Matrix**: Visual performance analysis
- **Comparative Analysis**: Good vs bad cheque classification results

## 🚀 Technology Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL with fuzzy matching capabilities
- **AI/ML**:
  - Roboflow (Object Detection)
  - Microsoft TrOCR (Optical Character Recognition)
  - OpenCV (Image Processing)
  - scikit-image (Computer Vision)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Deployment**: Docker & Docker Compose
- **Image Processing**: PIL, NumPy, Supervision

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Docker (optional but recommended)
- Roboflow API key

## 🔧 Installation

### Method 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashcode002x/cheque-fault-detection-system.git
   cd cheque-fault-detection-system
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   ```
   http://localhost:5000
   ```

### Method 2: Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashcode002x/cheque-fault-detection-system.git
   cd cheque-fault-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**
   ```bash
   createdb cheque_detection
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## ⚙️ Configuration

Update `config.py` with your settings:

```python
# Database Configuration
DATABASE_URL = "postgresql://username:password@localhost/cheque_detection"

# API Keys
ROBOFLOW_API_KEY = "your_roboflow_api_key"

# Security Settings
SIGNATURE_MATCH_THRESHOLD = 0.7
MAX_TRANSACTION_AMOUNT = 100000

# File Upload Settings
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## 🗄️ Database Setup

The system automatically creates required tables:

- `accounts` - Bank account information
- `signatures` - Reference signature storage
- `transactions` - Transaction history
- `blacklist` - Risk management
- `audit_logs` - System audit trail

## 📱 Usage

### 1. Model Setup
- Navigate to `/setup`
- Configure Roboflow API credentials
- Initialize the AI model

### 2. Banking System Setup
- Go to `/setup-banking`
- Create database tables
- Insert sample accounts for testing

### 3. Signature Management
- Upload reference signatures at `/upload-signature`
- Link signatures to account numbers

### 4. Process Cheques
- **Simple Processing**: `/process-cheque` (basic analysis)
- **Full Transaction**: `/process-cheque-with-transaction` (complete banking simulation)

### 5. Model Evaluation
- Use `/evaluate` for batch testing
- Upload good and bad cheque samples
- View performance metrics

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Flask Backend  │    │   PostgreSQL    │
│   (Bootstrap)   │◄──►│   (Python)      │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   AI Services   │
                    │  ┌─────────────┐ │
                    │  │ Roboflow AI │ │
                    │  │   TrOCR     │ │
                    │  │   OpenCV    │ │
                    │  └─────────────┘ │
                    └─────────────────┘
```

## 🔐 Security Features

- **Multi-factor Validation**: Signature + Account Status + Balance + Limits
- **MICR Code Verification**: Cross-validation with account details
- **Date Range Checking**: Prevents stale and post-dated cheques
- **Fuzzy Matching**: Handles OCR errors in MICR/account numbers
- **Risk Scoring**: Automated fraud detection algorithms

## 📊 Sample Results

| Metric | Good Cheques | Bad Cheques | Overall |
|--------|--------------|-------------|---------|
| Precision | 76.2% | 71.8% | 74.0% |
| Recall | 78.1% | 68.9% | 73.5% |
| F1-Score | 77.1% | 70.3% | 73.7% |

## 🚦 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process-cheque` | POST | Process single cheque |
| `/api/validate-signature` | POST | Verify signature match |
| `/api/account-lookup` | GET | Retrieve account details |
| `/api/transaction-history` | GET | Get transaction records |

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# End-to-end tests
python -m pytest tests/e2e/

# Coverage report
python -m pytest --cov=app tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For support and questions:

- 🐛 **Issues**: [GitHub Issues](https://github.com/ashcode002x/cheque-fault-detection-system/issues)
- 📖 **Documentation**: [Wiki](https://github.com/ashcode002x/cheque-fault-detection-system/wiki)

## 🚀 Roadmap

- [ ] Mobile app integration
- [ ] Real-time API endpoints
- [ ] Advanced fraud detection ML models
- [ ] Multi-language OCR support
- [ ] Blockchain integration for audit trails
- [ ] Real-time notifications
- [ ] Advanced analytics dashboard

## 📈 Performance

- **Processing Speed**: ~5 minutes per cheque
- **Accuracy**: 73.5% overall classification accuracy
- **Throughput**: 12+ cheques per hour
- **Uptime**: 99.9% availability

## 🔗 Related Projects

- [Banking API Integration](https://github.com/ashcode002x/banking-api)
- [OCR Text Recognition](https://github.com/ashcode002x/ocr-engine)
- [Signature Verification](https://github.com/ashcode002x/signature-match)

## 📚 References

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Microsoft TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [SSIM Algorithm](https://en.wikipedia.org/wiki/Structural_similarity)

---

⭐ **Star this repository if you find it helpful!**

Built with ❤️ for the banking industry

---

## 🏷️ Tags

`python` `flask` `ai` `machine-learning` `computer-vision` `ocr` `banking` `fintech` `fraud-detection` `signature-verification` `postgresql` `docker` `roboflow` `opencv`
