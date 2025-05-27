import psycopg2
from config import Config

def get_db_connection():
    """Get database connection using config settings."""
    db_config = Config.DATABASE
    return psycopg2.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        dbname=db_config['dbname']
    )

def get_signature_path_by_account(account_number):
    """Get signature path by account number using fuzzy search."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT signature_path 
        FROM accounts 
        WHERE SIMILARITY(account_number::text, %s) > 0.3 
        ORDER BY SIMILARITY(account_number::text, %s) DESC 
        LIMIT 1;
    """
    
    cursor.execute(query, (account_number, account_number))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result[0] if result else None

def insert_signature(account_number, path):
    """Insert or update signature path for an account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO accounts (account_number, signature_path)
        VALUES (%s, %s)
        ON CONFLICT (account_number) 
        DO UPDATE SET signature_path = EXCLUDED.signature_path
    """
    cursor.execute(query, (account_number, path))
    conn.commit()
    cursor.close()
    conn.close()

def check_table():
    """Check if accounts table exists, create if not."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'accounts'
        )
    """)
    exists = cursor.fetchone()[0]
    
    if not exists:
        cursor.execute("""
            CREATE TABLE accounts (
                account_number VARCHAR(50) PRIMARY KEY,
                signature_path TEXT NOT NULL
            )
        """)
        conn.commit()
    
    cursor.close()
    conn.close()

def create_banking_tables():
    """Create banking tables for transaction simulation."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Enable pg_trgm extension for similarity functions
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        # Create banking_accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS banking_accounts (
                account_number VARCHAR(50) PRIMARY KEY,
                account_holder VARCHAR(255) NOT NULL,
                balance DECIMAL(15,2) DEFAULT 0.00,
                status VARCHAR(20) DEFAULT 'active',
                daily_limit DECIMAL(15,2) DEFAULT 50000.00,
                monthly_limit DECIMAL(15,2) DEFAULT 1000000.00,
                last_transaction_date TIMESTAMP,
                micr_code VARCHAR(50) UNIQUE,
                ifsc_code VARCHAR(20),
                frozen_reason TEXT,
                kyc_status VARCHAR(20) DEFAULT 'verified',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id SERIAL PRIMARY KEY,
                cheque_number VARCHAR(50),
                account_number VARCHAR(50),
                payee_name VARCHAR(255),
                amount DECIMAL(15,2),
                transaction_type VARCHAR(20),
                transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20),
                risk_score INTEGER DEFAULT 0
            )
        """)
        
        # Create blacklisted_payees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blacklisted_payees (
                id SERIAL PRIMARY KEY,
                payee_name VARCHAR(255) NOT NULL,
                reason TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Banking tables created successfully")
        
    except Exception as e:
        print(f"❌ Error creating banking tables: {e}")

def insert_sample_accounts():
    """Insert sample accounts for testing."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sample_accounts = [
            ('1234567890', 'John Doe', 50000.00, 'active', 50000.00, 1000000.00, 
             '123456789012345', 'SBIN0001234', 'verified'),
            ('9876543210', 'Jane Smith', 75000.00, 'active', 25000.00, 500000.00, 
             '987654321098765', 'HDFC0005678', 'verified'),
            ('5555666677', 'Bob Johnson', 30000.00, 'frozen', 30000.00, 600000.00, 
             '555566667777888', 'ICIC0009876', 'verified'),
        ]
        
        for account in sample_accounts:
            cursor.execute("""
                INSERT INTO banking_accounts 
                (account_number, account_holder, balance, status, daily_limit, 
                 monthly_limit, micr_code, ifsc_code, kyc_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (account_number) DO NOTHING
            """, account)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Sample accounts inserted successfully")
        
    except Exception as e:
        print(f"❌ Error inserting sample accounts: {e}")