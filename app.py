import pandas as pd
from typing import Optional, Dict, Any
import re
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid
import os
import signal
import sys
import time
from datetime import datetime

# Import your production grade modules 
from perfectpick.retrieval import ProductionHybridRetriever
from perfectpick.generation import DualModeRAGGenerator  
from perfectpick.session_memory import SessionMemoryDB
from langchain_groq import ChatGroq

# Import your existing services
from perfectpick.service import PerfectPickService
from perfectpick.config import Config
from utils.logger import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# Global variables for service instances
service = None
vector_store = None
df = None
retriever = None
memory_db = None
llm = None
generator = None

def initialize_services():
    """Initialize all services with proper error handling and retries"""
    global service, vector_store, df, retriever, memory_db, llm, generator
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing services (attempt {attempt + 1}/{max_retries})...")
            
            Config.validate()
            
            file_path = os.getenv("DATA_FILE_PATH", "data/Flipkart_Mobiles_cleaned.csv").strip('"')
            fallback_paths = [file_path]
            
            data_file = None
            for path in fallback_paths:
                if os.path.exists(path):
                    data_file = path
                    logger.info(f"Using data file: {path}")
                    break
            
            if not data_file:
                raise FileNotFoundError(f"No data file found in: {fallback_paths}")
            
            service = PerfectPickService(file_path=data_file, overwrite=False)
            vector_store = service.vector_store
            
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_file, encoding=encoding, on_bad_lines="skip", low_memory=False)
                    logger.info(f"Loaded data with {encoding}: {df.shape} rows")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("Failed to read data file with any encoding")
            
            logger.info(f"Data loaded successfully: {df.shape} rows, {len(df['Model'].unique())} unique models")
            
            retriever = ProductionHybridRetriever(vector_store=vector_store, docs_df=df)
            logger.info("ProductionHybridRetriever initialized")
            
            db_url = os.getenv("DB_URL", Config.DB_URL)
            try:
                memory_db = SessionMemoryDB(db_url, max_load=20)
                logger.info("SessionMemoryDB initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SessionMemoryDB: {e}")
                logger.warning("Continuing without session memory functionality")
                memory_db = None
            
            llm = ChatGroq(
                api_key=Config.GROQ_API_KEY, 
                model=Config.LLM_MODEL, 
                temperature=0.3,
                timeout=30,
                max_retries=2
            )
            logger.info("LLM initialized")
            
            # CHANGED: Initialize DUAL MODE generator
            generator = DualModeRAGGenerator(
                retriever=retriever, 
                llm=llm, 
                memory_db=memory_db
            )
            logger.info("✅ DualModeRAGGenerator initialized with card/explanation modes")
            
            logger.info("✅ All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service initialization failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("All service initialization attempts failed")
                return False

def extract_price_from_query(query: str) -> Optional[float]:
    """Extract price from query with multiple format support"""
    query_lower = query.lower()
    
    pattern1 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)'
    match1 = re.search(pattern1, query_lower)
    if match1:
        try:
            price_str = match1.group(2).replace(',', '').replace('.', '')
            price = float(price_str) * 100000
            logger.info(f"Extracted LAKH price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    pattern2 = r'\b(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)\b'
    match2 = re.search(pattern2, query_lower)
    if match2:
        try:
            price_str = match2.group(1).replace(',', '').replace('.', '')
            price = float(price_str) * 100000
            logger.info(f"Extracted LAKH price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    pattern3 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)(k|K|thousand)'
    match3 = re.search(pattern3, query_lower)
    if match3:
        try:
            price_str = match3.group(2).replace(',', '').replace('.', '')
            price = float(price_str) * 1000
            logger.info(f"Extracted THOUSAND price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    pattern4 = r'\b(\d+)\s*(k|K)\b'
    match4 = re.search(pattern4, query_lower)
    if match4:
        try:
            price = float(match4.group(1)) * 1000
            logger.info(f"Extracted THOUSAND price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    pattern5 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)'
    match5 = re.search(pattern5, query_lower)
    if match5:
        try:
            price_str = match5.group(2).replace(',', '').replace('.', '')
            price = float(price_str)
            logger.info(f"Extracted EXACT price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    return None

def process_query(query: str) -> str:
    """Enhanced query processing for better retrieval"""
    if not query or not query.strip():
        return query
    
    if len(query) > 300:
        query = query[:300]
        logger.info("Query truncated to 300 characters")
    
    price = extract_price_from_query(query)
    if price:
        logger.info(f"Price query detected (₹{price:,.0f}), keeping original: {query}")
        return query
    
    if len(query.split()) <= 8:
        return query
    
    logger.debug(f"Processing query: {query[:100]}...")
    
    keywords = []
    
    common_brands = ['samsung', 'oneplus', 'xiaomi', 'oppo', 'vivo', 'realme', 'apple', 'google', 'asus', 'poco', 'iqoo', 'nokia', 'motorola']
    for brand in common_brands:
        if brand in query.lower():
            keywords.append(brand)
    
    ram_match = re.search(r"(\d+)\s*(GB|gb)\s*(RAM|ram)", query.lower())
    if ram_match:
        keywords.append(f"{ram_match.group(1)}gb ram")
    
    storage_match = re.search(r"(\d+)\s*(GB|gb)\s*(storage|rom)", query.lower())
    if storage_match:
        keywords.append(f"{storage_match.group(1)}gb storage")
    
    for term in ['gaming', 'camera', 'battery', 'performance', 'budget', 'premium']:
        if term in query.lower():
            keywords.append(term)
    
    processed = " ".join(keywords) if keywords else query
    if processed != query:
        logger.info(f"Query processed: '{query}' -> '{processed}'")
    
    return processed

@app.before_request
def before_request():
    """Log request details"""
    if request.path != '/api/health':
        logger.info(f"Incoming request: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Log response details and add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    if request.path != '/api/health':
        logger.info(f"Response: {request.method} {request.path} - {response.status_code}")
    
    return response

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("10 per minute")
def recommend():
    """Main recommendation endpoint with dual mode support"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    try:
        if not request.is_json:
            logger.warning(f"[{request_id}] Invalid content type")
            return jsonify({
                "error": "Content-Type must be application/json",
                "request_id": request_id
            }), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning(f"[{request_id}] Missing query in request")
            return jsonify({
                "error": "Missing 'query' in request body",
                "request_id": request_id
            }), 400

        session_id = data.get("session_id") or str(uuid.uuid4())
        query = data['query'].strip()
        debug_mode = request.args.get('debug', 'false').lower() == 'true'
        
        logger.info(f"[{request_id}] Request: '{query}' (session={session_id})")

        if not query:
            return jsonify({
                "answer": "Please provide a specific question about mobile phones.\n\nExamples:\n- Phones under ₹15,000\n- Best camera phone\n- Gaming phones with 8GB RAM\n- OPPO A53 specifications",
                "query": "",
                "session_id": session_id,
                "sources": [],
                "request_id": request_id,
                "output_format": "explanation"
            }), 400

        if not all([service, generator]):
            logger.error(f"[{request_id}] Services not initialized")
            return jsonify({
                "error": "Service temporarily unavailable",
                "message": "Please try again in a few moments",
                "request_id": request_id
            }), 503

        processed_query = query
        
        result = generator.generate_answer(processed_query, session_id=session_id)
        
        # Determine output format for frontend
        output_format = "cards" if result.output_format == "card" else "explanation"
        
        response_data = {
            "query": query,
            "answer": result.answer,
            "sources": result.sources,
            "session_id": session_id,
            "resolved_query": result.resolved_query,
            "request_id": request_id,
            "processing_time": round(time.time() - start_time, 2),
            "output_format": output_format,
            "mode": result.output_format
        }
        
        if debug_mode:
            response_data["debug"] = result.debug_info

        logger.info(f"[{request_id}] Response: {output_format} mode, {len(result.sources)} sources, {len(result.answer)} chars, {response_data['processing_time']}s")
        return jsonify(response_data)
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"[{request_id}] API error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": "Unable to process your request at this time. Please try again.",
            "request_id": request_id,
            "processing_time": processing_time
        }), 500

@app.route('/health')
@limiter.exempt
def simple_health():
    """Simple health check for Kubernetes"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health():
    """Enhanced health check with comprehensive service status"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "services": {}
    }
    
    status_code = 200
    
    try:
        if df is not None:
            health_data["services"]["data"] = {
                "status": "healthy",
                "rows": len(df),
                "columns": len(df.columns),
                "models": len(df['Model'].unique())
            }
        else:
            health_data["services"]["data"] = {"status": "unhealthy"}
            status_code = 503
            
        if generator is not None:
            test_result_card = generator.generate_answer("Samsung Galaxy S23 specs", session_id="health-check")
            test_result_explanation = generator.generate_answer("Which phone is better for gaming?", session_id="health-check")
            
            health_data["services"]["generator"] = {
                "status": "healthy" if test_result_card and test_result_explanation else "degraded",
                "type": "DualModeRAGGenerator",
                "modes_supported": ["card", "explanation"],
                "card_mode_test": "passed" if test_result_card and test_result_card.answer else "failed",
                "explanation_mode_test": "passed" if test_result_explanation and test_result_explanation.answer else "failed"
            }
        else:
            health_data["services"]["generator"] = {"status": "unhealthy"}
            status_code = 503
            
        if memory_db is not None:
            try:
                test_session = "health-check-session"
                test_memory = memory_db.load_memory(test_session)
                health_data["services"]["memory"] = {
                    "status": "healthy",
                    "session_test": "passed"
                }
            except Exception as e:
                health_data["services"]["memory"] = {
                    "status": "degraded",
                    "error": str(e)
                }
        else:
            health_data["services"]["memory"] = {
                "status": "disabled",
                "message": "Session memory not available"
            }
            
        unhealthy_services = [svc for svc, status in health_data["services"].items() 
                            if status["status"] not in ["healthy", "disabled"]]
        if unhealthy_services:
            health_data["status"] = "degraded" if status_code == 200 else "unhealthy"
            
    except Exception as e:
        health_data.update({
            "status": "unhealthy",
            "error": str(e),
            "services": {"overall": "health_check_failed"}
        })
        status_code = 503
    
    return jsonify(health_data), status_code

@app.route('/api/session/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    """Clear session memory"""
    try:
        if generator and generator.clear_memory(session_id):
            logger.info(f"Session {session_id} cleared successfully")
            return jsonify({
                "message": f"Session {session_id} cleared successfully",
                "session_id": session_id
            })
        else:
            logger.warning(f"Session {session_id} not found or already cleared")
            return jsonify({
                "error": f"Session {session_id} not found or already cleared"
            }), 404
    except Exception as e:
        logger.error(f"Clear session failed: {e}")
        return jsonify({
            "error": "Failed to clear session",
            "session_id": session_id
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint - UPDATED for dual mode"""
    return jsonify({
        "name": "PerfectPick Phone Recommendation API",
        "version": "1.0.1",
        "description": "Dual-mode RAG-based phone recommendation system with AI-powered insights",
        "features": {
            "response_modes": ["cards", "explanation"],
            "card_mode": "Product specifications in structured card format",
            "explanation_mode": "Comparisons, advice, and detailed recommendations"
        },
        "endpoints": {
            "POST /api/recommend": "Get phone recommendations (auto-detects mode)",
            "GET /api/health": "Service health check",
            "DELETE /api/session/{id}": "Clear session memory",
            "GET /api/info": "This information"
        },
        "examples": {
            "card_queries": [
                "Samsung Galaxy S23 specs",
                "Phones with 8GB RAM under ₹20,000",
                "iPhone 15 Pro Max price and features"
            ],
            "explanation_queries": [
                "Which phone is better for gaming?",
                "Should I buy iPhone or Android?",
                "Recommend a good camera phone under ₹25,000",
                "What's the difference between AMOLED and LCD displays?"
            ]
        }
    })

def shutdown_handler(signum, frame):
    """Graceful shutdown"""
    logger.info(f"Shutdown signal {signum} received. Starting graceful shutdown...")
    
    try:
        if memory_db:
            memory_db.close_pool()
            logger.info("Memory DB connection pool closed")
    except Exception as e:
        logger.warning(f"Failed to close memory DB: {e}")
    
    logger.info("Shutdown complete")
    sys.exit(0)

# Initialize services on startup
if __name__ == "__main__":
    logger.info("🚀 Starting PerfectPick Phone Recommendation Service...")
    
    if not initialize_services():
        logger.error("❌ Service initialization failed. Exiting.")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"✅ Starting Flask server on http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )
else:
    if not initialize_services():
        logger.error("❌ Service initialization failed in WSGI mode")