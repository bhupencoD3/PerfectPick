import pandas as pd
from typing import Optional
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

# Import your production grade modules
from flipkart.retrieval import ProductionHybridRetriever
from flipkart.generation import ProductionRAGGenerator
from flipkart.session_memory import SessionMemoryDB
from langchain_groq import ChatGroq

# Import your existing services
from flipkart.service import FlipkartRecommendationService
from flipkart.config import Config
from utils.logger import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Validate configuration
Config.validate()

# File path
file_path = os.getenv("DATA_FILE_PATH", "data/Flipkart_Mobiles_cleaned.csv")
if not os.path.exists(file_path):
    logger.error(f"Data file not found: {file_path}")
    raise FileNotFoundError(f"Data file not found: {file_path}")

# Initialize services
logger.info("Initializing services...")

try:
    # Load data and initialize vector store
    service = FlipkartRecommendationService(file_path=file_path, overwrite=False)
    vector_store = service.vector_store
    df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    logger.info(f"Loaded data: {df.shape} rows, Sample models: {df['Model'].unique()[:3]}")

    # Initialize production-grade retriever
    retriever = ProductionHybridRetriever(vector_store=vector_store, docs_df=df)
    logger.info("ProductionHybridRetriever initialized")

    # Initialize session memory
    memory_db = SessionMemoryDB(Config.DB_URL, max_load=10)
    logger.info("SessionMemoryDB initialized")

    # Initialize LLM
    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY, 
        model=Config.LLM_MODEL, 
        temperature=0.3
    )
    logger.info("LLM initialized")

    # Initialize production-grade generator
    generator = ProductionRAGGenerator(
        retriever=retriever, 
        llm=llm, 
        memory_db=memory_db
    )
    logger.info("ProductionRAGGenerator initialized")

except Exception as e:
    logger.error(f"Service initialization failed: {e}")
    raise

def extract_price_from_query(query: str) -> Optional[float]:
    """Extract price from query with multiple format support - FIXED for lakh"""
    query_lower = query.lower()
    
    # Pattern 1: "under 1 lakh", "below 2 lac"
    pattern1 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)'
    match1 = re.search(pattern1, query_lower)
    if match1:
        try:
            price_str = match1.group(2).replace(',', '').replace('.', '')
            price = float(price_str) * 100000  # Convert lakh to actual amount
            logger.info(f"Extracted LAKH price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    # Pattern 2: "1 lakh phones", "2 lac budget"
    pattern2 = r'\b(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)\b'
    match2 = re.search(pattern2, query_lower)
    if match2:
        try:
            price_str = match2.group(1).replace(',', '').replace('.', '')
            price = float(price_str) * 100000  # Convert lakh to actual amount
            logger.info(f"Extracted LAKH price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    # Pattern 3: "under 20k", "below 15k"
    pattern3 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)(k|K|thousand)'
    match3 = re.search(pattern3, query_lower)
    if match3:
        try:
            price_str = match3.group(2).replace(',', '').replace('.', '')
            price = float(price_str) * 1000  # Convert k to actual amount
            logger.info(f"Extracted THOUSAND price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    # Pattern 4: "20k phones", "15k budget"
    pattern4 = r'\b(\d+)\s*(k|K)\b'
    match4 = re.search(pattern4, query_lower)
    if match4:
        try:
            price = float(match4.group(1)) * 1000
            logger.info(f"Extracted THOUSAND price: ₹{price:,.0f}")
            return price
        except (ValueError, IndexError):
            pass
    
    # Pattern 5: "under 20000", "below 15000"
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
    """
    Enhanced query processing for better retrieval - SIMPLIFIED
    """
    if not query or not query.strip():
        return query
    
    # Handle very long queries
    if len(query) > 300:
        query = query[:300]
        logger.info("Query truncated to 300 characters")
    
    # For price queries, don't over-process - keep them simple
    price = extract_price_from_query(query)
    if price:
        logger.info(f"Price query detected (₹{price:,.0f}), keeping original: {query}")
        return query
    
    if len(query.split()) <= 8:  # Short queries don't need processing
        return query
    
    logger.debug(f"Processing query: {query[:100]}...")
    
    # Extract key components
    keywords = []
    
    # Brand detection
    common_brands = ['samsung', 'oneplus', 'xiaomi', 'oppo', 'vivo', 'realme', 'apple', 'google', 'asus', 'poco', 'iqoo', 'nokia', 'motorola']
    for brand in common_brands:
        if brand in query.lower():
            keywords.append(brand)
    
    # RAM requirements
    ram_match = re.search(r"(\d+)\s*(GB|gb)\s*(RAM|ram)", query.lower())
    if ram_match:
        keywords.append(f"{ram_match.group(1)}gb ram")
    
    # Storage requirements
    storage_match = re.search(r"(\d+)\s*(GB|gb)\s*(storage|rom)", query.lower())
    if storage_match:
        keywords.append(f"{storage_match.group(1)}gb storage")
    
    # Use cases
    for term in ['gaming', 'camera', 'battery', 'performance', 'budget', 'premium']:
        if term in query.lower():
            keywords.append(term)
    
    processed = " ".join(keywords) if keywords else query
    if processed != query:
        logger.info(f"Query processed: '{query}' -> '{processed}'")
    
    return processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("10 per minute")
def recommend():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        session_id = data.get("session_id") or str(uuid.uuid4())
        query = data['query'].strip()
        debug_mode = request.args.get('debug', 'false').lower() == 'true'
        
        logger.info(f"Request: '{query}' (session={session_id})")

        # Handle empty query
        if not query:
            return jsonify({
                "answer": "Please provide a specific question about mobile phones.\n\nExamples:\n- Phones under ₹15,000\n- Best camera phone\n- Gaming phones with 8GB RAM\n- OPPO A53 specifications",
                "query": "",
                "session_id": session_id,
                "sources": []
            }), 400

        # Process query for better retrieval - USE ORIGINAL QUERY for better results
        processed_query = query  # Use original query instead of processed
        
        # Generate answer using production generator
        result = generator.generate_answer(processed_query, session_id=session_id)
        
        # Convert dataclass to dict for JSON response
        response_data = {
            "query": query,  # Original query
            "answer": result.answer,
            "sources": result.sources,
            "session_id": session_id,
            "resolved_query": result.resolved_query
        }
        
        # Add debug info if requested
        if debug_mode:
            response_data["debug"] = result.debug_info

        logger.info(f"Response: {len(result.sources)} sources, {len(result.answer)} chars")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": "Unable to process your request at this time. Please try again.",
            "request_id": str(uuid.uuid4())[:8]
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Enhanced health check"""
    try:
        # Test basic functionality with a lakh query
        test_result = generator.generate_answer("phones under 1 lakh", session_id="health-check")
        rag_status = "healthy" if test_result and test_result.answer else "degraded"
        
        return jsonify({
            "status": "healthy",
            "rag_service": rag_status,
            "services": ["retriever", "generator", "memory_db", "llm"],
            "timestamp": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }), 503

@app.route('/api/session/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    """Clear session memory"""
    try:
        if generator.clear_memory(session_id):
            return jsonify({
                "message": f"Session {session_id} cleared successfully",
                "session_id": session_id
            })
        else:
            return jsonify({
                "error": f"Session {session_id} not found or already cleared"
            }), 404
    except Exception as e:
        logger.error(f"Clear session failed: {e}")
        return jsonify({
            "error": "Failed to clear session",
            "session_id": session_id
        }), 500

def shutdown_handler(signum, frame):
    """Graceful shutdown"""
    logger.info(f"Shutdown signal {signum} received...")
    try:
        memory_db.close()
        logger.info("Memory DB closed")
    except Exception as e:
        logger.warning(f"Failed to close memory DB: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == "__main__":
    logger.info("Starting Flask server on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)