import torch
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    text: str
    source: str
    metadata: Dict[str, Any]
    score: float = 0.0

class ProductionHybridRetriever:
    def __init__(self, vector_store, docs_df: pd.DataFrame, 
                 reranker_model: str = "BAAI/bge-reranker-base",
                 bm25_weight: float = 0.4,
                 semantic_weight: float = 0.6):
        """
        Enhanced production-grade hybrid retriever with CURATED DATA injection and FIXED price filtering
        """
        self.vector_store = vector_store
        self.docs_df = docs_df.copy()
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Initialize curated data
        self._init_curated_data()
        
        # Analyze price distribution in the data
        self._analyze_price_distribution()
        
        # Domain knowledge for query enhancement
        self._init_domain_knowledge()
        
        # Initialize BM25 with curated data
        self._init_bm25()
        
        # Initialize reranker
        self._init_reranker(reranker_model)
        
        # Enhanced query patterns
        self._init_query_patterns()
        
        logger.info(f"ProductionHybridRetriever initialized with {len(docs_df)} products + {len(self.curated_data)} curated phones")

    def _init_curated_data(self):
        """Initialize curated phone data for exact matching"""
        self.curated_data = [
            # POPULAR PHONES
            {
                "name": "iPhone 17 Pro", "description": "Latest flagship with revolutionary AI features",
                "specs": "A19 Pro chip, 6.3-inch Always-On ProMotion XDR, 48MP quad camera with periscope zoom, 256GB storage",
                "price": 119999, "release_date": "2025-09-19", "source": "flipkart",
                "category": "popular", "tags": ["popular", "trending", "best selling", "flagship", "new", "latest"]
            },
            {
                "name": "Samsung Galaxy S25 Ultra", "description": "AI-powered smartphone with enhanced S-Pen", 
                "specs": "Snapdragon 8 Gen 4, 7.1-inch Dynamic AMOLED 3X, 250MP camera with 10x optical zoom, 512GB storage",
                "price": 149999, "release_date": "2025-02-14", "source": "flipkart",
                "category": "popular", "tags": ["popular", "trending", "flagship", "ai", "s-pen"]
            },
            {
                "name": "Google Pixel 9 Pro", "description": "Foldable Pixel with advanced Gemini AI",
                "specs": "Google Tensor G5, 7.6-inch foldable LTPO, 5x telephoto, 16GB RAM, 512GB storage", 
                "price": 139999, "release_date": "2025-08-15", "source": "flipkart",
                "category": "popular", "tags": ["popular", "foldable", "ai", "google", "latest"]
            },
            {
                "name": "OnePlus 13", "description": "Flagship with next-gen fast charging",
                "specs": "Snapdragon 8 Gen 4, 6.82-inch LTPO AMOLED, 50MP triple camera, 150W charging, 256GB storage",
                "price": 79999, "release_date": "2025-01-28", "source": "flipkart", 
                "category": "popular", "tags": ["popular", "fast charging", "flagship", "oneplus"]
            },
            {
                "name": "Xiaomi 15 Ultra", "description": "Flagship with Leica photography partnership",
                "specs": "Snapdragon 8 Gen 4, 6.78-inch LTPO AMOLED, 50MP Leica quad camera, 120W HyperCharge",
                "price": 89999, "release_date": "2025-03-21", "source": "flipkart",
                "category": "popular", "tags": ["popular", "leica", "camera", "flagship"]
            },
            
            # NEW PHONES THIS MONTH
            {
                "name": "iPhone 17 Pro", "description": "Latest flagship with revolutionary AI features",
                "specs": "A19 Pro chip, 6.3-inch Always-On ProMotion XDR, 48MP quad camera with periscope zoom, 256GB storage",
                "price": 119999, "release_date": "2025-09-19", "source": "flipkart", 
                "category": "new", "tags": ["new", "latest", "september", "2025", "just launched", "recent"]
            },
            {
                "name": "Google Pixel 9 Pro", "description": "Foldable Pixel with advanced Gemini AI", 
                "specs": "Google Tensor G5, 7.6-inch foldable LTPO, 5x telephoto, 16GB RAM, 512GB storage",
                "price": 139999, "release_date": "2025-08-15", "source": "flipkart",
                "category": "new", "tags": ["new", "foldable", "september", "latest", "google"]
            },
            
            # TOP SELLING PHONES
            {
                "name": "Samsung Galaxy A25", "description": "Best-selling budget smartphone with 5G",
                "specs": "Exynos 1480, 6.6-inch Super AMOLED, 108MP triple camera, 6000mAh battery", 
                "price": 22999, "release_date": "2024-12-10", "source": "flipkart",
                "category": "top_selling", "tags": ["top selling", "best seller", "popular", "bestseller", "high sales"]
            },
            {
                "name": "iPhone 16", "description": "Popular mainstream iPhone with Action button", 
                "specs": "A18 Bionic chip, 6.3-inch Super Retina XDR, 48MP dual camera, 128GB storage",
                "price": 79999, "release_date": "2024-09-20", "source": "flipkart",
                "category": "top_selling", "tags": ["top selling", "best seller", "popular", "iphone", "high sales"]
            },
            {
                "name": "OnePlus 13", "description": "Flagship with next-gen fast charging",
                "specs": "Snapdragon 8 Gen 4, 6.82-inch LTPO AMOLED, 50MP triple camera, 150W charging, 256GB storage",
                "price": 79999, "release_date": "2025-01-28", "source": "flipkart",
                "category": "top_selling", "tags": ["top selling", "popular", "best seller", "oneplus"]
            },
            
            # FLIP PHONES
            {
                "name": "Samsung Galaxy Z Flip7", "description": "Latest compact foldable flip phone",
                "specs": "Snapdragon 8 Gen 3, 6.9-inch foldable AMOLED, 4.0-inch cover display, 200MP main camera, 512GB storage",
                "price": 109999, "release_date": "2025-08-11", "source": "flipkart",
                "category": "flip", "tags": ["flip", "foldable", "flip phone", "samsung", "compact", "z flip"]
            },
            {
                "name": "Motorola Razr 50 Ultra", "description": "Premium flip phone with AI features", 
                "specs": "Snapdragon 8 Gen 3, 7.0-inch foldable pOLED, 4.5-inch cover display, 200MP main camera",
                "price": 99999, "release_date": "2025-07-18", "source": "flipkart",
                "category": "flip", "tags": ["flip", "foldable", "razr", "motorola", "flip phone"]
            },
            {
                "name": "Oppo Find N5 Flip", "description": "Latest vertical flip phone with improved hinge",
                "specs": "MediaTek Dimensity 9300, 6.8-inch foldable AMOLED, 3.5-inch cover display, 50MP triple camera", 
                "price": 84999, "release_date": "2025-06-12", "source": "flipkart",
                "category": "flip", "tags": ["flip", "foldable", "oppo", "vertical flip", "flip phone"]
            },
            {
                "name": "Tecno Phantom V Flip 2", "description": "Affordable flip phone with premium features",
                "specs": "MediaTek Dimensity 8300, 6.9-inch foldable AMOLED, 2.0-inch cover display, 108MP main camera",
                "price": 44999, "release_date": "2025-05-20", "source": "flipkart",
                "category": "flip", "tags": ["flip", "foldable", "affordable", "tecno", "flip phone", "budget flip"]
            },
        ]

    def _analyze_price_distribution(self):
        """Analyze price distribution in the dataset"""
        try:
            prices = []
            for _, row in self.docs_df.iterrows():
                price = self._extract_price(row.get('Selling Price'))
                if price != float('inf') and price > 0:
                    prices.append(price)
            
            if prices:
                self.min_price = min(prices)
                self.max_price = max(prices)
                self.avg_price = np.mean(prices)
                
                # Calculate price ranges for better filtering
                self.price_ranges = {
                    'budget': (0, 15000),
                    'mid_range': (15001, 30000),
                    'premium': (30001, 70000),
                    'flagship': (70001, self.max_price)
                }
                
                logger.info(f"Price distribution - Min: ₹{self.min_price:,.0f}, Max: ₹{self.max_price:,.0f}, Avg: ₹{self.avg_price:,.0f}")
                logger.info(f"Price ranges - Budget: ₹0-15K, Mid-range: ₹15-30K, Premium: ₹30-70K, Flagship: ₹70K+")
            else:
                self.min_price = 0
                self.max_price = 100000
                self.avg_price = 20000
                self.price_ranges = {
                    'budget': (0, 15000),
                    'mid_range': (15001, 30000),
                    'premium': (30001, 70000),
                    'flagship': (70001, 100000)
                }
                logger.warning("No valid prices found in dataset")
                
        except Exception as e:
            logger.error(f"Price analysis failed: {e}")
            self.min_price = 0
            self.max_price = 100000
            self.avg_price = 20000
            self.price_ranges = {
                'budget': (0, 15000),
                'mid_range': (15001, 30000),
                'premium': (30001, 70000),
                'flagship': (70001, 100000)
            }

    def _init_domain_knowledge(self):
        """Initialize domain-specific knowledge for better recommendations"""
        self.domain_knowledge = {
            'gaming': {
                'brands': ['ASUS', 'POCO', 'IQOO', 'OnePlus', 'Realme', 'Nubia'],
                'features': ['high RAM', 'gaming processor', 'cooling system', 'high refresh rate'],
                'min_ram': 6,
                'min_rating': 4.0
            },
            'camera': {
                'brands': ['Apple', 'Samsung', 'Google', 'OnePlus', 'OPPO', 'Vivo'],
                'features': ['multiple cameras', 'high megapixel', 'OIS', 'night mode'],
                'min_rating': 4.2
            },
            'battery': {
                'brands': ['Xiaomi', 'Realme', 'Samsung', 'Motorola'],
                'features': ['fast charging', 'large battery', 'power saving'],
                'min_rating': 4.0
            },
            'budget': {
                'brands': ['Redmi', 'Realme', 'Samsung', 'OPPO', 'Vivo'],
                'features': ['value for money', 'basic features'],
                'max_price': 15000
            },
            'premium': {
                'brands': ['Apple', 'Samsung', 'OnePlus', 'Google'],
                'features': ['premium build', 'best camera', 'high performance'],
                'min_price': 30000,
                'max_price': 70000
            },
            'flagship': {
                'brands': ['Apple', 'Samsung', 'Google'],
                'features': ['flagship', 'premium', 'best camera', 'high performance'],
                'min_price': 70000
            }
        }

    def _init_query_patterns(self):
        """Initialize comprehensive query patterns"""
        self.price_patterns = [
            r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)?',
            r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)(k|K|thousand)?',
            r'(\d+[.,]?\d*)\s*-\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)?',
            r'(\d+[.,]?\d*)\s*-\s*(\d+[.,]?\d*)\s*(k|K|thousand)?',
            r'budget\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)?',
            r'budget\s*(\d+[.,]?\d*)(k|K|thousand)?',
            r'[₹]?\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh).*(phone|mobile)',
            r'[₹]?\s*(\d+[.,]?\d*)(k|K|thousand).*(phone|mobile)',
            r'best.*(\d+)\s*(lakh|lac|L|Lakh)',
            r'best.*(\d+)\s*(k|K)'
        ]
        
        self.rating_patterns = [
            r'rating\s*(above|over|>|at least|more than)\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*stars?\s*(and above|and up|or more)',
            r'high\s*rated',
            r'good\s*rating'
        ]
        
        self.ram_patterns = [
            r'(\d+)\s*GB?\s*RAM',
            r'RAM\s*(\d+)',
            r'(\d+)\s*GB?\s*memory',
            r'memory\s*(\d+)'
        ]
        
        self.storage_patterns = [
            r'(\d+)\s*GB?\s*storage',
            r'storage\s*(\d+)',
            r'(\d+)\s*GB?\s*ROM',
            r'ROM\s*(\d+)'
        ]
        
        self.use_case_patterns = {
            'gaming': r'\b(gaming|games|pubg|cod|fortnite|gamer)\b',
            'camera': r'\b(camera|photo|picture|photography|selfie|video)\b',
            'battery': r'\b(battery|backup|charging|power)\b',
            'performance': r'\b(performance|fast|speed|smooth|lag.free)\b',
            'budget': r'\b(budget|cheap|affordable|economical|low.price)\b',
            'premium': r'\b(premium|flagship|high.end|luxury|best)\b'
        }

    def _init_bm25(self):
        """Initialize BM25 with enhanced text corpus including CURATED DATA"""
        try:
            self.bm25_corpus = []
            self.bm25_metadata = []  # Store metadata for each document
            
            # 1. Add main dataset
            for idx, row in self.docs_df.iterrows():
                doc_terms = self._create_document_terms(row)
                self.bm25_corpus.append(doc_terms)
                self.bm25_metadata.append({
                    "type": "main",
                    "index": idx,
                    "source": "dataset"
                })
            
            # 2. Add CURATED DATA with SUPER-ENHANCED terms
            for phone in self.curated_data:
                curated_terms = self._create_curated_document_terms(phone)
                self.bm25_corpus.append(curated_terms)
                self.bm25_metadata.append({
                    "type": "curated", 
                    "phone": phone,
                    "source": "curated"
                })
            
            self.bm25 = BM25Okapi(self.bm25_corpus)
            logger.info(f"BM25 initialized with {len(self.bm25_corpus)} documents ({len(self.curated_data)} curated)")
            
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")
            raise

    def _create_curated_document_terms(self, phone: Dict) -> List[str]:
        """Create SUPER-ENHANCED terms for curated phones"""
        terms = [
            phone["name"],
            phone["description"],
            phone["specs"],
            f"price ₹{phone['price']}",
            f"released {phone['release_date']}",
            "mobile phone smartphone"
        ]
        
        # Add category-specific terms
        category = phone.get("category", "")
        if category == "popular":
            terms.extend(["popular", "trending", "best selling", "top selling", "in demand", "hot", "viral"])
        elif category == "new":
            terms.extend(["new", "latest", "recent", "just launched", "just released", "fresh", "brand new"])
        elif category == "top_selling":
            terms.extend(["top selling", "best seller", "high sales", "most sold", "popular", "bestseller"])
        elif category == "flip":
            terms.extend(["flip", "foldable", "fold", "flip phone", "clamshell", "razr", "z flip"])
        
        # Add custom tags
        terms.extend(phone.get("tags", []))
        
        # Add brand and model variations
        name_lower = phone["name"].lower()
        if "iphone" in name_lower:
            terms.extend(["apple", "ios", "iphone"])
        if "samsung" in name_lower:
            terms.extend(["samsung", "android", "galaxy"])
        if "pixel" in name_lower:
            terms.extend(["google", "pixel", "android"])
        if "oneplus" in name_lower:
            terms.extend(["oneplus", "android"])
        if "xiaomi" in name_lower:
            terms.extend(["xiaomi", "mi", "android"])
        
        # Add price range terms
        price = phone["price"]
        if price <= 30000:
            terms.extend(["budget", "affordable", "cheap", "economical"])
        elif price <= 70000:
            terms.extend(["mid-range", "mid range", "premium"])
        else:
            terms.extend(["flagship", "premium", "expensive", "luxury"])
        
        # Split specs into individual terms
        specs_terms = phone["specs"].lower().replace(",", " ").split()
        terms.extend(specs_terms)
        
        # Split description into individual terms  
        desc_terms = phone["description"].lower().split()
        terms.extend(desc_terms)
        
        # Remove duplicates and clean
        unique_terms = list(dict.fromkeys([str(term).lower() for term in terms if term and str(term).strip()]))
        return unique_terms

    def _create_document_terms(self, row):
        """Create terms for main dataset documents (your existing logic)"""
        doc_terms = [
            str(row.get("Brand", "")),
            str(row.get("Model", "")),
            str(row.get("Color", "")),
            str(row.get("Memory", "")).replace("GB RAM", "GB RAM memory"),
            str(row.get("Storage", "")).replace("GB ROM", "GB storage"),
            f"rating {row.get('Rating', '')} stars",
            f"price ₹{row.get('Selling Price', '')}",
            "mobile phone smartphone"
        ]
        
        # Your existing domain-specific tagging logic...
        brand = str(row.get("Brand", "")).lower()
        memory = str(row.get("Memory", ""))
        price = self._extract_price(row.get('Selling Price', 0))
        
        # Gaming tags
        if any(gaming_brand in brand for gaming_brand in ['asus', 'poco', 'iqoo', 'nubia']):
            doc_terms.extend(["gaming", "performance"])
        elif 'GB' in memory:
            ram_match = re.search(r'(\d+)', memory)
            if ram_match and int(ram_match.group(1)) >= 8:
                doc_terms.extend(["gaming", "high performance"])
        
        # Camera tags
        if any(cam_brand in brand for cam_brand in ['apple', 'samsung', 'google', 'oppo', 'vivo']):
            doc_terms.extend(["camera", "photography"])
        
        # Price range tags
        if price <= 15000 and price != float('inf'):
            doc_terms.extend(["budget", "affordable", "under 15000", "cheap", "economical"])
        elif price <= 30000 and price != float('inf'):
            doc_terms.extend(["mid-range", "under 30000", "mid range", "30k"])
        elif price <= 50000 and price != float('inf'):
            doc_terms.extend(["premium", "under 50000", "50k", "mid premium"])
        elif price <= 70000 and price != float('inf'):
            doc_terms.extend(["premium", "under 70000", "70k", "high end"])
        elif price <= 100000 and price != float('inf'):
            doc_terms.extend(["flagship", "under 1 lakh", "1 lac", "premium flagship"])
        elif price > 100000 and price != float('inf'):
            doc_terms.extend(["ultra premium", "above 1 lakh", "expensive", "luxury"])
        
        doc_text = " ".join(doc_terms).lower()
        return doc_text.split()

    def _init_reranker(self, model_name: str):
        """Initialize reranker model with enhanced error handling"""
        try:
            if not hasattr(ProductionHybridRetriever, '_reranker'):
                logger.info(f"Loading reranker model: {model_name}")
                ProductionHybridRetriever._tokenizer = AutoTokenizer.from_pretrained(model_name)
                ProductionHybridRetriever._reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
                ProductionHybridRetriever._device = "cuda" if torch.cuda.is_available() else "cpu"
                ProductionHybridRetriever._reranker.to(ProductionHybridRetriever._device)
                ProductionHybridRetriever._reranker.eval()
                logger.info(f"Reranker loaded on {ProductionHybridRetriever._device}")
            
            self.tokenizer = ProductionHybridRetriever._tokenizer
            self.reranker = ProductionHybridRetriever._reranker
            self.device = ProductionHybridRetriever._device
            
        except Exception as e:
            logger.error(f"Reranker initialization failed: {e}")
            self.reranker = None
            self.tokenizer = None

    def _extract_price_from_query(self, query: str) -> Optional[float]:
        """Extract price from query with multiple format support - FIXED for lakh"""
        query_lower = query.lower()
        
        # Pattern 1: "under 1 lakh", "below 2 lac", "phones under 1.5 lakh"
        pattern1 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)\s*(lakh|lac|L|Lakh)'
        match1 = re.search(pattern1, query_lower)
        if match1:
            try:
                price_str = match1.group(2).replace(',', '').replace('.', '')
                price = float(price_str) * 100000  # Convert lakh to actual amount
                logger.info(f"Extracted LAKH price from pattern1: ₹{price:,.0f}")
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
                logger.info(f"Extracted LAKH price from pattern2: ₹{price:,.0f}")
                return price
            except (ValueError, IndexError):
                pass
        
        # Pattern 3: "under 20k", "below 15k" - THOUSANDS
        pattern3 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)(k|K|thousand)'
        match3 = re.search(pattern3, query_lower)
        if match3:
            try:
                price_str = match3.group(2).replace(',', '').replace('.', '')
                price = float(price_str) * 1000  # Convert k to actual amount
                logger.info(f"Extracted THOUSAND price from pattern3: ₹{price:,.0f}")
                return price
            except (ValueError, IndexError):
                pass
        
        # Pattern 4: "20k phones", "15k budget" - THOUSANDS
        pattern4 = r'\b(\d+)\s*(k|K)\b'
        matches4 = re.findall(pattern4, query_lower)
        if matches4:
            try:
                # Take the first price found
                price = float(matches4[0][0]) * 1000
                logger.info(f"Extracted THOUSAND price from pattern4: ₹{price:,.0f}")
                return price
            except (ValueError, IndexError):
                pass
        
        # Pattern 5: "under 20000", "below 15000" - EXACT AMOUNT
        pattern5 = r'(under|below|less than|upto|within)\s*[₹]?\s*(\d+[.,]?\d*)'
        match5 = re.search(pattern5, query_lower)
        if match5:
            try:
                price_str = match5.group(2).replace(',', '').replace('.', '')
                price = float(price_str)
                logger.info(f"Extracted EXACT price from pattern5: ₹{price:,.0f}")
                return price
            except (ValueError, IndexError):
                pass
        
        # Pattern 6: "around 30000", "budget 25000" - EXACT AMOUNT
        pattern6 = r'(around|about|budget|max|maximum)\s*[₹]?\s*(\d+[.,]?\d*)'
        match6 = re.search(pattern6, query_lower)
        if match6:
            try:
                price_str = match6.group(2).replace(',', '').replace('.', '')
                price = float(price_str)
                logger.info(f"Extracted EXACT price from pattern6: ₹{price:,.0f}")
                return price
            except (ValueError, IndexError):
                pass
        
        logger.info(f"No price extracted from query: {query}")
        return None

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Enhanced filter extraction with better pattern matching"""
        filters = {}
        query_lower = query.lower()
        
        # Extract price using dedicated method
        price = self._extract_price_from_query(query)
        if price:
            # FIXED: Different price buffer based on amount
            if price >= 50000:  # For high amounts (50k+), use smaller buffer
                filters['max_price'] = price * 1.05  # 5% buffer for high amounts
                filters['min_price'] = price * 0.7   # Ensure we don't get very cheap phones
            elif price >= 20000:  # For medium amounts (20k-50k)
                filters['max_price'] = price * 1.1   # 10% buffer
                filters['min_price'] = price * 0.6   # Reasonable lower bound
            else:  # For low amounts (<20k)
                filters['max_price'] = price * 1.15  # 15% buffer for budget phones
            
            logger.info(f"Price filter: ₹{filters.get('min_price', 0):,.0f} - ₹{filters['max_price']:,.0f}")
        
        # Rating filters
        for pattern in self.rating_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                try:
                    rating_str = match.group(2) if match.group(2) else match.group(1)
                    rating = float(rating_str)
                    if 1 <= rating <= 5:
                        filters['min_rating'] = rating
                        logger.info(f"Rating filter: min {rating} stars")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Rating extraction failed: {e}")
        
        # RAM filters
        for pattern in self.ram_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                try:
                    ram = int(match.group(1))
                    if 1 <= ram <= 16:  # Reasonable RAM range
                        filters['min_ram'] = ram
                        logger.info(f"RAM filter: min {ram}GB")
                except (ValueError, IndexError) as e:
                    logger.warning(f"RAM extraction failed: {e}")
        
        # Storage filters
        for pattern in self.storage_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                try:
                    storage = int(match.group(1))
                    if 16 <= storage <= 512:  # Reasonable storage range
                        filters['min_storage'] = storage
                        logger.info(f"Storage filter: min {storage}GB")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Storage extraction failed: {e}")
        
        # Use case based filters
        for use_case, pattern in self.use_case_patterns.items():
            if re.search(pattern, query_lower):
                knowledge = self.domain_knowledge.get(use_case, {})
                if 'min_ram' in knowledge and ('min_ram' not in filters or filters['min_ram'] < knowledge['min_ram']):
                    filters['min_ram'] = knowledge['min_ram']
                if 'min_rating' in knowledge and ('min_rating' not in filters or filters['min_rating'] < knowledge['min_rating']):
                    filters['min_rating'] = knowledge['min_rating']
                if 'max_price' in knowledge and ('max_price' not in filters or filters['max_price'] > knowledge['max_price']):
                    filters['max_price'] = knowledge['max_price']
                if 'min_price' in knowledge and ('min_price' not in filters or filters['min_price'] < knowledge['min_price']):
                    filters['min_price'] = knowledge['min_price']
        
        return filters

    def _enhance_query_with_domain_knowledge(self, query: str) -> str:
        """Enhanced query processing with ULTRA-AGGRESSIVE CURATED DATA patterns"""
        original_query = query
        enhanced_terms = []
        
        query_lower = query.lower()
        
        # ULTRA-AGGRESSIVE CURATED DATA PATTERN MATCHING
        if any(term in query_lower for term in ['popular', 'trending', 'best selling', 'top selling']):
            enhanced_terms.extend(['popular', 'trending', 'best selling', 'top selling', 'in demand', 'hot', 'viral'])
            # Add ALL curated phone names for exact matching
            popular_phones = [phone["name"] for phone in self.curated_data]
            enhanced_terms.extend(popular_phones)
            logger.info("Detected POPULAR phones query - adding ALL curated phone names")
        
        if any(term in query_lower for term in ['new', 'latest', 'recent', 'just launched', 'this month']):
            enhanced_terms.extend(['new', 'latest', 'recent', 'just launched', '2025', 'september'])
            # Add ALL curated phone names
            new_phones = [phone["name"] for phone in self.curated_data]
            enhanced_terms.extend(new_phones)
            logger.info("Detected NEW phones query - adding ALL curated phone names")
        
        if any(term in query_lower for term in ['top selling', 'best seller', 'most sold', 'most selling']):
            enhanced_terms.extend(['top selling', 'best seller', 'high sales', 'most sold', 'bestseller', 'most selling'])
            # Add ALL curated phone names
            top_phones = [phone["name"] for phone in self.curated_data]
            enhanced_terms.extend(top_phones)
            logger.info("Detected TOP SELLING phones query - adding ALL curated phone names")
        
        if any(term in query_lower for term in ['flip', 'foldable', 'fold', 'razr']):
            enhanced_terms.extend(['flip', 'foldable', 'fold', 'flip phone', 'clamshell', 'razr', 'z flip'])
            # Add ALL curated phone names
            flip_phones = [phone["name"] for phone in self.curated_data]
            enhanced_terms.extend(flip_phones)
            logger.info("Detected FLIP phones query - adding ALL curated phone names")
        
        # Your existing enhancement logic...
        # Detect use cases and add relevant terms
        for use_case, pattern in self.use_case_patterns.items():
            if re.search(pattern, query_lower):
                knowledge = self.domain_knowledge.get(use_case, {})
                enhanced_terms.extend(knowledge.get('brands', []))
                enhanced_terms.extend(knowledge.get('features', []))
        
        # Add price-specific terms for better BM25 matching
        price = self._extract_price_from_query(query)
        if price:
            # Add appropriate price range terms
            if price <= 15000:
                enhanced_terms.extend(["budget", "affordable", "under 15000", "cheap", "economical"])
            elif price <= 30000:
                enhanced_terms.extend(["mid-range", "under 30000", "mid range", "30k"])
            elif price <= 50000:
                enhanced_terms.extend(["premium", "under 50000", "50k", "mid premium"])
            elif price <= 70000:
                enhanced_terms.extend(["premium", "under 70000", "70k", "high end"])
            elif price <= 100000:
                enhanced_terms.extend(["flagship", "under 1 lakh", "1 lac", "premium flagship", "expensive"])
            elif price > 100000:
                enhanced_terms.extend(["ultra premium", "above 1 lakh", "luxury", "expensive"])
            
            # Add the actual price terms
            if price >= 100000:
                lakh_amount = price / 100000
                enhanced_terms.append(f"{lakh_amount} lakh")
                enhanced_terms.append(f"{lakh_amount} lac")
            else:
                enhanced_terms.append(f"{int(price/1000)}k")
            enhanced_terms.append(f"₹{int(price)}")
        
        # Add general domain terms for better semantic search
        if any(term in query_lower for term in ['phone', 'mobile', 'smartphone']):
            enhanced_terms.extend(['mobile', 'phone', 'smartphone', 'device'])
        
        # Add "best" and "recommend" terms for better matching
        if any(term in query_lower for term in ['best', 'top', 'good', 'recommend', 'suggest']):
            enhanced_terms.extend(['best', 'recommended', 'popular', 'top rated'])
        
        # Remove duplicates and create enhanced query
        if enhanced_terms:
            unique_terms = list(dict.fromkeys(enhanced_terms))
            enhanced_query = f"{query} {' '.join(unique_terms)}"
            if enhanced_query != original_query:
                logger.info(f"Query enhanced: '{original_query}' -> '{enhanced_query}'")
            return enhanced_query
        
        return query

    def _apply_filters(self, candidates: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Apply filters with IMPROVED price validation - FIXED for lakh amounts"""
        if not filters:
            return candidates
        
        filtered = []
        skipped_count = 0
        
        for candidate in candidates:
            try:
                metadata = candidate.metadata
                include = True
                skip_reasons = []
                
                # Price filter with MIN price enforcement for high amounts
                if 'max_price' in filters:
                    selling_price = self._extract_price(metadata.get('Selling Price'))
                    if selling_price > filters['max_price']:
                        include = False
                        skip_reasons.append(f"price ₹{selling_price} > ₹{filters['max_price']:,.0f}")
                
                # CRITICAL FIX: Enforce MIN price for high amount queries
                if 'min_price' in filters and include:
                    selling_price = self._extract_price(metadata.get('Selling Price'))
                    if selling_price < filters['min_price']:
                        include = False
                        skip_reasons.append(f"price ₹{selling_price} < min ₹{filters['min_price']:,.0f}")
                
                # Rating filter
                if 'min_rating' in filters and include:
                    rating = self._extract_rating(metadata.get('Rating'))
                    if rating < filters['min_rating']:
                        include = False
                        skip_reasons.append(f"rating {rating} < {filters['min_rating']}")
                
                # RAM filter
                if 'min_ram' in filters and include:
                    ram = self._extract_ram(metadata.get('Memory', ''))
                    if ram < filters['min_ram']:
                        include = False
                        skip_reasons.append(f"RAM {ram}GB < {filters['min_ram']}GB")
                
                # Storage filter
                if 'min_storage' in filters and include:
                    storage = self._extract_storage(metadata.get('Storage', ''))
                    if storage < filters['min_storage']:
                        include = False
                        skip_reasons.append(f"storage {storage}GB < {filters['min_storage']}GB")
                
                if include:
                    filtered.append(candidate)
                else:
                    skipped_count += 1
                    if skipped_count <= 5:  # Log only first few skips
                        logger.debug(f"Skipped {candidate.source}: {', '.join(skip_reasons)}")
                    
            except Exception as e:
                logger.warning(f"Filter application failed for {candidate.source}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Filtered {len(candidates)} -> {len(filtered)} results ({skipped_count} skipped)")
        return filtered

    def _extract_price(self, price_value) -> float:
        """Extract price from various formats - IMPROVED"""
        try:
            if pd.isna(price_value) or price_value is None:
                return float('inf')
            if isinstance(price_value, (int, float)):
                return float(price_value)
            
            price_str = str(price_value)
            
            # Handle string formats like "₹11,990", "11990", "N/A", etc.
            if price_str.upper() in ['N/A', 'NAN', '']:
                return float('inf')
            
            # Remove currency symbols and commas
            price_str = re.sub(r'[₹$,]', '', price_str).strip()
            
            # Try to convert to float
            if price_str and price_str.upper() != 'NAN':
                return float(price_str)
            else:
                return float('inf')
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Price extraction failed for '{price_value}': {e}")
            return float('inf')

    def _extract_rating(self, rating_value) -> float:
        """Extract rating from various formats"""
        try:
            if pd.isna(rating_value) or rating_value is None:
                return 0.0
            if isinstance(rating_value, (int, float)):
                return float(rating_value)
            rating_str = str(rating_value).strip()
            if rating_str.upper() in ['N/A', 'NAN', '']:
                return 0.0
            return float(rating_str) if rating_str and rating_str.upper() != 'NAN' else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_ram(self, memory_text: str) -> int:
        """Extract RAM size from memory text"""
        try:
            if pd.isna(memory_text) or memory_text is None:
                return 0
            memory_str = str(memory_text)
            ram_match = re.search(r'(\d+)\s*GB', memory_str, re.IGNORECASE)
            return int(ram_match.group(1)) if ram_match else 0
        except (ValueError, AttributeError):
            return 0

    def _extract_storage(self, storage_text: str) -> int:
        """Extract storage size from storage text"""
        try:
            if pd.isna(storage_text) or storage_text is None:
                return 0
            storage_str = str(storage_text)
            storage_match = re.search(r'(\d+)\s*GB', storage_str, re.IGNORECASE)
            return int(storage_match.group(1)) if storage_match else 0
        except (ValueError, AttributeError):
            return 0

    def semantic_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Enhanced semantic search with query enhancement"""
        try:
            logger.debug(f"Semantic search: '{query}' (top_k={top_k})")
            
            # Enhance query with domain knowledge
            enhanced_query = self._enhance_query_with_domain_knowledge(query)
            
            results = self.vector_store.similarity_search(enhanced_query, k=top_k)
            
            search_results = []
            for doc in results:
                try:
                    metadata = doc.metadata
                    if not self._is_valid_product(metadata):
                        continue
                    
                    text = self._format_product_text(metadata)
                    
                    search_results.append(SearchResult(
                        text=text,
                        source=metadata.get("Model", "Unknown"),
                        metadata=metadata,
                        score=0.0
                    ))
                except Exception as e:
                    logger.warning(f"Failed to process semantic result: {e}")
                    continue
            
            logger.debug(f"Semantic search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}\n{traceback.format_exc()}")
            return []

    def keyword_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Enhanced keyword search that includes CURATED DATA with BOOSTED scoring"""
        try:
            logger.debug(f"Keyword search: '{query}' (top_k={top_k})")
            
            # Enhance query for better BM25 matching
            enhanced_query = self._enhance_query_with_domain_knowledge(query)
            tokenized_query = enhanced_query.lower().split()
            
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k * 2]  # Get more results for filtering
            
            search_results = []
            curated_results = []
            regular_results = []
            
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                    
                try:
                    metadata_info = self.bm25_metadata[idx]
                    
                    if metadata_info["type"] == "main":
                        # Regular dataset result
                        row = self.docs_df.iloc[metadata_info["index"]]
                        metadata = row.to_dict()
                        
                        if not self._is_valid_product(metadata):
                            continue
                        
                        text = self._format_product_text(metadata)
                        source = metadata.get("Model", "Unknown")
                        
                        regular_results.append(SearchResult(
                            text=text,
                            source=source,
                            metadata=metadata,
                            score=float(scores[idx])  # Regular score
                        ))
                        
                    else:  # curated
                        # CURATED DATA result - APPLY SCORE BOOST
                        phone = metadata_info["phone"]
                        metadata = {
                            "Brand": phone["name"].split()[0],
                            "Model": phone["name"],
                            "Description": phone["description"],
                            "Specs": phone["specs"],
                            "Selling Price": phone["price"],
                            "Release Date": phone["release_date"],
                            "Source": "curated",  # Mark as curated
                            "Category": phone.get("category", ""),
                            "Tags": phone.get("tags", [])
                        }
                        
                        text = self._format_curated_product_text(phone)
                        source = phone["name"]
                        
                        # BOOST curated data scores for better ranking
                        boosted_score = float(scores[idx]) * 1.5  # 50% boost
                        
                        curated_results.append(SearchResult(
                            text=text,
                            source=source,
                            metadata=metadata,
                            score=boosted_score
                        ))
                    
                except Exception as e:
                    logger.warning(f"Failed to process BM25 result at index {idx}: {e}")
                    continue
            
            # Prioritize curated results by putting them first
            search_results = curated_results + regular_results
            
            # Take top_k overall
            search_results = search_results[:top_k]
            
            logger.debug(f"Keyword search returned {len(search_results)} results ({len(curated_results)} curated)")
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}\n{traceback.format_exc()}")
            return []

    def _format_curated_product_text(self, phone: Dict) -> str:
        """Format curated phone data for display"""
        return (
            f"Brand: {phone['name'].split()[0]}, "
            f"Model: {phone['name']}, "
            f"Description: {phone['description']}, "
            f"Specs: {phone['specs']}, "
            f"Price: ₹{phone['price']:,.0f}, "
            f"Release Date: {phone['release_date']}, "
            f"Source: {phone['source']}"
        )

    def _is_valid_product(self, metadata: Dict) -> bool:
        """Enhanced product validation - MORE LENIENT"""
        try:
            # Check essential fields
            if pd.isna(metadata.get('Model')) or not str(metadata.get('Model', '')).strip():
                return False
            
            # Check price validity - be more lenient
            selling_price = self._extract_price(metadata.get('Selling Price'))
            if selling_price <= 0 or selling_price == float('inf'):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Product validation failed: {e}")
            return False

    def _format_product_text(self, metadata: Dict) -> str:
        """Enhanced product text formatting"""
        try:
            brand = metadata.get('Brand', 'N/A')
            model = metadata.get('Model', 'N/A')
            color = metadata.get('Color', 'N/A')
            memory = metadata.get('Memory', 'N/A')
            storage = metadata.get('Storage', 'N/A')
            rating = self._extract_rating(metadata.get('Rating', 'N/A'))
            selling_price = self._extract_price(metadata.get('Selling Price', 'N/A'))
            original_price = self._extract_price(metadata.get('Original Price', 'N/A'))
            
            # Format with proper handling of missing values
            formatted = (
                f"Brand: {brand}, "
                f"Model: {model}, "
                f"Color: {color}, "
                f"Memory: {memory}, "
                f"Storage: {storage}, "
                f"Rating: {rating:.1f} stars, "
                f"Selling Price: ₹{selling_price:,.0f}"
            )
            
            if original_price != float('inf') and original_price > selling_price:
                formatted += f", Original Price: ₹{original_price:,.0f}"
            
            return formatted
            
        except Exception as e:
            logger.warning(f"Text formatting failed: {e}")
            return f"Model: {metadata.get('Model', 'Unknown')}"

    def _is_curated_pattern_query(self, query: str) -> bool:
        """
        Detect if query matches curated data patterns that should use BM25-only
        """
        query_lower = query.lower().strip()
        
        # Check for exact matches with common variations FIRST (most reliable)
        exact_patterns = [
            'popular phones',
            'popular phone', 
            'new phones',
            'new phone',
            'latest phones',
            'latest phone',
            'top selling phones',
            'best selling phones', 
            'most selling phones',
            'most sold phones',
            'high sales phones',
            'bestseller phones',
            'best seller phones',
            'flip phones',
            'foldable phones',
            # Add individual terms for broader matching
            'popular',
            'new',
            'latest',
            'top selling',
            'best selling',
            'most selling',
            'most sold',
            'bestseller',
            'best seller',
            'flip',
            'foldable'
        ]
        
        # Check exact patterns first (most reliable)
        for pattern in exact_patterns:
            if pattern in query_lower:
                logger.info(f"Exact curated pattern matched: '{pattern}' in '{query}'")
                return True
        
        # More aggressive pattern matching for curated queries
        curated_patterns = [
            # Popular phones patterns
            r'(popular|trending).*phone',
            r'phone.*(popular|trending)',
            r'which.*popular',
            r'what.*popular',
            r'show me.*popular',
            r'recommend.*popular',
            r'most.*popular',
            r'top.*popular',
            
            # New phones patterns  
            r'(new|latest|recent).*phone',
            r'phone.*(new|latest|recent)',
            r'any.*new',
            r'what.*new',
            r'latest.*release',
            r'recent.*release',
            r'just.*released',
            r'new.*launch',
            
            # Top selling patterns - FIXED: Remove word boundaries for multi-word phrases
            r'(top selling|best seller|high sales|most sold|bestseller|most selling|best selling|highest selling|top sold)',
            r'what.*(selling|seller)',
            r'which.*(selling|seller)',
            r'most.*(selling|sold)',
            r'highest.*selling',
            r'best.*seller',
            
            # Flip phones patterns
            r'(flip|foldable|fold).*phone',
            r'phone.*(flip|foldable)',
            r'show me.*flip',
            r'what.*flip',
            r'recommend.*flip',
            r'best.*flip'
        ]
        
        # Then check regex patterns
        for pattern in curated_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.info(f"Regex curated pattern matched: '{pattern}' for query: '{query}'")
                return True
        
        # Additional direct term check as final fallback
        selling_terms = ['selling', 'seller', 'sold', 'bestseller', 'sales']
        if any(term in query_lower for term in selling_terms):
            # Check if it's about phones (not other products)
            if any(phone_term in query_lower for phone_term in ['phone', 'mobile', 'smartphone']):
                logger.info(f"Direct selling term match in phone query: '{query}'")
                return True
        
        logger.info(f"No curated pattern matched for query: '{query}'")
        return False

    def _prioritize_curated_results(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Prioritize curated results for pattern queries to ensure they appear first
        """
        if not results:
            return []
        
        curated_results = []
        regular_results = []
        
        # Separate curated and regular results
        for result in results:
            if hasattr(result, 'metadata') and result.metadata.get('Source') == 'curated':
                curated_results.append(result)
            else:
                regular_results.append(result)
        
        logger.info(f"Prioritizing {len(curated_results)} curated results over {len(regular_results)} regular results")
        
        # Sort curated results by score (highest first)
        curated_results.sort(key=lambda x: x.score, reverse=True)
        
        # Combine: curated first, then regular
        final_results = curated_results + regular_results
        
        # Return top_k results
        return final_results[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Enhanced hybrid search with CURATED DATA priority for pattern queries
        """
        try:
            # Handle empty or invalid queries
            if not query or not isinstance(query, str) or not query.strip():
                logger.warning("Empty or invalid query received")
                return []

            # Limit query length to prevent abuse
            if len(query) > 500:
                query = query[:500]
                logger.info(f"Query truncated to 500 characters")

            logger.info(f"Hybrid search START: '{query}'")

            # Extract filters
            filters = self._extract_filters(query)
            
            # DETECT CURATED PATTERN QUERIES - CRITICAL FIX
            is_curated_pattern_query = self._is_curated_pattern_query(query)
            
            if is_curated_pattern_query:
                logger.info(f"CURATED PATTERN QUERY DETECTED: '{query}' - Using BM25-only approach")
                # For curated pattern queries, use ONLY BM25 to ensure curated data appears
                keyword_results = self.keyword_search(query, top_k * 3)  # Get even more BM25 results
                semantic_results = []  # Skip semantic search entirely
                
                # Apply filters
                filtered_results = self._apply_filters(keyword_results, filters)
                
                # Prioritize curated results in final output
                final_results = self._prioritize_curated_results(filtered_results, top_k)
                
                logger.info(f"BM25-only search returned {len(final_results)} results ({sum(1 for r in final_results if r.metadata.get('Source') == 'curated')} curated)")
                
            else:
                # Normal hybrid search for other queries
                logger.info(f"Regular query: '{query}' - Using hybrid approach")
                semantic_results = self.semantic_search(query, 25)
                keyword_results = self.keyword_search(query, 25)
                
                # Combine results
                combined_results = self._combine_results(semantic_results, keyword_results)
                
                if not combined_results:
                    logger.warning("No results found from either semantic or keyword search")
                    return []
                
                # Apply IMPROVED filters (with min price enforcement)
                filtered_results = self._apply_filters(combined_results, filters)
                
                # If no results with filtering, try without MIN price filter but keep MAX
                if not filtered_results and filters and 'min_price' in filters:
                    logger.info("No results with min price filter, trying without min price")
                    filters_without_min = {k: v for k, v in filters.items() if k != 'min_price'}
                    filtered_results = self._apply_filters(combined_results, filters_without_min)
                
                # Rerank if we have results
                final_results = self.rerank(query, filtered_results, top_k)

            logger.info(f"Hybrid search END: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}\n{traceback.format_exc()}")
            return []

    def _combine_results(self, semantic_results: List[SearchResult], 
                        keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Combine results with priority to semantic results"""
        seen_sources = set()
        combined = []
        
        # Add semantic results first (higher quality)
        for result in semantic_results:
            if result.source not in seen_sources:
                seen_sources.add(result.source)
                combined.append(result)
        
        # Add keyword results (wider coverage)
        for result in keyword_results:
            if result.source not in seen_sources:
                seen_sources.add(result.source)
                combined.append(result)
        
        return combined

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Enhanced reranking with fallback"""
        if not candidates:
            return []
        
        # Skip reranking for very specific filter queries
        filters = self._extract_filters(query)
        if len(filters) >= 2:  # Multiple specific filters
            logger.debug("Skipping rerank for heavily filtered query")
            return candidates[:top_k]
        
        if not self.reranker or len(candidates) <= 2:
            return candidates[:top_k]
        
        try:
            logger.debug(f"Reranking {len(candidates)} candidates")
            
            pairs = [[query, candidate.text] for candidate in candidates]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.reranker(**inputs, return_dict=True)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            
            # Combine scores
            reranked_candidates = []
            for candidate, rerank_score in zip(candidates, scores):
                combined_score = (
                    self.semantic_weight * rerank_score + 
                    self.bm25_weight * candidate.score
                )
                candidate.score = combined_score
                reranked_candidates.append(candidate)
            
            reranked_candidates.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"Reranking completed. Top score: {reranked_candidates[0].score:.3f}")
            return reranked_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed, returning original order: {e}")
            return candidates[:top_k]

    def debug_curated_search(self, query: str):
        """Debug method to see what's happening with curated search"""
        logger.info(f"=== DEBUG CURATED SEARCH: '{query}' ===")
        
        # Check pattern detection
        is_curated = self._is_curated_pattern_query(query)
        logger.info(f"Pattern detection: {is_curated}")
        
        # Test BM25 search
        bm25_results = self.keyword_search(query, 20)
        curated_count = sum(1 for r in bm25_results if r.metadata.get('Source') == 'curated')
        logger.info(f"BM25 results: {len(bm25_results)} total, {curated_count} curated")
        
        # Show top BM25 results
        for i, result in enumerate(bm25_results[:10]):
            source_type = "CURATED" if result.metadata.get('Source') == 'curated' else "REGULAR"
            logger.info(f"BM25 #{i+1}: {result.source} ({source_type}) - Score: {result.score:.4f}")
        
        # Test semantic search
        semantic_results = self.semantic_search(query, 10)
        logger.info(f"Semantic results: {len(semantic_results)}")
        
        # Test hybrid search
        hybrid_results = self.hybrid_search(query, 10)
        hybrid_curated = sum(1 for r in hybrid_results if r.metadata.get('Source') == 'curated')
        logger.info(f"Hybrid results: {len(hybrid_results)} total, {hybrid_curated} curated")
        
        # Show final results
        for i, result in enumerate(hybrid_results):
            source_type = "CURATED" if result.metadata.get('Source') == 'curated' else "REGULAR"
            logger.info(f"FINAL #{i+1}: {result.source} ({source_type}) - Score: {result.score:.4f}")
        
        logger.info("=== END DEBUG ===")