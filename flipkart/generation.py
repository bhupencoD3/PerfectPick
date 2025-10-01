from collections import deque
import re
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)

@dataclass
class GenerationResponse:
    query: str
    answer: str
    sources: List[str]
    session_id: Optional[str]
    debug_info: Dict[str, Any]
    resolved_query: str

@dataclass
class SearchResult:
    text: str
    source: str
    metadata: Dict[str, Any]
    score: float = 0.0

class EnhancedSessionMemoryManager:
    """Enhanced session memory management with conversation tracking"""
    
    def __init__(self, memory_db, memory_size: int = 10):
        self.memory_db = memory_db
        self.memory_size = memory_size
        
    def load_memory(self, session_id: str) -> deque:
        """Load session memory with comprehensive error handling"""
        try:
            if self.memory_db:
                memory = self.memory_db.load_memory(session_id)
                logger.info(f"Loaded {len(memory)} memory entries for session {session_id}")
                return memory
            return deque(maxlen=self.memory_size)
        except Exception as e:
            logger.error(f"Memory load failed for {session_id}: {e}")
            return deque(maxlen=self.memory_size)
    
    def save_memory(self, session_id: str, memory: deque) -> bool:
        """Save session memory with validation"""
        try:
            if self.memory_db and memory:
                # Validate memory size
                if len(memory) > self.memory_size * 2:  # Prevent memory bloat
                    memory = deque(list(memory)[-self.memory_size:], maxlen=self.memory_size)
                
                self.memory_db.save_memory(session_id, memory)
                logger.debug(f"Saved {len(memory)} entries for session {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Memory save failed for {session_id}: {e}")
            return False

class EnhancedQueryResolver:
    """Enhanced query resolution with multi-question handling"""
    
    def __init__(self, valid_models: List[str]):
        self.valid_models = valid_models
        self.pronouns = ["it", "its", "they", "their", "them", "this", "that", "the phone"]
        self.follow_up_keywords = [
            'spec', 'specs', 'specification', 'ram', 'storage', 'memory',
            'color', 'price', 'cost', 'rating', 'battery', 'camera',
            'which', 'what', 'how much', 'details', 'about', 'tell me'
        ]
        self.question_patterns = [
            r'([^.!?]*\?)',  # Questions ending with ?
            r'(what|which|how|when|where|why)\s+[^.!?]*[.!?]'  # Question words
        ]
    
    def resolve(self, query: str, session_memory: List[str]) -> Tuple[str, Optional[str]]:
        """
        Enhanced query resolution with multi-question handling
        """
        try:
            # Handle empty query
            if not query or not query.strip():
                return query, None
            
            # Handle very long queries
            if len(query) > 300:
                query = self._summarize_long_query(query)
            
            # Extract primary question from multi-question queries
            primary_question = self._extract_primary_question(query)
            
            logger.info(f"Resolving query: '{primary_question}' with {len(session_memory)} memory entries")
            
            # Extract current model from query
            current_model = self._extract_model_from_text(primary_question)
            
            # Extract last model from memory
            last_model = self._extract_last_model_from_memory(session_memory)
            
            # Determine if this is a follow-up query
            is_follow_up = self._is_follow_up_query(primary_question)
            
            resolved_query = primary_question
            resolved_model = current_model or last_model
            
            # Enhanced pronoun resolution
            if is_follow_up and last_model and self._has_pronouns(primary_question) and not current_model:
                resolved_query = self._replace_pronouns(primary_question, last_model)
                logger.info(f"Pronoun resolution: '{primary_question}' -> '{resolved_query}'")
            
            # Enhanced context addition
            elif is_follow_up and last_model and not current_model:
                if not any(last_model.lower() in resolved_query.lower() for last_model in [last_model]):
                    resolved_query = f"{resolved_query} {last_model}"
                    logger.info(f"Context addition: '{primary_question}' -> '{resolved_query}'")
            
            logger.info(f"Query resolved: model='{resolved_model}', resolved='{resolved_query}'")
            return resolved_query, resolved_model
            
        except Exception as e:
            logger.error(f"Query resolution failed: {e}")
            return query, None
    
    def _summarize_long_query(self, query: str) -> str:
        """Summarize very long queries to key components"""
        logger.info(f"Summarizing long query ({len(query)} chars)")
        
        # Extract key components
        key_components = []
        
        # Extract models
        models_found = []
        for model in self.valid_models:
            if model.lower() in query.lower():
                models_found.append(model)
        
        if models_found:
            key_components.extend(models_found[:2])  # Limit to 2 models
        
        # Extract key terms
        key_terms = ['price', 'ram', 'storage', 'camera', 'battery', 'rating', 'gaming', 'performance']
        for term in key_terms:
            if term in query.lower():
                key_components.append(term)
        
        summarized = " ".join(key_components) if key_components else query[:200]
        logger.info(f"Long query summarized to: '{summarized}'")
        return summarized
    
    def _extract_primary_question(self, query: str) -> str:
        """Extract primary question from multi-question queries"""
        questions = []
        
        # Find all questions
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                questions.append(match.group(1).strip())
        
        if questions:
            # Return the first question or the one with most context
            primary = max(questions, key=len)
            if len(questions) > 1:
                logger.info(f"Multiple questions detected, using primary: '{primary}'")
            return primary
        
        # If no clear questions, return the query as is
        return query
    
    def _extract_model_from_text(self, text: str) -> Optional[str]:
        """Enhanced model extraction with better matching"""
        if not text or not text.strip():
            return None
            
        text_lower = text.lower()
        
        # Sort by length to match longer names first
        sorted_models = sorted(self.valid_models, key=len, reverse=True)
        
        for model in sorted_models:
            model_lower = model.lower()
            # Use word boundaries for exact matching
            if re.search(rf'\b{re.escape(model_lower)}\b', text_lower):
                return model
        
        return None
    
    def _extract_last_model_from_memory(self, session_memory: List[str]) -> Optional[str]:
        """Enhanced memory model extraction"""
        if not session_memory:
            return None
        
        # Check recent memory with priority
        recent_memory = session_memory[-3:]  # Last 3 interactions for better context
        
        for memory_entry in reversed(recent_memory):
            if " | " in memory_entry:
                try:
                    question, answer = memory_entry.split(" | ", 1)
                    # Check answer first (more likely to contain model)
                    model = self._extract_model_from_text(answer)
                    if model:
                        return model
                    # Then check question
                    model = self._extract_model_from_text(question)
                    if model:
                        return model
                except Exception as e:
                    logger.warning(f"Memory parsing failed: {e}")
                    continue
        
        return None
    
    def _is_follow_up_query(self, query: str) -> bool:
        """Enhanced follow-up detection"""
        if not query or len(query.strip()) < 3:
            return False
            
        query_lower = query.lower()
        
        # Check for follow-up keywords
        has_keywords = any(keyword in query_lower for keyword in self.follow_up_keywords)
        
        # Check for short/context-dependent queries
        is_short = len(query_lower.split()) <= 4
        
        # Check for pronoun usage
        has_pronouns = self._has_pronouns(query)
        
        return has_keywords or is_short or has_pronouns
    
    def _has_pronouns(self, query: str) -> bool:
        """Check if query contains pronouns"""
        if not query:
            return False
        query_lower = query.lower()
        return any(pronoun in query_lower for pronoun in self.pronouns)
    
    def _replace_pronouns(self, query: str, model: str) -> str:
        """Enhanced pronoun replacement"""
        resolved = query
        for pronoun in self.pronouns:
            pattern = rf'\b{re.escape(pronoun)}\b'
            if re.search(pattern, resolved, re.IGNORECASE):
                resolved = re.sub(pattern, model, resolved, flags=re.IGNORECASE)
                # Only replace the first occurrence
                break
        return resolved

class ProductionRAGGenerator:
    """
    Enhanced production-grade RAG generator with IMPROVED formatting
    """
    
    def __init__(self, retriever, llm, memory_db=None, memory_size: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.memory_manager = EnhancedSessionMemoryManager(memory_db, memory_size)
        self.valid_models = retriever.docs_df['Model'].unique().tolist()
        self.query_resolver = EnhancedQueryResolver(self.valid_models)
        
        # Initialize enhanced prompt templates
        self._init_prompt_templates()
        
        logger.info(f"ProductionRAGGenerator initialized with {len(self.valid_models)} models")
    
    # In the _init_prompt_templates method, update the system prompt:
    def _init_prompt_templates(self):
        """Initialize enhanced system prompts with CARD formatting"""
        self.system_prompt = """You are an expert Flipkart mobile assistant. Provide responses in CARD format.

    CRITICAL CARD FORMATTING RULES:
    1. For EACH phone, use this EXACT card format:
    ┌─ PHONE CARD ──────────────────────┐
    │ Model: Full Model Name            │
    │ Price: ₹X,XXX                     │
    │ RAM: X GB                         │
    │ Storage: X GB                     │
    │ Display: X inches                 │
    │ Camera: X MP + X MP               │
    │ Battery: X mAh                    │
    │ Rating: ⭐ X.X/5                  │
    │ Brand: Brand Name                 │
    └───────────────────────────────────┘

    2. For multiple phones, separate each with a blank line
    3. Always include ALL specifications when available
    4. Use consistent formatting across all cards
    5. If specification is unknown, use "Not specified"
    6. Keep price formatting consistent with ₹ symbol
    7. Use star ratings with ⭐ symbol

    BAD EXAMPLE:
    Samsung Galaxy S23 has 8GB RAM and 256GB storage

    GOOD EXAMPLE:
    ┌─ PHONE CARD ──────────────────────┐
    │ Model: Samsung Galaxy S23         │
    │ Price: ₹74,999                    │
    │ RAM: 8 GB                         │
    │ Storage: 256 GB                   │
    │ Display: 6.1 inches               │
    │ Camera: 50 MP + 12 MP + 10 MP     │
    │ Battery: 3900 mAh                 │
    │ Rating: ⭐ 4.5/5                  │
    │ Brand: Samsung                    │
    └───────────────────────────────────┘

    ┌─ PHONE CARD ──────────────────────┐
    │ Model: OnePlus 11R                │
    │ Price: ₹39,999                    │
    │ RAM: 8 GB                         │
    │ Storage: 128 GB                   │
    │ Display: 6.74 inches              │
    │ Camera: 50 MP + 8 MP + 2 MP       │
    │ Battery: 5000 mAh                 │
    │ Rating: ⭐ 4.3/5                  │
    │ Brand: OnePlus                    │
    └───────────────────────────────────┘

    CONTEXT DATA:
    {context}

    CONVERSATION HISTORY:
    {history}

    USER QUERY: {query}

    Provide responses in clean card format with complete specifications:"""
        
    def _build_messages(self, query: str, contexts: List[SearchResult], 
                       session_memory: List[str]) -> List[BaseMessage]:
        """Build enhanced conversation messages"""
        try:
            # Validate inputs
            if not query or not query.strip():
                return [HumanMessage(content="Please provide a valid question about mobile phones.")]
            
            # Format context with better organization
            context_text = self._format_contexts(contexts)
            
            # Format conversation history
            history_text = self._format_history(session_memory)
            
            # Build enhanced system message
            system_content = self.system_prompt.format(
                context=context_text,
                history=history_text,
                query=query
            )
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            logger.debug(f"Built prompt with {len(contexts)} contexts and {len(session_memory)} history entries")
            return messages
            
        except Exception as e:
            logger.error(f"Message building failed: {e}")
            return [HumanMessage(content=query)]
    
    def _format_contexts(self, contexts: List[SearchResult]) -> str:
        """Enhanced context formatting with clean organization"""
        if not contexts:
            return "No product information available in the current database."
        
        context_lines = ["AVAILABLE PRODUCTS:"]
        
        for i, context in enumerate(contexts, 1):
            try:
                context_lines.append(f"\n{i}. {context.text}")
            except Exception as e:
                logger.warning(f"Failed to format context {i}: {e}")
                continue
        
        return "\n".join(context_lines)
    
    def _format_history(self, session_memory: List[str]) -> str:
        """Enhanced history formatting without emojis"""
        if not session_memory:
            return "No previous conversation in this session."
        
        history_lines = ["PREVIOUS CONVERSATION:"]
        added_count = 0
        
        for memory in session_memory[-3:]:  # Last 3 exchanges for relevance
            if " | " in memory:
                try:
                    question, answer = memory.split(" | ", 1)
                    # Truncate long responses for context
                    if len(answer) > 200:
                        answer = answer[:200] + "..."
                    history_lines.append(f"User: {question}")
                    history_lines.append(f"Assistant: {answer}\n")
                    added_count += 1
                except Exception as e:
                    logger.warning(f"Memory formatting failed: {e}")
                    continue
        
        return "\n".join(history_lines) if added_count > 0 else "No relevant previous conversation."
    
    def _extract_sources(self, contexts: List[SearchResult]) -> List[str]:
        """Extract sources with validation"""
        sources = []
        for context in contexts:
            try:
                if (context.source and 
                    context.source not in sources and 
                    context.source != 'Unknown' and
                    len(sources) < 5):  # Limit to top 5
                    sources.append(context.source)
            except Exception as e:
                logger.warning(f"Source extraction failed: {e}")
                continue
        return sources
    
    def _validate_and_format_response(self, response: str) -> str:
        """Enhanced response validation with clean formatting"""
        if not response or not response.strip():
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        try:
            cleaned = response.strip()
            
            # Fix common formatting issues
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            cleaned = re.sub(r'\.{3,}', '...', cleaned)  # Fix multiple dots
            cleaned = re.sub(r'\!{2,}', '!', cleaned)  # Fix multiple exclamations
            
            # Remove ALL markdown formatting
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove **bold**
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove *italic*
            cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)        # Remove _italic_
            cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)        # Remove `code`
            
            # Remove emojis and replace with clean text
            emoji_replacements = {
                "📱": "Phone",
                "💰": "Price:",
                "⭐": "Rating:",
                "🏷️": "Brand:",
                "🔍": "Search",
                "💡": "Suggestion",
                "🤔": "",
                "⚠️": "",
                "😔": "",
                "🎯": "",
                "✅": "",
                "🔥": "",
                "💯": "",
                "👍": ""
            }
            
            for emoji, replacement in emoji_replacements.items():
                cleaned = cleaned.replace(emoji, replacement)
            
            # Replace placeholder values with better text
            replacements = {
                "Memory: nan": "Memory: Not specified",
                "Rating: nan": "Rating: Not rated yet",
                "Price: ₹nan": "Price: Currently unavailable",
                "Color: nan": "Color: Various options available",
                "Storage: nan": "Storage: Not specified",
                "Brand: nan": "Brand: Not specified"
            }
            
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
            
            # Ensure response ends with proper punctuation
            if not cleaned[-1] in ['.', '!', '?']:
                cleaned += '.'
            
            # Add proper spacing for readability
            cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return "Here are the phone details based on your query."
    
    def generate_answer(self, query: str, session_id: Optional[str] = None) -> GenerationResponse:
        """
        Enhanced answer generation with comprehensive error handling
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Validate input query
            if not query or not query.strip():
                return self._create_empty_query_response(session_id)
            
            # Handle very long queries
            if len(query) > 1000:
                query = query[:500] + "..."
                logger.info(f"Query truncated due to excessive length")
            
            logger.info(f"Generation START: '{query}' (session: {session_id})")
            
            # Load session memory
            session_memory = self.memory_manager.load_memory(session_id)
            
            # Resolve query with context
            resolved_query, context_model = self.query_resolver.resolve(query, list(session_memory))
            
            # Retrieve relevant contexts - INCREASED top_k for better coverage
            contexts = self.retriever.hybrid_search(resolved_query, top_k=15)  # Increased from 10
            
            # Generate answer
            if not contexts:
                answer = self._handle_no_results(resolved_query, context_model)
                sources = []
            else:
                messages = self._build_messages(resolved_query, contexts, list(session_memory))
                llm_response = self.llm.invoke(messages)
                answer = self._extract_llm_content(llm_response)
                answer = self._validate_and_format_response(answer)
                sources = self._extract_sources(contexts)
            
            # Update session memory
            if session_id and answer:
                self._update_memory(session_id, query, answer, session_memory)
            
            # Prepare enhanced debug info
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            debug_info = {
                "resolved_query": resolved_query,
                "context_model": context_model,
                "contexts_count": len(contexts),
                "sources_count": len(sources),
                "memory_entries": len(session_memory),
                "processing_time_seconds": round(processing_time, 2)
            }
            
            logger.info(f"Generation END: {len(answer)} chars, {len(sources)} sources, {processing_time:.2f}s")
            
            return GenerationResponse(
                query=query,
                answer=answer,
                sources=sources,
                session_id=session_id,
                debug_info=debug_info,
                resolved_query=resolved_query
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            return self._create_error_response(query, session_id, str(e))
    
    def _create_empty_query_response(self, session_id: Optional[str]) -> GenerationResponse:
        """Handle empty query case with clean formatting"""
        return GenerationResponse(
            query="",
            answer="Please provide a specific question about mobile phones.\n\nExamples:\n- Phones under ₹20,000\n- Best camera phone\n- OPPO A53 specifications\n- Gaming phones with 8GB RAM",
            sources=[],
            session_id=session_id,
            debug_info={"error": "Empty query"},
            resolved_query=""
        )
    
    def _handle_no_results(self, query: str, context_model: Optional[str]) -> str:
        """Enhanced no-results handling with clean formatting"""
        base_msg = "I couldn't find exact matches in our current database."
        
        suggestions = []
        
        if context_model:
            suggestions.append(f"- '{context_model}' might be out of stock or discontinued")
            suggestions.append("- Check Flipkart's official website for latest availability")
        else:
            suggestions.append("- Try different search terms (e.g., 'budget phones' instead of 'cheap phones')")
            suggestions.append("- Be more specific with your requirements")
            suggestions.append("- Check if you've spelled the model name correctly")
            suggestions.append("- Try a different price range")
        
        suggestion_text = "\n".join(suggestions)
        
        return f"{base_msg}\n\nSuggestions:\n{suggestion_text}\n\nYou can also visit Flipkart.com for the most up-to-date information."
    
    def _extract_llm_content(self, llm_response) -> str:
        """Enhanced LLM content extraction"""
        try:
            if hasattr(llm_response, 'content'):
                content = str(llm_response.content)
            elif isinstance(llm_response, dict):
                content = str(llm_response.get('text', llm_response.get('content', 'No response generated')))
            else:
                content = str(llm_response)
            
            # Basic validation
            if not content or content.strip() == "" or content == "No response generated":
                return "I received an empty response. Please try asking your question differently."
            
            return content
            
        except Exception as e:
            logger.error(f"LLM response extraction failed: {e}")
            return "I encountered an issue while processing your request. Please try again."
    
    def _update_memory(self, session_id: str, query: str, answer: str, session_memory: deque):
        """Enhanced memory update with validation"""
        try:
            # Validate inputs
            if not query or not answer:
                return
                
            # Truncate very long answers for memory
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            memory_entry = f"Q: {query} | A: {answer}"
            session_memory.append(memory_entry)
            
            # Save memory
            success = self.memory_manager.save_memory(session_id, session_memory)
            if success:
                logger.debug(f"Updated memory for session {session_id}")
            else:
                logger.warning(f"Memory update failed for session {session_id}")
                
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
    
    def _create_error_response(self, query: str, session_id: Optional[str], error: str) -> GenerationResponse:
        """Enhanced error response"""
        return GenerationResponse(
            query=query,
            answer="I'm experiencing technical difficulties at the moment. Please try again in a few moments or rephrase your question.",
            sources=[],
            session_id=session_id,
            debug_info={"error": error, "type": "generation_error"},
            resolved_query=query
        )
    
    def clear_memory(self, session_id: str) -> bool:
        """Enhanced memory clearing"""
        try:
            if self.memory_manager.memory_db:
                empty_memory = deque(maxlen=self.memory_manager.memory_size)
                success = self.memory_manager.save_memory(session_id, empty_memory)
                if success:
                    logger.info(f"Cleared memory for session {session_id}")
                return success
            return False
        except Exception as e:
            logger.error(f"Memory clear failed: {e}")
            return False