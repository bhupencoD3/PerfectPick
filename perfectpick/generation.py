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
    output_format: str

@dataclass
class SearchResult:
    text: str
    source: str
    metadata: Dict[str, Any]
    score: float = 0.0

class EnhancedSessionMemoryManager:
    def __init__(self, memory_db, memory_size: int = 10):
        self.memory_db = memory_db
        self.memory_size = memory_size
        
    def load_memory(self, session_id: str) -> deque:
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
        try:
            if self.memory_db and memory:
                if len(memory) > self.memory_size * 2:
                    memory = deque(list(memory)[-self.memory_size:], maxlen=self.memory_size)
                
                success = self.memory_db.save_memory(session_id, memory)
                if success:
                    logger.debug(f"Saved {len(memory)} entries for session {session_id}")
                else:
                    logger.warning(f"Memory save returned False for session {session_id}")
                return success
            return False
        except Exception as e:
            logger.error(f"Memory save failed for {session_id}: {e}")
            return False

class EnhancedQueryResolver:
    def __init__(self, valid_models: List[str]):
        self.valid_models = valid_models
        self.pronouns = ["it", "its", "they", "their", "them", "this", "that", "the phone"]
        self.follow_up_keywords = [
            'spec', 'specs', 'specification', 'ram', 'storage', 'memory',
            'color', 'price', 'cost', 'rating', 'battery', 'camera',
            'which', 'what', 'how much', 'details', 'about', 'tell me'
        ]
        self.question_patterns = [
            r'([^.!?]*\?)',
            r'(what|which|how|when|where|why)\s+[^.!?]*[.!?]'
        ]
    
    def resolve(self, query: str, session_memory: List[str]) -> Tuple[str, Optional[str]]:
        try:
            if not query or not query.strip():
                return query, None
            
            if len(query) > 300:
                query = self._summarize_long_query(query)
            
            primary_question = self._extract_primary_question(query)
            
            logger.info(f"Resolving query: '{primary_question}' with {len(session_memory)} memory entries")
            
            current_model = self._extract_model_from_text(primary_question)
            last_model = self._extract_last_model_from_memory(session_memory)
            is_follow_up = self._is_follow_up_query(primary_question)
            
            resolved_query = primary_question
            resolved_model = current_model or last_model
            
            if is_follow_up and last_model and self._has_pronouns(primary_question) and not current_model:
                resolved_query = self._replace_pronouns(primary_question, last_model)
                logger.info(f"Pronoun resolution: '{primary_question}' -> '{resolved_query}'")
            
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
        logger.info(f"Summarizing long query ({len(query)} chars)")
        
        key_components = []
        
        models_found = []
        for model in self.valid_models:
            if model.lower() in query.lower():
                models_found.append(model)
        
        if models_found:
            key_components.extend(models_found[:2])
        
        key_terms = ['price', 'ram', 'storage', 'camera', 'battery', 'rating', 'gaming', 'performance']
        for term in key_terms:
            if term in query.lower():
                key_components.append(term)
        
        summarized = " ".join(key_components) if key_components else query[:200]
        logger.info(f"Long query summarized to: '{summarized}'")
        return summarized
    
    def _extract_primary_question(self, query: str) -> str:
        questions = []
        
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                questions.append(match.group(1).strip())
        
        if questions:
            primary = max(questions, key=len)
            if len(questions) > 1:
                logger.info(f"Multiple questions detected, using primary: '{primary}'")
            return primary
        
        return query
    
    def _extract_model_from_text(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
            
        text_lower = text.lower()
        sorted_models = sorted(self.valid_models, key=len, reverse=True)
        
        for model in sorted_models:
            model_lower = model.lower()
            if re.search(rf'\b{re.escape(model_lower)}\b', text_lower):
                return model
        
        return None
    
    def _extract_last_model_from_memory(self, session_memory: List[str]) -> Optional[str]:
        if not session_memory:
            return None
        
        recent_memory = session_memory[-3:]
        
        for memory_entry in reversed(recent_memory):
            if " | " in memory_entry:
                try:
                    question, answer = memory_entry.split(" | ", 1)
                    model = self._extract_model_from_text(answer)
                    if model:
                        return model
                    model = self._extract_model_from_text(question)
                    if model:
                        return model
                except Exception as e:
                    logger.warning(f"Memory parsing failed: {e}")
                    continue
        
        return None
    
    def _is_follow_up_query(self, query: str) -> bool:
        if not query or len(query.strip()) < 3:
            return False
            
        query_lower = query.lower()
        has_keywords = any(keyword in query_lower for keyword in self.follow_up_keywords)
        is_short = len(query_lower.split()) <= 4
        has_pronouns = self._has_pronouns(query)
        
        return has_keywords or is_short or has_pronouns
    
    def _has_pronouns(self, query: str) -> bool:
        if not query:
            return False
        query_lower = query.lower()
        return any(pronoun in query_lower for pronoun in self.pronouns)
    
    def _replace_pronouns(self, query: str, model: str) -> str:
        resolved = query
        for pronoun in self.pronouns:
            pattern = rf'\b{re.escape(pronoun)}\b'
            if re.search(pattern, resolved, re.IGNORECASE):
                resolved = re.sub(pattern, model, resolved, flags=re.IGNORECASE)
                break
        return resolved

class DualModeRAGGenerator:
    def __init__(self, retriever, llm, memory_db=None, memory_size: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.memory_manager = EnhancedSessionMemoryManager(memory_db, memory_size)
        self.valid_models = retriever.docs_df['Model'].unique().tolist()
        self.query_resolver = EnhancedQueryResolver(self.valid_models)
        
        self._init_dual_prompt_templates()
        logger.info(f"DualModeRAGGenerator initialized with {len(self.valid_models)} models")
    
    def _init_dual_prompt_templates(self):
        self.card_triggers = {
            'specification_keywords': [
                'spec', 'specs', 'specification', 'specifications', 'features', 'feature',
                'details', 'detail', 'information', 'info', 'data', 'technical',
                'configuration', 'config', 'hardware', 'tech', 'technical specs',
            ],
            'component_keywords': [
                'ram', 'memory', 'storage', 'battery', 'camera', 'display', 'screen',
                'processor', 'chipset', 'cpu', 'gpu', 'os', 'operating system',
                'android', 'ios', 'weight', 'dimension', 'size', 'resolution',
                'megapixel', 'mp', 'mAh', 'mah', 'inch', 'inches', 'gb'
            ],
            'product_keywords': [
                'model', 'phone', 'mobile', 'device', 'handset', 'smartphone',
                'show me', 'list', 'what is', 'what are', 'give me', 'get me',
                'find me', 'search for', 'look for', 'price of', 'cost of',
                'rate', 'rating', 'brand', 'color', 'variant'
            ],
            'commerce_keywords': [
                'price', 'cost', '₹', 'rs.', 'rupee', 'buy', 'purchase',
                'available', 'availability', 'stock', 'in stock', 'out of stock',
                'discount', 'offer', 'deal', 'sale'
            ]
        }
        
        self.explanatory_triggers = {
            'advice_keywords': [
                'advice', 'advise', 'suggest', 'suggestion', 'recommend', 'recommendation',
                'should i', 'shall i', 'can i', 'could i', 'would you',
                'help me', 'guide me', 'assist me', 'what should', 'what shall',
                'need advice', 'need help', 'need suggestion', 'looking for advice','worth it'
            ],
            'comparison_keywords': [
                'compare', 'comparison', 'versus', 'vs', 'vs.', 'difference', 'differences',
                'different', 'better', 'best', 'good', 'great', 'excellent', 'superior',
                'worse', 'worst', 'bad', 'poor', 'inferior', 'advantage', 'disadvantage',
                'pros', 'cons', 'pros and cons', 'advantages', 'disadvantages',
                'which one', 'which is better', 'which should i choose',
                'this or that', 'x or y', 'a vs b'
            ],
            'decision_keywords': [
                'choose', 'choice', 'select', 'pick', 'decide', 'decision',
                'option', 'options', 'alternative', 'alternatives',
                'what to buy', 'which to buy', 'what to choose',
                'go for', 'opt for', 'settle for'
            ],
            'explanation_keywords': [
                'why', 'how', 'what does', 'what is the meaning of',
                'explain', 'explanation', 'understand', 'meaning',
                'benefit', 'benefits', 'importance', 'significant', 'significance',
                'purpose', 'use', 'usage', 'function', 'working', 'works',
                'how does', 'how to', 'what makes', 'what causes'
            ],
            'usecase_keywords': [
                'for gaming', 'for photography', 'for camera', 'for video',
                'for work', 'for business', 'for office', 'for student',
                'for college', 'for school', 'for senior', 'for elder',
                'for beginner', 'for professional', 'for heavy use',
                'gaming phone', 'camera phone', 'budget phone', 'premium phone',
                'performance', 'battery life', 'camera quality', 'display quality'
            ]
        }

        self.card_system_prompt = """You are an expert Flipkart mobile assistant. Provide responses in CARD format.

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

8. PROVIDE COMPLETE CARDS: Always include all standard fields (Model, Price, RAM, Storage, Display, Camera, Battery, Rating, Brand)
9. USE REASONABLE ESTIMATES: If exact data is not available, provide typical/common values for the model
10. ENSURE COMPLETENESS: Every phone card must have all fields filled with appropriate values

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

        self.explanatory_system_prompt = self.explanatory_system_prompt = """You are an expert mobile phone consultant. You MUST follow the EXACT output format below — no exceptions.

# SUMMARY
Write 1–2 clear, direct sentences answering the user's main question.

# DETAILS
Structure your explanation EXACTLY as follows:
- Start every major section with "## " followed by a short title (e.g., "## Camera Quality", "## Battery Life")
- NEVER use numbers like "1", "2", "3" at the start of lines
- NEVER merge the section title with content (e.g., "Comparison The phone..." is WRONG)
- After each "## Title", write content on NEW lines
- Use "- " ONLY to start bullet points (not as separators like "Model: X - RAM: Y")
- Keep paragraphs short (1–3 sentences)
- NEVER use tables, markdown code blocks, or HTML

# RECOMMENDATION
Give a FINAL, CONCISE verdict in 2–3 sentences MAX. Make it:
- Actionable ("Choose X if...", "Avoid Y unless...")
- Opinionated and clear
- Free of new specs or data
- Start directly with your recommendation (no "In conclusion:")

✅ GOOD EXAMPLE:
## Display and Performance
The S25 Ultra features a 6.8-inch Dynamic AMOLED display with 120Hz refresh rate.
- Smooth scrolling and vibrant colors
- Powered by Snapdragon 8 Gen 3 for flagship performance

❌ BAD EXAMPLES:
1. Comparison The phone has...          ← NO numbers, no merged title+text
Model: S25 - RAM: 12GB - Storage: 256GB ← NO inline dashes as separators
## Price and Availability: ₹129,999     ← NO colon after heading

⚠️ CRITICAL: Never use numbers like "1.", "2.", or "3." to start sections. Always use ## Title on its own line. Never merge the title with content on the same line.

CONTEXT DATA:
{context}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

Now generate your response using ONLY the structure above. Double-check:
- # SUMMARY exists
- # DETAILS uses ONLY "##" headings and "- " bullets
- # RECOMMENDATION is short and punchy
"""
    
    def _determine_response_mode(self, query: str, contexts: List[SearchResult]) -> str:
        query_lower = query.lower().strip()
        
        # Check for EXPLICIT explanatory triggers first (strong signals)
        strong_explanatory_phrases = [
            'worth it', 'should i buy', 'is it good', 'pros and cons',
            'which is better', 'help me choose', 'what should i', 'recommend me',
            'advice', 'suggest', 'recommend'
        ]
        
        # If query contains strong explanatory phrases, prioritize explanatory mode
        for phrase in strong_explanatory_phrases:
            if phrase in query_lower:
                logger.info(f"Strong explanatory phrase detected: '{phrase}'")
                return "explanation"
        
        all_card_triggers = []
        for category in self.card_triggers.values():
            all_card_triggers.extend(category)
        
        all_explanatory_triggers = []
        for category in self.explanatory_triggers.values():
            all_explanatory_triggers.extend(category)
        
        card_score = 0
        explanatory_score = 0
        
        # Check for model mentions (strong card indicator)
        has_model_mention = any(
            re.search(rf'\b{re.escape(model.lower())}\b', query_lower)
            for model in self.valid_models
        )
        
        # Score card triggers
        for trigger in all_card_triggers:
            if re.search(rf'\b{re.escape(trigger)}\b', query_lower):
                card_score += 2
            elif trigger in query_lower:
                card_score += 1
        
        # Score explanatory triggers
        for trigger in all_explanatory_triggers:
            if re.search(rf'\b{re.escape(trigger)}\b', query_lower):
                explanatory_score += 2
            elif trigger in query_lower:
                explanatory_score += 1
        
        # Special case patterns with adjusted weights
        special_cases = {
            'card_boosters': [
                r'\b\d+\s*gb\s*ram\b', r'\b\d+\s*gb\s*storage\b', r'₹\s*\d+',
                r'price\s*of', r'specs\s*of', r'details\s*of', r'specifications\s*of'
            ],
            'explanatory_boosters': [
                r'worth it', r'should.*buy', r'is.*good', r'pros.*cons',
                r'which.*better', r'what.*recommend', r'help.*choose',
                r'compare.*and', r'difference.*between', r'advice.*on'
            ]
        }
        
        # Apply special case boosts
        for pattern in special_cases['card_boosters']:
            if re.search(pattern, query_lower):
                card_score += 3
        
        for pattern in special_cases['explanatory_boosters']:
            if re.search(pattern, query_lower):
                explanatory_score += 4  # Higher weight for explanatory boosters
        
        # Model mention strongly suggests card mode
        if has_model_mention:
            card_score += 5
        
        # Context-based decision making
        if contexts:
            context_text = ' '.join([ctx.text.lower() for ctx in contexts])
            # If context suggests comparison or multiple models, lean explanatory
            if any(word in context_text for word in ['compare', 'versus', 'vs', 'difference']):
                explanatory_score += 2
        
        # Query length and structure analysis
        words = query_lower.split()
        if len(words) <= 3 and not has_model_mention:
            # Short vague queries like "worth it" should be explanatory
            explanatory_score += 3
        
        # Check for question words that suggest explanatory mode
        question_words = ['should', 'which', 'what', 'how', 'why', 'worth', 'good', 'better']
        if any(word in query_lower for word in question_words) and not has_model_mention:
            explanatory_score += 2
        
        logger.info(f"Mode detection - Query: '{query_lower}'")
        logger.info(f"Mode detection - Card: {card_score}, Explanatory: {explanatory_score}")
        logger.info(f"Model mention: {has_model_mention}")
        
        # Final decision with clear thresholds
        if explanatory_score >= card_score + 2:  # Explanatory needs clear advantage
            return "explanation"
        elif card_score > explanatory_score:
            return "card"
        else:
            # Tie-breaker: default to explanatory for advice/opinion queries
            advice_indicators = ['worth', 'should', 'good', 'better', 'recommend', 'advice']
            if any(indicator in query_lower for indicator in advice_indicators):
                return "explanation"
            return "card"
        
    def _build_messages(self, query: str, contexts: List[SearchResult], 
                       session_memory: List[str], response_mode: str) -> List[BaseMessage]:
        try:
            if not query or not query.strip():
                return [HumanMessage(content="Please provide a valid question about mobile phones.")]
            
            context_text = self._enhance_context_with_specs(contexts)
            history_text = self._format_history(session_memory)
            
            if response_mode == "card":
                system_content = self.card_system_prompt.format(
                    context=context_text,
                    history=history_text,
                    query=query
                )
            else:
                system_content = self.explanatory_system_prompt.format(
                    context=context_text,
                    history=history_text,
                    query=query
                )
            
            logger.info(f"Using {response_mode.upper()} mode for query: '{query}'")
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            return messages
            
        except Exception as e:
            logger.error(f"Message building failed: {e}")
            return [HumanMessage(content=query)]
    
    def _enhance_context_with_specs(self, contexts: List[SearchResult]) -> str:
        if not contexts:
            return "No product information available in the current database."
        
        context_lines = ["📱 AVAILABLE PHONE SPECIFICATIONS:"]
        
        for i, context in enumerate(contexts, 1):
            try:
                clean_text = re.sub(r'\s+', ' ', context.text).strip()
                
                enhanced_text = self._highlight_specs(clean_text)
                
                if len(enhanced_text) > 400:
                    enhanced_text = enhanced_text[:400] + "..."
                
                context_lines.append(f"\n{i}. {enhanced_text}")
                
            except Exception as e:
                logger.warning(f"Failed to format context {i}: {e}")
                continue
        
        context_lines.append("\n🔍 RESPONSE GUIDELINES:")
        context_lines.append("- Provide complete phone cards with all standard fields")
        context_lines.append("- Include Model, Price, RAM, Storage, Display, Camera, Battery, Rating, Brand")
        context_lines.append("- Ensure every field has appropriate values")
        context_lines.append("- Maintain consistent card formatting")
        
        return "\n".join(context_lines)
    
    def _highlight_specs(self, text: str) -> str:
        patterns = {
            'ram': r'(\d+\s*GB\s*RAM)',
            'storage': r'(\d+\s*GB\s*Storage)',
            'camera': r'(\d+\s*MP[^,]*camera)',
            'display': r'(\d+\.\d+\s*inch[^,]*|[\d.]+\s*inches[^,]*)',
            'battery': r'(\d+\s*mAh)',
            'rating': r'([\d.]+\s*\/\s*5|\d+\s*stars)',
            'price': r'(₹\s*[\d,]+)'
        }
        
        highlighted = text
        for key, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = match.group(1)
                highlighted = highlighted.replace(spec, f"**{spec}**")
        
        return highlighted

    def _format_contexts(self, contexts: List[SearchResult]) -> str:
        if not contexts:
            return "No product information available in the current database."
        
        context_lines = ["AVAILABLE PRODUCT INFORMATION:"]
        
        for i, context in enumerate(contexts, 1):
            try:
                clean_text = re.sub(r'\s+', ' ', context.text).strip()
                if len(clean_text) > 300:
                    clean_text = clean_text[:300] + "..."
                context_lines.append(f"\n{i}. {clean_text}")
            except Exception as e:
                logger.warning(f"Failed to format context {i}: {e}")
                continue
        
        return "\n".join(context_lines)
    
    def _format_history(self, session_memory: List[str]) -> str:
        if not session_memory:
            return "No previous conversation in this session."
        
        history_lines = ["PREVIOUS CONVERSATION:"]
        added_count = 0
        
        for memory in session_memory[-3:]:
            if " | " in memory:
                try:
                    question, answer = memory.split(" | ", 1)
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
        sources = []
        for context in contexts:
            try:
                if (context.source and 
                    context.source not in sources and 
                    context.source != 'Unknown' and
                    len(sources) < 5):
                    sources.append(context.source)
            except Exception as e:
                logger.warning(f"Source extraction failed: {e}")
                continue
        return sources
    
    def _validate_and_format_response(self, response: str, mode: str = "card") -> str:
        if not response or not response.strip():
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        try:
            cleaned = response.strip()
            
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'\.{3,}', '...', cleaned)
            cleaned = re.sub(r'\!{2,}', '!', cleaned)
            
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
            cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)
            cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
            
            if mode == "card":
                cleaned = self._fix_card_formatting(cleaned)
                
                emoji_replacements = {
                    "📱": "Phone", "💰": "Price:", "⭐": "Rating:",
                    "🏷️": "Brand:", "🔍": "Search", "💡": "Tip",
                    "🤔": "", "⚠️": "", "😔": "", "🎯": "", "✅": "",
                    "🔥": "", "💯": "", "👍": "", "👎": "", "❌": ""
                }
                
                for emoji, replacement in emoji_replacements.items():
                    cleaned = cleaned.replace(emoji, replacement)
                
                cleaned = self._fix_rating_format(cleaned)
                
                replacements = {
                    "Memory: nan": "Memory: Not specified",
                    "Rating: nan": "Rating: Not rated",
                    "Price: ₹nan": "Price: Not specified", 
                    "Color: nan": "Color: Various",
                    "Storage: nan": "Storage: Not specified",
                    "Brand: nan": "Brand: Not specified",
                    "Display: nan": "Display: Not specified",
                    "Camera: nan": "Camera: Not specified", 
                    "Battery: nan": "Battery: Not specified"
                }
                
                for old, new in replacements.items():
                    cleaned = cleaned.replace(old, new)
                    
            else:
                cleaned = re.sub(r'^\s*[\*\-\+]\s+', '- ', cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r'\.\s*([A-Z])', r'.\n\n\1', cleaned)
            
            if cleaned and not cleaned[-1] in ['.', '!', '?']:
                cleaned += '.'
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return "Here's the information based on your query."
    
    def _fix_card_formatting(self, text: str) -> str:
        text = re.sub(r'Rating: Rating:', 'Rating:', text)
        
        text = re.sub(r'Rating: ₹?\d+[,.]?\d*/5', lambda m: m.group(0).replace('Rating: ', 'Rating: ').replace('₹', '').split('/')[0] + '/5', text)
        
        text = re.sub(r'┌─\s+┐', '┌─ PHONE CARD ──────────────────────┐', text)
        text = re.sub(r'└─\s+┘', '└───────────────────────────────────┘', text)
        
        return text
    
    def _fix_rating_format(self, text: str) -> str:
        price_rating_pattern = r'Rating: (₹?\d+[,.]?\d*)/5'
        
        def replace_price_rating(match):
            rating_value = match.group(1).replace('₹', '').replace(',', '')
            try:
                rating_num = float(rating_value)
                if rating_num > 10:
                    return "Rating: 4.2/5"
                else:
                    return f"Rating: {rating_num}/5"
            except:
                return "Rating: 4.2/5"
        
        text = re.sub(price_rating_pattern, replace_price_rating, text)
        
        rating_pattern = r'Rating: ([\d.]+)/5'
        
        def validate_rating(match):
            try:
                rating = float(match.group(1))
                if rating < 1 or rating > 5:
                    return "Rating: 4.2/5"
                return match.group(0)
            except:
                return "Rating: 4.2/5"
        
        text = re.sub(rating_pattern, validate_rating, text)
        
        return text

    def generate_answer(self, query: str, session_id: Optional[str] = None) -> GenerationResponse:
        start_time = pd.Timestamp.now()
        
        try:
            if not query or not query.strip():
                return self._create_empty_query_response(session_id)
            
            if len(query) > 1000:
                query = query[:500] + "..."
                logger.info(f"Query truncated due to excessive length")
            
            logger.info(f"Generation START: '{query}' (session: {session_id})")
            
            session_memory = self.memory_manager.load_memory(session_id)
            
            resolved_query, context_model = self.query_resolver.resolve(query, list(session_memory))
            
            contexts = self.retriever.hybrid_search(resolved_query, top_k=15)
            
            response_mode = self._determine_response_mode(resolved_query, contexts)
            
            if not contexts:
                answer = self._handle_no_results(resolved_query, context_model, response_mode)
                sources = []
            else:
                messages = self._build_messages(resolved_query, contexts, list(session_memory), response_mode)
                llm_response = self.llm.invoke(messages)
                answer = self._extract_llm_content(llm_response)
                answer = self._validate_and_format_response(answer, response_mode)
                sources = self._extract_sources(contexts)
            
            if session_id and answer:
                self._update_memory(session_id, query, answer, session_memory)
            
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            debug_info = {
                "resolved_query": resolved_query,
                "context_model": context_model,
                "contexts_count": len(contexts),
                "sources_count": len(sources),
                "memory_entries": len(session_memory),
                "processing_time_seconds": round(processing_time, 2),
                "response_mode": response_mode,
                "query_length": len(query)
            }
            
            logger.info(f"Generation END: {response_mode.upper()} mode, {len(answer)} chars, {len(sources)} sources, {processing_time:.2f}s")
            
            return GenerationResponse(
                query=query,
                answer=answer,
                sources=sources,
                session_id=session_id,
                debug_info=debug_info,
                resolved_query=resolved_query,
                output_format=response_mode
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            return self._create_error_response(query, session_id, str(e))
    
    def _create_empty_query_response(self, session_id: Optional[str]) -> GenerationResponse:
        return GenerationResponse(
            query="",
            answer="Please provide a specific question about mobile phones.\n\nExamples:\n- Phones under ₹20,000\n- Best camera phone\n- OPPO A53 specifications\n- Gaming phones with 8GB RAM",
            sources=[],
            session_id=session_id,
            debug_info={"error": "Empty query"},
            resolved_query="",
            output_format="explanation"
        )
    
    def _handle_no_results(self, query: str, context_model: Optional[str], mode: str) -> str:
        base_msg = "I couldn't find exact matches in our current database."
        
        suggestions = []
        
        if context_model:
            suggestions.append(f"- '{context_model}' might be out of stock or discontinued")
            suggestions.append("- Check Flipkart's official website for latest availability")
            suggestions.append("- Try searching for similar models from the same brand")
        else:
            if mode == "card":
                suggestions.append("- Check the spelling of the model name")
                suggestions.append("- Try searching with specific specifications (RAM, storage, etc.)")
            else:
                suggestions.append("- Try being more specific with your requirements")
                suggestions.append("- Consider different price ranges or use cases")
                suggestions.append("- Check recent reviews and comparisons online")
        
        suggestions.append("- Visit Flipkart.com for the most up-to-date information")
        
        suggestion_text = "\n".join(suggestions)
        
        return f"{base_msg}\n\nSuggestions:\n{suggestion_text}"
    
    def _extract_llm_content(self, llm_response) -> str:
        try:
            if hasattr(llm_response, 'content'):
                content = str(llm_response.content)
            elif isinstance(llm_response, dict):
                content = str(llm_response.get('text', llm_response.get('content', 'No response generated')))
            else:
                content = str(llm_response)
            
            if not content or content.strip() == "" or content == "No response generated":
                return "I received an empty response. Please try asking your question differently."
            
            return content
            
        except Exception as e:
            logger.error(f"LLM response extraction failed: {e}")
            return "I encountered an issue while processing your request. Please try again."
    
    def _update_memory(self, session_id: str, query: str, answer: str, session_memory: deque):
        try:
            if not query or not answer:
                return
                
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            memory_entry = f"Q: {query} | A: {answer}"
            session_memory.append(memory_entry)
            
            success = self.memory_manager.save_memory(session_id, session_memory)
            if success:
                logger.debug(f"Updated memory for session {session_id}")
            else:
                logger.warning(f"Memory update failed for session {session_id}")
                
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
    
    def _create_error_response(self, query: str, session_id: Optional[str], error: str) -> GenerationResponse:
        return GenerationResponse(
            query=query,
            answer="I'm experiencing technical difficulties at the moment. Please try again in a few moments or rephrase your question.",
            sources=[],
            session_id=session_id,
            debug_info={"error": error, "type": "generation_error"},
            resolved_query=query,
            output_format="explanation"
        )
    
    def clear_memory(self, session_id: str) -> bool:
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

def create_dual_mode_generator(retriever, llm, memory_db=None, memory_size: int = 10):
    return DualModeRAGGenerator(retriever, llm, memory_db, memory_size)