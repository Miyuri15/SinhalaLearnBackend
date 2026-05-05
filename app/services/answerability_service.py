# app/services/answerability_service.py
import logging
import re
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AnswerabilityService:
    """
    Determines whether a question can be answered
    using the retrieved context only.
    """

    # Stopwords for Sinhala and English
    STOPWORDS = {
        # Sinhala stopwords
        # "යනු", "කුමක්ද",  "මොනවාද", "මගින්"
        "ද", "පිළිබඳ", "පිලිබඳ",
        "කරුණාකර", "පැහැදිලි", "කරන්න", "සඳහා", "මෙම", "මෙහි", "මෙහී",
        "එම", "ඒ", "මේ", "සහ", "හා", "තුළ", "අනුව",
        "ලෙස", "වන", "වූ", "බව", "සිට", "දක්වා",
        "පමණ", "හැකි", "විය", "ය", "සමග",
        "වැනි", "පිළිබඳව", "ගැන", "අතර", "සඳහන්",
        "දෙන්න", "ගන්න", "වෙන්න", "විසින්", "ලියන", 
        "ලද්දේ", "ලද්දේද", "කවුරුන්", "කුමන", "මොනවා",
        "ඇයි", "කොහේද", "කවදාද", "කොහොමද", "කියන්න",
        "ලැබුණේද", "ලැබුනේද", "ලැබුනාද", "ලැබුණාද", "ලැබුණේ", "කවුද", "කොහොම", "කොහොමත්", "කොහොමහරි",
        # English stopwords
        "what", "is", "are", "the", "of", "in", "on", "at",
        "to", "for", "with", "by", "about", "as", "an",
        "hello", "hi", "hey", "good", "morning", "afternoon", "evening",
        "give", "get", "me", "summary"
    }

    # Greeting terms for special handling
    GREETING_TERMS = {
        "හායි", "හලෝ", "ආයුබෝවන්", "කොහොමද", 
        "ගුඩ් මෝනින්", "good morning", "good afternoon", 
        "good evening", "hello", "hi", "hey"
    }

    # Content generation intents - these are ALWAYS answerable if chunks exist
    # These intents ask to GENERATE content FROM the provided material
    CONTENT_GENERATION_INTENTS = {
        "summary",      # "Summarize this content" - always possible if content exists
        "qa_generate"   # "Generate questions from this" - always possible if content exists
    }
    
    # Topic-specific intents - these require the topic to be IN the content
    # "explanation" is NOT in CONTENT_GENERATION_INTENTS because it asks about a specific topic

    @staticmethod
    def extract_key_terms(text: str) -> List[str]:
        """
        Extract meaningful content words from text.
        Handles both Sinhala and English.
        """
        # Extract words (Sinhala Unicode range + English letters)
        words = re.findall(r"[a-zA-Zඅ-෴]+", text.lower())
        
        # Filter out stopwords and very short words
        key_terms = [
            word for word in words 
            if word not in AnswerabilityService.STOPWORDS 
            and len(word) > 2
        ]
        
        return key_terms

    @staticmethod
    def extract_main_topic(question: str) -> List[str]:
        """
        Extract the main topic(s) from a question.
        For "Explain X", this returns X as the main topic.
        """
        question_lower = question.lower()
        
        # Pattern 1: "X පැහැදිලි කරන්න" or "X විස්තර කරන්න"
        pattern1 = r"([\u0d80-\u0dff]+(?:\s+[\u0d80-\u0dff]+)*)\s+(?:පැහැදිලි|විස්තර)\s+කරන්න"
        match1 = re.search(pattern1, question_lower)
        if match1:
            topic = match1.group(1).strip()
            logger.debug(f"Extracted topic using pattern1: '{topic}'")
            return [topic]
        
        # Pattern 2: "X ගැන පැහැදිලි කරන්න" or "X ගැන විස්තර කරන්න"
        pattern2 = r"([\u0d80-\u0dff]+(?:\s+[\u0d80-\u0dff]+)*)\s+ගැන\s+(?:පැහැදිලි|විස්තර)\s+කරන්න"
        match2 = re.search(pattern2, question_lower)
        if match2:
            topic = match2.group(1).strip()
            logger.debug(f"Extracted topic using pattern2: '{topic}'")
            return [topic]
        
        # # Pattern 3: "X යනු කුමක්ද" (What is X)
        pattern3 = r"([\u0d80-\u0dff]+(?:\s+[\u0d80-\u0dff]+)*)\s+යනු\s+කුමක්ද"
        match3 = re.search(pattern3, question_lower)
        if match3:
            topic = match3.group(1).strip()
            logger.debug(f"Extracted topic using pattern3: '{topic}'")
            return [topic]
        
        # Pattern 4: "X කියන්න" (Tell about X)
        pattern4 = r"([\u0d80-\u0dff]+(?:\s+[\u0d80-\u0dff]+)*)\s+කියන්න"
        match4 = re.search(pattern4, question_lower)
        if match4:
            topic = match4.group(1).strip()
            logger.debug(f"Extracted topic using pattern4: '{topic}'")
            return [topic]
        
        # Pattern 5: "Explain X" or "Tell me about X" (English)
        pattern5 = r"(?:explain|tell\s+me\s+about|describe)\s+([a-zA-Z\s]+)"
        match5 = re.search(pattern5, question_lower)
        if match5:
            topic = match5.group(1).strip()
            logger.debug(f"Extracted topic using pattern5: '{topic}'")
            return [topic]
        
        # Fallback: extract all meaningful terms
        words = re.findall(r"[a-zA-Zඅ-෴]{3,}", question_lower)
        meaningful = [w for w in words if w not in AnswerabilityService.STOPWORDS]
        
        if meaningful:
            logger.debug(f"Fallback extracted topics: {meaningful[:3]}")
            return meaningful[:3]
        
        logger.debug("No clear topic found in question")
        return []

    @staticmethod
    def calculate_relevance_score(question: str, context: str, chunks: List[Dict]) -> float:
        """
        Calculate a comprehensive relevance score between question and context.
        Returns a score between 0 and 1.
        """
        if not context or not context.strip():
            return 0.0

        # Extract question terms
        question_terms = set(AnswerabilityService.extract_key_terms(question))
        context_lower = context.lower()
        
        # If question has no content terms, rely on chunk similarity
        if not question_terms:
            logger.debug("No content terms in question, relying on chunk similarity")
            chunk_similarities = []
            if chunks:
                for chunk in chunks[:3]:
                    sim_score = chunk.get("similarity")
                    if sim_score is not None:
                        chunk_similarities.append(sim_score)
            
            avg_chunk_similarity = np.mean(chunk_similarities) if chunk_similarities else 0
            return avg_chunk_similarity
        
        # Method 1: Key term overlap (with boundary awareness)
        term_matches = 0
        matched_terms = []
        for term in question_terms:
            # Use a more flexible boundary for Sinhala to handle inflections
            # Still require start of word/whitespace or start of string
            # But allow common Sinhala suffixes or just whitespace/punctuation after
            if re.search(fr"(^|\s|[.,!?;]){re.escape(term)}", context_lower):
                term_matches += 1
                matched_terms.append(term)
        
        # If no terms match at all, score should be 0
        if term_matches == 0:
            logger.debug(f"No matches for terms: {question_terms}")
            return 0.0
        
        term_overlap_ratio = term_matches / len(question_terms)
        
        # Optimized overlap for short queries
        if len(question_terms) <= 3:
            if term_matches < 1:
                term_overlap_ratio = 0
            elif term_matches == 1:
                # Small penalty for only 1 match if query was multi-term
                if len(question_terms) > 1:
                    term_overlap_ratio *= 0.5
            elif term_matches == 2 and len(question_terms) == 3:
                # 2/3 is quite good in Sinhala due to spelling variations
                term_overlap_ratio = 0.8
        
        # Method 2: Use chunk similarity scores
        chunk_similarities = []
        if chunks:
            for chunk in chunks[:3]:
                sim_score = chunk.get("similarity")
                if sim_score is not None:
                    chunk_similarities.append(sim_score)
        
        avg_chunk_similarity = np.mean(chunk_similarities) if chunk_similarities else 0
        
        # Method 3: Check for exact phrase matches
        phrases = AnswerabilityService._extract_phrases(question)
        phrase_matches = 0
        for phrase in phrases:
            if phrase.lower() in context_lower:
                phrase_matches += 1
        
        phrase_score = min(phrase_matches / max(len(phrases), 1), 1.0) if phrases else 0
        
        # Combine scores with weights
        relevance_score = (
            term_overlap_ratio * 0.5 +      # Term overlap (higher weight)
            avg_chunk_similarity * 0.3 +     # Semantic similarity
            phrase_score * 0.2                # Phrase matches
        )
        
        # Cap at 1.0
        relevance_score = min(relevance_score, 1.0)
        
        logger.info(f"Relevance breakdown - Terms matched: {matched_terms}, "
                    f"Term overlap: {term_overlap_ratio:.3f}, "
                    f"Chunk similarity: {avg_chunk_similarity:.3f}, "
                    f"Phrase matches: {phrase_score:.3f} → Final: {relevance_score:.3f}")
        
        return relevance_score

    @staticmethod
    def _extract_phrases(text: str, max_phrase_length: int = 4) -> List[str]:
        """
        Extract meaningful phrases (n-grams) from text.
        """
        words = re.findall(r"[a-zA-Zඅ-෴]+", text.lower())
        phrases = []
        
        # Extract bigrams and trigrams
        for n in range(2, min(max_phrase_length + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                # Filter out phrases with stopwords at boundaries
                if phrase.split()[0] not in AnswerabilityService.STOPWORDS and \
                   phrase.split()[-1] not in AnswerabilityService.STOPWORDS:
                    phrases.append(phrase)
        
        return phrases

    @staticmethod
    def has_relevant_content(
        question: str, 
        context: str, 
        chunks: List[Dict], 
        intent: str = "qa_answer",
        threshold: float = 0.3
    ) -> bool:
        """
        Determine if the context contains content relevant to the question.
        Returns True if relevance score exceeds threshold.
        
        Intent-aware handling:
        - summary, qa_generate: ALWAYS answerable if chunks exist (generate from content)
        - explanation: MUST have topic match in context
        - qa_answer: MUST have term matches and meet threshold
        """
        # Handle empty cases
        if not context or not context.strip():
            logger.info("Empty context - no relevant content")
            return False
        
        # Check for greetings
        normalized = question.lower().strip()
        if any(greeting in normalized for greeting in AnswerabilityService.GREETING_TERMS):
            logger.info("Greeting detected - treating as answerable")
            return True
        
        # For TRUE content generation intents (summary, qa_generate), 
        # if we have chunks, the question is answerable
        if intent in AnswerabilityService.CONTENT_GENERATION_INTENTS:
            if chunks and len(chunks) > 0:
                logger.info(f"Content generation intent '{intent}' with chunks available - treating as answerable")
                return True
            else:
                logger.info(f"Content generation intent '{intent}' but no chunks - unanswerable")
                return False
        
        # For explanation intent, we need to check if the TOPIC exists in context
        if intent == "explanation":
            # Extract main topic from question
            topics = AnswerabilityService.extract_main_topic(question)
            
            if not topics:
                logger.info("No clear topic found in explanation question - treating as unanswerable")
                return False
            
            # Check if any topic appears in context (as a whole phrase)
            context_lower = context.lower()
            for topic in topics:
                if topic in context_lower:
                    logger.info(f"Explanation topic '{topic}' found in context - answerable")
                    return True
                
                # For multi-word topics, check if most words appear
                topic_words = topic.split()
                if len(topic_words) > 1:
                    matches = sum(1 for word in topic_words if word in context_lower)
                    match_ratio = matches / len(topic_words)
                    if match_ratio >= 0.7:  # 70% of words match
                        logger.info(f"Explanation topic '{topic}' partially found ({match_ratio:.0%} words) - answerable")
                        return True
            
            logger.info(f"Explanation topics {topics} not found in context - unanswerable")
            return False
        
        # For QA_ANSWER intents, we need strict relevance checking
        # Extract question terms
        question_terms = AnswerabilityService.extract_key_terms(question)
        
        # If the question has no meaningful content terms, treat as unanswerable
        if not question_terms:
            logger.info("Question has no meaningful content terms - treating as unanswerable")
            return False
        
        # Check if at least one key term appears in context
        context_lower = context.lower()
        has_term_match = any(term in context_lower for term in question_terms)
        
        if not has_term_match:
            logger.info(f"No key terms from question found in context: {question_terms[:5]}")
            return False
        
        # Calculate relevance score
        relevance_score = AnswerabilityService.calculate_relevance_score(question, context, chunks)
        
        # Determine if content is relevant
        is_relevant = relevance_score >= threshold
        
        logger.info(f"Content relevance: score={relevance_score:.3f}, "
                   f"threshold={threshold}, has_term_match={has_term_match}, "
                   f"relevant={is_relevant}")
        
        return is_relevant

    @staticmethod
    def is_answerable(question: str, context: str, intent: str = "qa_answer") -> bool:
        """
        Legacy method with intent awareness.
        """
        # Handle greetings
        normalized = question.lower().strip()
        if any(greeting in normalized for greeting in AnswerabilityService.GREETING_TERMS):
            return True
        
        # For summary and qa_generate, if context exists, it's answerable
        if intent in AnswerabilityService.CONTENT_GENERATION_INTENTS:
            return bool(context and context.strip())
        
        # For explanation, check if topic exists in context
        if intent == "explanation":
            topics = AnswerabilityService.extract_main_topic(question)
            if not topics:
                return False
            context_lower = context.lower()
            for topic in topics:
                if topic in context_lower:
                    return True
                topic_words = topic.split()
                if len(topic_words) > 1:
                    matches = sum(1 for word in topic_words if word in context_lower)
                    if matches / len(topic_words) >= 0.7:
                        return True
            return False
        
        key_terms = AnswerabilityService.extract_key_terms(question)

        if not key_terms:
            return False

        context_lower = context.lower()
        hits = sum(1 for t in key_terms if t.lower() in context_lower)

        # require at least one strong overlap
        return hits > 0