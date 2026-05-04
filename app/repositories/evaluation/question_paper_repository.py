# app/repositories/evaluation/question_paper_repository.py

from typing import Optional, List, Dict
from uuid import UUID
from sqlalchemy.orm import Session

from app.shared.models.question_papers import QuestionPaper, Question, SubQuestion


class QuestionPaperRepository:
    """Data access layer for question paper models."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_question_paper(
        self,
        resource_id: UUID,
        chat_session_id: Optional[UUID] = None,
        evaluation_session_id: Optional[UUID] = None,
        extracted_text: Optional[str] = None
    ) -> QuestionPaper:
        """Create a question paper entry."""
        question_paper = QuestionPaper(
            chat_session_id=chat_session_id,
            evaluation_session_id=evaluation_session_id,
            resource_id=resource_id,
            extracted_text=extracted_text
        )
        self.db.add(question_paper)
        self.db.commit()
        self.db.refresh(question_paper)
        return question_paper
    
    def get_question_paper(self, question_paper_id: UUID) -> Optional[QuestionPaper]:
        """Get question paper by ID."""
        return self.db.query(QuestionPaper).filter(
            QuestionPaper.id == question_paper_id
        ).first()
    
    def get_question_papers_by_chat_session(
        self,
        chat_session_id: UUID
    ) -> List[QuestionPaper]:
        """Get all question papers for a chat session."""
        return self.db.query(QuestionPaper).filter(
            QuestionPaper.chat_session_id == chat_session_id
        ).all()

    def delete_question_papers_by_chat_session(
        self,
        chat_session_id: UUID
    ) -> None:
        """Delete all question papers for a chat session."""
        papers = self.get_question_papers_by_chat_session(chat_session_id)
        paper_ids = [paper.id for paper in papers]
        if not paper_ids:
            return

        question_ids = [
            question_id
            for (question_id,) in self.db.query(Question.id)
            .filter(Question.question_paper_id.in_(paper_ids))
            .all()
        ]
        if question_ids:
            self.db.query(SubQuestion).filter(
                SubQuestion.question_id.in_(question_ids)
            ).delete(synchronize_session=False)
            self.db.query(Question).filter(
                Question.id.in_(question_ids)
            ).delete(synchronize_session=False)

        self.db.query(QuestionPaper).filter(
            QuestionPaper.chat_session_id == chat_session_id
        ).delete(synchronize_session=False)
        self.db.commit()

    def get_question_papers_by_evaluation_session(
        self,
        evaluation_session_id: UUID
    ) -> List[QuestionPaper]:
        """Get all question papers for an evaluation session."""
        return self.db.query(QuestionPaper).filter(
            QuestionPaper.evaluation_session_id == evaluation_session_id
        ).all()
    
    def create_question(
        self,
        question_paper_id: UUID,
        question_number: str,
        question_text: str,
        max_marks: Optional[int] = None,
        part_name: Optional[str] = None,
        shared_stem: Optional[str] = None,
        inherits_shared_stem_from: Optional[str] = None,
        correct_answer: Optional[str] = None,
    ) -> Question:
        """Create a main question."""
        question = Question(
            question_paper_id=question_paper_id,
            question_number=question_number,
            question_text=question_text,
            max_marks=max_marks,
            part_name=part_name,
            shared_stem=shared_stem,
            inherits_shared_stem_from=inherits_shared_stem_from,
            correct_answer=correct_answer,
        )
        self.db.add(question)
        self.db.commit()
        self.db.refresh(question)
        return question
    
    def get_questions_by_paper(self, question_paper_id: UUID) -> List[Question]:
        """Get all questions for a question paper (Natural Sort)."""
        questions = self.db.query(Question).filter(
            Question.question_paper_id == question_paper_id
        ).all()

        def natural_sort_key(q):
            import re
            text = q.question_number or ""
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

        return sorted(questions, key=natural_sort_key)
    
    def create_sub_question(
        self,
        question_id: UUID,
        label: str,
        sub_question_text: str,
        max_marks: Optional[int] = None,
        parent_sub_question_id: Optional[UUID] = None,
        correct_answer: Optional[str] = None,
    ) -> SubQuestion:
        """Create a sub-question."""
        sub_question = SubQuestion(
            question_id=question_id,
            label=label,
            sub_question_text=sub_question_text,
            max_marks=max_marks,
            parent_sub_question_id=parent_sub_question_id,
            correct_answer=correct_answer,
        )
        self.db.add(sub_question)
        self.db.commit()
        self.db.refresh(sub_question)
        return sub_question
    
    def get_sub_questions_by_question(self, question_id: UUID) -> List[SubQuestion]:
        """Get all sub-questions for a main question."""
        return self.db.query(SubQuestion).filter(
            SubQuestion.question_id == question_id
        ).order_by(SubQuestion.label).all()
    

    def create_structured_questions(
        self,
        question_paper_id: UUID,
        structured_data: Dict,
        part_name: Optional[str] = None
    ) -> List[Question]:
        """
        Create questions and sub-questions from structured data.
        """
        created_questions = []
        
        for q_num, q_data in structured_data.items():
            # Handle options for MCQs by appending to text if present
            q_text = q_data.get("text", "")
            if "options" in q_data and isinstance(q_data["options"], list):
                options_text = "\n".join(q_data["options"])
                q_text = f"{q_text}\n\nOptions:\n{options_text}"
            
            # Create main question
            question = self.create_question(
                question_paper_id=question_paper_id,
                question_number=str(q_num),
                question_text=q_text,
                max_marks=q_data.get("marks"),
                part_name=part_name,
                shared_stem=q_data.get("shared_stem"),
                inherits_shared_stem_from=str(q_data.get("inherits_shared_stem_from")) if q_data.get("inherits_shared_stem_from") else None,
                correct_answer=q_data.get("correct_answer")
            )
            created_questions.append(question)
            
            # Handle sub-questions
            if "sub_questions" in q_data and isinstance(q_data["sub_questions"], dict):
                self._create_recursive_sub_questions(
                    question.id, 
                    q_data["sub_questions"]
                )
                
        return created_questions

    def _create_recursive_sub_questions(
        self, 
        question_id: UUID, 
        sub_questions_data: Dict, 
        parent_sub_question_id: Optional[UUID] = None
    ):
        for label, sub_data in sub_questions_data.items():
            # Create sub-question
            sub_q = self.create_sub_question(
                question_id=question_id,
                label=str(label),
                sub_question_text=sub_data.get("text", ""),
                max_marks=sub_data.get("marks"),
                parent_sub_question_id=parent_sub_question_id,
                correct_answer=sub_data.get("correct_answer")
            )
            
            # Check for nested sub-questions (if any)
            if "sub_questions" in sub_data and isinstance(sub_data["sub_questions"], dict):
                 self._create_recursive_sub_questions(
                    question_id,
                    sub_data["sub_questions"],
                    parent_sub_question_id=sub_q.id
                )

    def get_question_paper_with_questions(
        self,
        question_paper_id: UUID
    ) -> Optional[Dict]:
        """Get complete question paper with all questions and sub-questions in hierarchical structure."""
        paper = self.get_question_paper(question_paper_id)
        if not paper:
            return None
        
        questions = self.get_questions_by_paper(question_paper_id)
        
        def build_sub_question_tree(sub_questions: List[SubQuestion], parent_id=None):
            """Recursively build hierarchical sub-question structure."""
            children = []
            for sq in sub_questions:
                if sq.parent_sub_question_id == parent_id:
                    child_tree = {
                        "id": sq.id,
                        "label": sq.label,
                        "text": sq.sub_question_text,
                        "max_marks": sq.max_marks,
                        "parent_sub_question_id": sq.parent_sub_question_id,
                        "children": build_sub_question_tree(sub_questions, sq.id)
                    }
                    children.append(child_tree)
            return children
        
        result = {
            "id": paper.id,
            "evaluation_session_id": paper.evaluation_session_id,
            "resource_id": paper.resource_id,
            "extracted_text": paper.extracted_text,
            "created_at": paper.created_at,
            "questions": []
        }
        
        for question in questions:
            sub_questions = self.get_sub_questions_by_question(question.id)
            result["questions"].append({
                "id": question.id,
                "question_number": question.question_number,
                "question_text": question.question_text,
                "max_marks": question.max_marks,
                "sub_questions": build_sub_question_tree(sub_questions)
            })
        
        return result
