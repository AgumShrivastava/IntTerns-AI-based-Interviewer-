import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AIEvaluator:
    def __init__(self):
        print("System: Waking up the AI Evaluator engine...")
        # Loads the model once when the class is created
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_soft_skills(self, answer_text):
        """ Penalizes filler words and assesses confidence. """
        filler_pattern = r'\b(um|uh|like|basically|actually|literally|you know|sort of|kind of)\b'
        fillers_found = re.findall(filler_pattern, answer_text.lower())
        filler_count = len(fillers_found)
        
        penalty = 0.0
        if filler_count >= 4:
            penalty = 1.5
        elif filler_count >= 2:
            penalty = 0.5
        return penalty

    def evaluate_answer(self, ideal_answer, candidate_answer, is_behavioral=False):
        """ Evaluates the answer using math and NLP. """
        clean_ans = candidate_answer.lower().strip().replace(".", "").replace(",", "")
        word_count = len(candidate_answer.split())
        
        # Check for skips
        skip_phrases = ["i dont know", "i do not know", "idk", "not sure", "sorry", "pass", "no idea", "no", "skip"]
        if clean_ans in skip_phrases or any(phrase in clean_ans for phrase in ["dont know", "no idea", "not sure"]):
            return {"score_out_of_10": 0.0, "skipped_question": True}

        # Check for extremely short answers
        if word_count < 5:
            return {"score_out_of_10": 1.0, "skipped_question": False}

        # Apply soft skill penalty
        soft_skill_penalty = self.evaluate_soft_skills(candidate_answer)

        # Behavioral scoring (length & confidence)
        if is_behavioral:
            base_score = min(10.0, word_count / 5.0) 
            final_score = max(0.0, base_score - soft_skill_penalty)
            return {"score_out_of_10": round(final_score, 1), "skipped_question": False}

        # Technical scoring (Cosine Similarity)
        ideal_embedding = self.model.encode([ideal_answer])
        candidate_embedding = self.model.encode([candidate_answer])

        similarity_score = cosine_similarity(ideal_embedding, candidate_embedding)[0][0]
        base_score = max(0, similarity_score * 10)
        
        final_score = round(max(0, base_score - soft_skill_penalty), 1)

        return {"score_out_of_10": final_score, "skipped_question": False}
