import pandas as pd
import numpy as np
import random
import time
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AIEvaluator:
    def __init__(self):
        print("System: Waking up the AI Evaluator engine...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_soft_skills(self, answer_text):
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
        clean_ans = candidate_answer.lower().strip().replace(".", "").replace(",", "")
        word_count = len(candidate_answer.split())
        
        skip_phrases = ["i dont know", "i do not know", "idk", "not sure", "sorry", "pass", "no idea", "no", "skip"]
        if clean_ans in skip_phrases or any(phrase in clean_ans for phrase in ["dont know", "no idea", "not sure"]):
            return {"score_out_of_10": 0.0, "skipped_question": True}

        if word_count < 5:
            return {"score_out_of_10": 1.0, "skipped_question": False}

        soft_skill_penalty = self.evaluate_soft_skills(candidate_answer)

        if is_behavioral:
            base_score = min(10.0, word_count / 5.0) 
            final_score = max(0.0, base_score - soft_skill_penalty)
            return {"score_out_of_10": round(final_score, 1), "skipped_question": False}

        ideal_embedding = self.model.encode([ideal_answer])
        candidate_embedding = self.model.encode([candidate_answer])

        similarity_score = cosine_similarity(ideal_embedding, candidate_embedding)[0][0]
        base_score = max(0, similarity_score * 10)
        
        final_score = round(max(0, base_score - soft_skill_penalty), 1)
        return {"score_out_of_10": final_score, "skipped_question": False}

# ==========================================
# PART 2: THE INTERVIEW LOOP
# ==========================================
intro_greetings = [
    "Hello there! I'm your AI Interviewer today. Thank you for taking the time to speak with me.",
    "Hi! Welcome to the interview. I'll be guiding you through a few technical and behavioral questions today.",
    "Good to meet you! Let's get started with our interview process."
]
positive_reactions = ["Great explanation.", "That makes a lot of sense.", "Spot on.", "I appreciate the detail there."]
neutral_reactions = ["Got it.", "Understood.", "Thank you for that answer.", "Okay, taking note of that."]
encouraging_reactions = ["That's a tricky one, thanks for your thoughts.", "No worries, let's pivot to something else."]
transitions = [
    "Moving on to the next topic...",
    "Let's switch gears a bit.",
    "For my next question..."
]

def start_interview(dataframe, evaluator, min_questions=5, max_questions=10, target_score=7.5, fail_score=2.5):
    print("\n" + "="*50)
    
    candidate_name = input("System: Please enter your name to begin: ")
    print("\n" + "="*50)
    print(f"Interviewer: {random.choice(intro_greetings)} It's great to have you here, {candidate_name}.")
    time.sleep(1) 
    
    used_indices = set()
    total_score = 0.0
    questions_asked = 0
    last_score = None 
    
    print(f"\nInterviewer: To kick things off, {candidate_name}, tell me a little bit about yourself and your background.")
    ice_ans = input(f"{candidate_name}: ")
    
    ice_eval = evaluator.evaluate_answer("", ice_ans, is_behavioral=True)
    
    if ice_eval['score_out_of_10'] > 5.0:
        print("\nInterviewer: That's wonderful to hear. It sounds like you have a great foundation.")
    else:
        print("\nInterviewer: Thanks for sharing that. It's good to learn a bit more about you.")
        
    time.sleep(1)
    print("Interviewer: Alright, let's dive into the technical portion of the interview.")
    time.sleep(1)
    
    current_difficulty = 'Medium' 
    
    while questions_asked < max_questions:
        available_qs = dataframe[(dataframe['difficulty'].str.lower() == current_difficulty.lower()) & (~dataframe.index.isin(used_indices))]
        if available_qs.empty:
            available_qs = dataframe[~dataframe.index.isin(used_indices)]
            if available_qs.empty: break
                
        selected_index = available_qs.sample(1).index[0]
        used_indices.add(selected_index)
        
        row = dataframe.loc[selected_index]
        question = row['question']
        ideal_answer = row['answer']

        print("\n" + "-"*40)
        
        if questions_asked > 0:
            if last_score is not None:
                if last_score >= 8.0:
                    reaction = random.choice(positive_reactions)
                elif last_score < 5.0:
                    reaction = random.choice(encouraging_reactions)
                else:
                    reaction = random.choice(neutral_reactions)
                print(f"Interviewer: {reaction} {random.choice(transitions)}")
                time.sleep(0.5)

        print(f"Interviewer: {question}")
        candidate_answer = input(f"{candidate_name}: ")

        evaluation = evaluator.evaluate_answer(ideal_answer, candidate_answer, is_behavioral=False)
        last_score = evaluation['score_out_of_10']
        
        if last_score >= 8.0:
            current_difficulty = 'Hard'
        elif last_score < 5.0:
            current_difficulty = 'Easy'
        else:
            current_difficulty = 'Medium'
            
        total_score += last_score
        questions_asked += 1
        current_average = total_score / questions_asked
        should_exit = False

        if questions_asked >= min_questions:
            if current_average >= target_score or current_average < fail_score:
                should_exit = True

        if evaluation.get("skipped_question"):
            print("\nInterviewer: No problem at all, we can skip that one.")
            if should_exit: break
            continue

        if should_exit: break

    print("\n" + "="*50)
    print(f"Interviewer: Well {candidate_name}, that concludes all my questions for today. I really appreciate your time.")
    if current_average >= target_score:
        print("Interviewer: You had some really strong technical answers. Our HR team will be in touch very soon!")
    elif current_average >= 5.0:
        print("Interviewer: Thank you for sharing your knowledge. We will review your profile and get back to you.")
    else:
        print("Interviewer: Thank you for your time. We encourage you to keep studying the fundamentals, and we wish you the best.")
    
    print("\n" + "#"*50)
    print("SYSTEM FINAL REPORT (For HR Eyes Only)")
    print(f"Final Average Score: {current_average:.1f} / 10")
    print("#"*50)

# ==========================================
# PART 3: RUN THE PROGRAM
# ==========================================
if __name__ == "__main__":
    try:
        # Adjusted to match your exact Windows file path
        df = pd.read_csv(r"C:\Users\hp\Desktop\6 sem\InTerns\Datasets\Software Questions.csv", encoding='latin1')
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.dropna()
        
        evaluator = AIEvaluator()
        start_interview(df, evaluator, min_questions=5, max_questions=10, target_score=7.5, fail_score=2.5)
        
    except FileNotFoundError:
        print("Error: Could not find your CSV file. Please double check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
