import os
import json
import re
import requests
import time
import sys
import argparse
import numpy as np
import google.generativeai as genai
from datetime import datetime, timedelta
from config import settings
from pydantic import BaseModel
from enum import Enum

# --- Configuration ---
API_BASE = settings.LM_BASE_URL
INGEST_URL = f"{API_BASE}/v4/ingest"
QUERY_URL = f"{API_BASE}/v4/query"
LOG_FILE = "locomo_benchmark_analysis.log"
FAILED_LOG_FILE = "failed_cases.log"

# --- Gemini Schema Definitions ---
class VerdictEnum(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"

class JudgeResponse(BaseModel):
    verdict: VerdictEnum
    reason: str

# Configure Judge
genai.configure(api_key=settings.GEMINI_API_KEY)
judge_model = genai.GenerativeModel("gemini-2.5-pro") 

def log_to_file(filename, entry):
    """Appends results to a specified log file."""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to {filename}: {e}")

def parse_locomo_date(date_str):
    try:
        date_str = date_str.strip()
        date_str = re.sub(r'\s+', ' ', date_str)
        return datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        return datetime.now()

def parse_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data

def extract_sessions(conversation_data):
    sessions = []
    # Filter keys that look like "session_X"
    session_keys = [k for k in conversation_data.keys() if re.match(r"session_\d+$", k)]
    # Sort by the number in session_X
    session_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    for key in session_keys:
        time_key = f"{key}_date_time"
        raw_time = conversation_data.get(time_key, "")
        dt_obj = parse_locomo_date(raw_time)
        sessions.append({
            "session_id": key, 
            "start_time": dt_obj, 
            "messages": conversation_data[key]
        })
    return sessions

def ingest_message(user_id, session_id, speaker, text, timestamp_obj):
    payload = {
        "user_id": user_id,
        "text": text,
        "session_id": session_id,
        "speaker": speaker,
        "created_at": timestamp_obj.isoformat()
    }
    try:
        requests.post(INGEST_URL, json=payload).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\n[Error Ingesting] {e}")

def evaluate_retrieval(question, answer, adversarial_answer, retrieved_context):
    adversarial_clause = ""
    if adversarial_answer:
        adversarial_clause = f"""
    3. ADVERSARIAL TRAP DETECTION (CRITICAL):
       - An **Adversarial Answer** is provided: "{adversarial_answer}"
       - The verdict MUST be FAIL if the **RETRIEVED CONTEXT** conclude the **Adversarial Answer**: "{adversarial_answer}".
       - If the context supports the trap, do not PASS even if the context seems otherwise relevant.
    """

    prompt = f"""
    Role: Retrieval Accuracy Judge
    Objective: Determine if the **RETRIEVED CONTEXT** contains sufficient information to conclude the correct answer for the given **QUESTION**.

    INPUT DATA:
    - **Question**: "{question}"
    - **Expected Answer**: "{answer}"
    - **Adversarial Trap**: "{adversarial_answer if adversarial_answer else 'None'}"
    - **Retrieved Context**: "{retrieved_context}"

    INSTRUCTIONS:
    1. TEMPORAL REASONING:
       - If the **Question** uses relative time (yesterday, last week), check the timestamps in the **Retrieved Context**.
    
    2. VERDICT CRITERIA:
       - PASS: The context contains sufficient information to conclude the **Expected Answer**.
       - FAIL: The context is missing information, is irrelevant, contradicts the answer.

    {adversarial_clause}

    OUTPUT FORMAT:
    Return valid JSON only.
    """
    try:
        resp = judge_model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": JudgeResponse
            }
        )
        data = json.loads(resp.text)
        score = 1 if data.get("verdict") == "PASS" else 0
        return score, data.get("reason", "No reason.")
    except Exception as e:
        return 0, f"Judge Error: {str(e)}"

def calculate_percentiles(latencies):
    if not latencies: return 0, 0, 0, 0
    arr = np.array(latencies)
    return (np.percentile(arr, 50), np.percentile(arr, 90), 
            np.percentile(arr, 95), np.percentile(arr, 99))

def print_progress(story_num, session_i, total_sessions, msg_i, total_msgs, user_id):
    bar_len = 20
    filled_len = int(bar_len * msg_i // total_msgs)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    
    status = (
        f"\rStory {story_num} | Sess [{session_i}/{total_sessions}] "
        f"Msg [{msg_i}/{total_msgs}] |{bar}| "
        f"ID: {user_id}"
    )
    sys.stdout.write(status)
    sys.stdout.flush()

def run_evaluation(dataset_path, test_index=None):
    open(LOG_FILE, "w").close() 
    open(FAILED_LOG_FILE, "w").close()
    
    print(f"📂 Loading dataset: {dataset_path}...")
    all_stories = parse_dataset(dataset_path)
    
    # Store tuples of (Story_Number, Story_Data)
    stories_to_process = []
    
    if test_index is not None:
        idx = test_index - 1
        if 0 <= idx < len(all_stories):
            # Pass the explicit test_index as the story number
            stories_to_process = [(test_index, all_stories[idx])]
            print(f"⚠️  RUNNING IN TEST MODE: Executing Story {test_index} Only")
        else:
            print(f"❌ Error: Story index {test_index} out of range (Max: {len(all_stories)})")
            return
    else:
        print("🚀 RUNNING FULL BENCHMARK (All Stories)")
        # Enumerate starting at 1 so Story 1 is index 1
        stories_to_process = list(enumerate(all_stories, 1))
    
    total_stories = len(stories_to_process)
    print("="*60)

    total_score = 0
    total_questions_count = 0
    all_latencies = []

    for story_num, story in stories_to_process:
        
        # The user_id is now strictly the story number
        user_id = f"story_{story_num}"
        
        sessions = extract_sessions(story["conversation"])
        total_sessions = len(sessions)
        
        print(f"\nProcessing Story {story_num} (ID: {user_id})")

        #--- Ingestion Phase ---
        for sess_idx, session in enumerate(sessions, 1):
            msg_list = session['messages']
            total_msgs = len(msg_list)
            curr_time = session['start_time']
            
            for m_idx, msg in enumerate(msg_list, 1):
                print_progress(story_num, sess_idx, total_sessions, m_idx, total_msgs, user_id)
                ingest_message(user_id, session['session_id'], msg['speaker'], msg['text'], curr_time)
                curr_time += timedelta(seconds=10) 
        
        print(f"\n✅ Ingested. Indexing (3s)...")
        time.sleep(3) 
        
        # --- QA Phase ---
        qa_list = story.get("qa", [])
        total_qa = len(qa_list)
        print(f"🔍 Running {total_qa} Queries...")
        
        for q_idx, qa in enumerate(qa_list, 1):
            q_text = qa.get("question")
            answer = qa.get("answer")
            adversarial_answer = qa.get("adversarial_answer","")

            sys.stdout.write(f"\r   -> Query [{q_idx}/{total_qa}]: {q_text[:40]}...")
            sys.stdout.flush()

            try:
                start = time.time()
                print(f"\n[DEBUG] Querying with user: {user_id}")
                resp = requests.post(QUERY_URL, json={"user_id": user_id, "query": q_text}).json()
                latency = time.time() - start
                all_latencies.append(latency)
                
                results = resp.get("results", [])
                
                context_lines = []
                for r in results:
                    text = r.get('text', '').strip()
                    context_lines.append(f"{text}")
                
                context_str = "\n".join(context_lines)
                
                score, reason = evaluate_retrieval(q_text, answer, adversarial_answer, context_str)
                total_score += score
                total_questions_count += 1

                detail_entry = {
                    "id": total_questions_count, 
                    "story_number": story_num,
                    "user_id": user_id,
                    "verdict": "PASS" if score else "FAIL",
                    "reason": reason, 
                    "question": q_text, 
                    "answer": answer, 
                    "context": context_str, 
                    "latency": round(latency, 4)
                }

                log_to_file(LOG_FILE, {**detail_entry})
                if score == 0:
                    log_to_file(FAILED_LOG_FILE, detail_entry)
                
                icon = '✅' if score else '❌'
                print(f"\r   [{q_idx}/{total_qa}] {icon} {latency:.2f}s | {q_text[:50]}...")

            except Exception as e:
                print(f"\n   [ERROR] Query {q_idx} failed: {e}")

        print("-" * 60)

    # --- FINAL SUMMARY ---
    if total_questions_count > 0:
        p50, p90, p95, p99 = calculate_percentiles(all_latencies)
        accuracy = (total_score / total_questions_count) * 100
        avg_lat = sum(all_latencies) / len(all_latencies)
        
        print("\n" + "="*30)
        print("BENCHMARK COMPLETE")
        print("="*30)
        print(f"Total Stories    : {total_stories}")
        print(f"Total Questions  : {total_questions_count}")
        print(f"Accuracy         : {accuracy:.2f}%")
        print(f"Avg Latency      : {avg_lat:.3f}s")
        print(f"P99 Latency      : {p99:.3f}s")
        print(f"Full Log         : {LOG_FILE}")
        print("="*30)

        final_summary = {
            "entry_type": "summary",
            "timestamp": datetime.now().isoformat(),
            "total_stories": total_stories,
            "total_questions": total_questions_count,
            "accuracy_percent": round(accuracy, 2),
            "latency_avg": round(avg_lat, 4),
            "latency_p99": round(p99, 4)
        }
        log_to_file(LOG_FILE, final_summary)
    else:
        print("\nNo questions were evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoCoMo Benchmark")
    parser.add_argument("--test", type=int, help="Run a specific story number (1-10)", default=None)
    args = parser.parse_args()

    if not os.path.exists("locomo10.json"):
        print("Error: 'locomo10.json' not found.")
    else:
        run_evaluation("locomo10.json", test_index=args.test)