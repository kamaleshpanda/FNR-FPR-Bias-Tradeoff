import sqlite3
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/bert_mitigated_final")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../instance/sentinel_audit.db")

print("Loading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully")

# sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

identity_words = [
"muslim","christian","jew","hindu",
"black","white","asian",
"gay","lesbian","trans",
"woman","man","female","male"
]

def analyze_text(text):
    identity_mention = any(word in text.lower() for word in identity_words)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    toxicity_score = float(probs[1])

    if identity_mention and toxicity_score < 0.85:
        toxicity_score *= 0.6
        
    toxicity = "toxic" if toxicity_score > 0.5 else "safe"

    sentiment_result = sentiment_pipeline(text[:512])[0]
    sentiment = sentiment_result["label"]
    sentiment_score = float(sentiment_result["score"])

    #-----
    biased_words = ["hate","kill","stupid","dirty","idiot"]
    bias_score = 0.2
    bias = "neutral"

    if any(word in text.lower() for word in biased_words):
        bias_score = 0.7
        bias = "biased"

    

    risk_score = (toxicity_score + bias_score) / 2

    if risk_score < 0.25:
        risk = "LOW"
    elif risk_score < 0.5:
        risk = "MEDIUM"
    elif risk_score < 0.75:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "toxicity": toxicity,
        "toxicity_score": toxicity_score,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "bias": bias,
        "bias_score": bias_score,
        "overall_risk": risk,
        "risk_score": risk_score,
        "keywords": [],
        "highlighted_text": text,
        "identity_mention": identity_mention
    }
# ------------------------------
# Initialize database
# ------------------------------
def init_db():

    os.makedirs("instance", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            toxicity TEXT,
            sentiment TEXT,
            overall_risk TEXT,
            identity_mention INTEGER,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# ------------------------------
# Insert analysis result
# ------------------------------
def insert_analysis(text, result):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO history
        (text, toxicity, sentiment, overall_risk, identity_mention, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        text,
        result["toxicity"],
        result["sentiment"],
        result["overall_risk"],
        int(result["identity_mention"]),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ------------------------------
# Get history (for dashboard)
# ------------------------------
def get_history(limit=15, offset=0):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT id,text,toxicity,sentiment,overall_risk,identity_mention,created_at
        FROM history
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cur.fetchall()

    conn.close()

    items = []

    for r in rows:
        items.append({
            "id": r[0],
            "text": r[1],
            "toxicity": r[2],
            "sentiment": r[3],
            "overall_risk": r[4],
            "identity_mention": bool(r[5]),
            "created_at": r[6]
        })

    return {
        "total": total,
        "items": items
    }


# ------------------------------
# Get stats (for dashboard cards)
# ------------------------------
def get_stats():

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE toxicity='toxic'")
    toxic_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE sentiment='POSITIVE'")
    positive_count = cur.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "positive_count": positive_count,
        "negative_count": total - positive_count,
        "toxic_count": toxic_count,
        "avg_sentiment_score": 0.5,
        "avg_toxicity_score": 0.5
    }
