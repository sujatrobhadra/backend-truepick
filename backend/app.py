from flask import Flask, jsonify, request
from flask_cors import CORS

import os
import re
import math
import json
import requests

from bs4 import BeautifulSoup
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# APP SETUP
# =========================================================

app = Flask(__name__)
CORS(app)

# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
JSON_PATH = os.path.join(PUBLIC_DIR, "weekly_deals_analysis.json")

os.makedirs(PUBLIC_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# =========================================================
# HEADERS
# =========================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}

# =========================================================
# SENTIMENT ENGINE
# =========================================================

SCAM_KEYWORDS = {
    "scam": 1.0,
    "fake": 0.8,
    "fraud": 1.0,
    "never received": 0.9,
    "do not buy": 0.9,
    "waste of money": 0.7,
    "broken": 0.6,
    "cheap quality": 0.5,
    "cheap copy": 0.7,
    "not same as shown": 0.8,
    "no refund": 0.9,
    "customer service bad": 0.5
}

IGNORE_PATTERNS = ["link pls", "where did you buy", "@"]

analyzer = SentimentIntensityAnalyzer()

# =========================================================
# HELPERS
# =========================================================

def clean_comment(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_comment(comment):
    if not comment:
        return False
    for p in IGNORE_PATTERNS:
        if p in comment:
            return False
    return True


def sentiment_score(comment):
    return analyzer.polarity_scores(comment)["compound"]


def credibility_score(user):
    return round(
        0.35 * min(user.get("account_age_days", 0) / 365, 1) +
        0.30 * min(user.get("engagement", 0) / 100, 1) +
        0.20 * min(user.get("karma", 0) / 10000, 1) +
        0.15 * (1 if user.get("verified") else 0),
        3
    )


def volume_confidence(n):
    return round(min(math.log(n + 1, 10), 1), 3)


def red_flag_penalty(comments):
    total = len(comments)
    if total == 0:
        return 0

    penalty = 0
    for c in comments:
        c = c.lower()
        for k in SCAM_KEYWORDS:
            if k in c:
                penalty += SCAM_KEYWORDS[k]
                break

    return round(min((penalty / total) ** 0.7 * 1.5, 1), 3)


def recency_factor(date):
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
        return round(math.exp(-(datetime.now() - d).days / 180), 3)
    except:
        return 0.5


# =========================================================
# SYNTHETIC AMAZON RATING SUPPORT
# =========================================================

def synthetic_rating_comment(rating):
    if not rating:
        return "neutral product review"

    try:
        stars = float(rating.split()[0])

        if stars >= 4.5:
            return "excellent product highly recommended"
        elif stars >= 4:
            return "good product positive experience"
        elif stars >= 3:
            return "average product mixed experience"
        elif stars >= 2:
            return "bad product negative experience"
        else:
            return "terrible product avoid buying"

    except:
        return "neutral product review"


# =========================================================
# TRUST ENGINE (UPGRADED)
# =========================================================

def calculate_trust_score(comments):

    valid = []
    raw = []
    weighted = []
    total_w = 0
    C_list = []
    R_list = []
    verified = 0
    rating_bonus = 0
    amazon_review_count = 0

    for item in comments:

        text_input = item.get("comment")

        if not text_input:
            text_input = synthetic_rating_comment(item.get("rating", ""))

        text = clean_comment(text_input)

        if not is_valid_comment(text):
            continue

        valid.append(text)

        engagement = max(item["user"].get("engagement", 1), 1)
        weight = math.log(engagement + 1)
        total_w += weight

        s = sentiment_score(text)
        weighted.append(s * weight)
        raw.append(s)

        C_list.append(credibility_score(item["user"]))
        R_list.append(recency_factor(item["date"]))

        if item["user"].get("verified"):
            verified += 1

        if "rating" in item:
            amazon_review_count += 1
            try:
                rating_bonus += float(item["rating"].split()[0]) / 5
            except:
                pass

    n = len(valid)

    if n == 0:
        return {
            "trust_score": 0,
            "status": "REJECTED"
        }

    S = sum(weighted) / (total_w + 1e-6)
    C = sum(C_list) / n
    R = sum(R_list) / n
    V = volume_confidence(n)
    F = red_flag_penalty(valid)
    verified_ratio = verified / n

    avg = sum(raw) / len(raw)
    sentiment_variance = sum((x - avg) ** 2 for x in raw) / len(raw)
    stability_bonus = max(0, 1 - sentiment_variance)

    amazon_bonus = 0
    if amazon_review_count > 0:
        amazon_bonus = (rating_bonus / amazon_review_count) * 0.10

    trust = (
        0.30 * S +
        0.25 * C +
        0.15 * R +
        0.10 * V -
        0.15 * F +
        0.05 * stability_bonus +
        0.10 * verified_ratio +
        amazon_bonus
    )

    trust = round(max(min(trust, 1), 0), 3)

    status = "APPROVED" if trust > 0.70 and F < 0.25 else "REJECTED"

    confidence_interval = round(1 / math.sqrt(n), 3)

    return {
        "trust_score": trust,
        "status": status,
        "confidence_interval": confidence_interval,
        "sentiment_score": round(S, 3),
        "credibility_score": round(C, 3),
        "recency_score": round(R, 3),
        "volume_confidence": round(V, 3),
        "red_flag_penalty": round(F, 3),
        "verified_ratio": round(verified_ratio, 3),
        "stability_bonus": round(stability_bonus, 3),
        "sentiment_variance": round(sentiment_variance, 3),
        "comment_count": n
    }


# =========================================================
# AMAZON SCRAPER
# =========================================================

def scrape_amazon(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")

        title = soup.select_one("#productTitle")
        img = soup.select_one("#landingImage")
        review_blocks = soup.select("[data-hook='review-body']")

        comments = []

        for r in review_blocks[:10]:
            text = r.get_text(strip=True)
            if text:
                comments.append({
                    "comment": text,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "user": {
                        "engagement": 10,
                        "karma": 50,
                        "verified": False
                    }
                })

        return {
            "product_title": title.get_text(strip=True) if title else None,
            "product_image": img.get("src") if img else None,
            "amazon_comments": comments
        }

    except:
        return {
            "product_title": None,
            "product_image": None,
            "amazon_comments": []
        }


# =========================================================
# REDDIT SCRAPER (UNCHANGED)
# =========================================================

def scrape_reddit(url):
    try:
        if not url.endswith(".json"):
            url = url.rstrip("/") + ".json"

        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()

        comments = []

        for child in data[1]["data"]["children"]:
            body = child["data"].get("body")

            if body:
                comments.append({
                    "comment": body,
                    "date": datetime.utcfromtimestamp(
                        child["data"]["created_utc"]
                    ).strftime("%Y-%m-%d"),
                    "user": {
                        "engagement": child["data"].get("score", 1),
                        "karma": child["data"].get("ups", 1),
                        "verified": False
                    }
                })

        return comments

    except:
        return []


# =========================================================
# SAVE JSON
# =========================================================

def save_result(data):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing.append(data)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)


# =========================================================
# ANALYZE ROUTE
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    body = request.json

    external_url = body.get("external_url")
    reddit_url = body.get("reddit_url")

    if not external_url:
        return jsonify({"error": "external_url required"}), 400

    amazon = scrape_amazon(external_url)

    comments = []
    comments.extend(amazon.get("amazon_comments", []))

    if reddit_url:
        comments.extend(scrape_reddit(reddit_url))

    result = calculate_trust_score(comments)

    final = {
        "product_name": amazon.get("product_title"),
        "reddit_url": reddit_url,
        "external_url": external_url,
        "product_image_url": amazon.get("product_image"),
        "trust_analysis": result
    }

    save_result(final)

    return jsonify(final)


# =========================================================
# GET DATA
# =========================================================

@app.route("/weekly-deals-analysis", methods=["GET"])
def get_all():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return jsonify({"status": "running"})


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run()