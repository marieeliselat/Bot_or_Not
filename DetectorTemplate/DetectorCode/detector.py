from abc_classes import ADetector
from teams_classes import DetectionMark
from textblob import TextBlob
import numpy as np
import re
from datetime import datetime
import concurrent.futures
import openai
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import random

class Detector(ADetector):
    def __init__(self):
        self.api_key = "sk-svcacct-hLryFGI6k7id2c8JNbQfOIBQku4N4FaI4ZNudgEHicS4kO5HhTcm5HKt3cPxi4RWKggfT3BlbkFJse802MFwaCvHoH_nSgEZPX-f2sQIfNh6e71_LtPsxnN5uE8milyoxoJBpVBrpioInWwA"
        openai.api_key = self.api_key

    def detect_bot(self, session_data):
        accounts = []
        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
        invalid_location_patterns = ["she/her", "he/him", "they/them", "planet", "moon", "universe"]

        stop_words = ["the", "it", "by", "of", "that", "is", "with", "this", "for", "a", "to", "at", "and", "in", "on", "an"]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words)
        clf = IsolationForest(contamination=0.1)

        all_posts = [post['text'] for post in session_data.posts]
        tfidf_matrix = vectorizer.fit_transform(all_posts)
        
        user_features = np.array([
            [
                user.get('tweet_count', 0),
                user.get('z_score', 0)
            ] for user in session_data.users
        ])

        # Fit the IsolationForest on the collected features
        clf.fit(user_features)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_user, user, session_data, vectorizer, clf, emoji_pattern, invalid_location_patterns)
                for user in session_data.users
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    accounts.append(result)
                    username = next((u["username"] for u in session_data.users if u["id"] == result.user_id), "Unknown")
                    #print(f"Username: {username}, Is Bot: {result.bot}, Confidence: {result.confidence}")

        # print(f"Total users: {len(session_data.users)}")
        # print(f"Total tweets: {len(session_data.posts)}")

        return accounts

    def process_user(self, user, session_data, vectorizer, clf, emoji_pattern, invalid_location_patterns):
        try:
            
           # Initialize heuristic score
            score = 50  # Default starting score
            
            # Collect user's posts
            user_posts = [post['text'] for post in session_data.posts if post['author_id'] == user['id']]
            
            if user_posts:
                # Average word count
                avg_word_count = sum(len(post.split()) for post in user_posts) / len(user_posts)
                
                # Vocabulary Diversity
                unique_words = set(word for post in user_posts for word in post.split())
                total_words = sum(len(post.split()) for post in user_posts)
                vocab_diversity = len(unique_words) / total_words if total_words > 0 else 0
                if vocab_diversity < 0.1:
                    score += 20
                elif vocab_diversity < 0.3:
                    score += 10
                else:
                    score -= 10
                
                # Sentiment Consistency
                sentiments = [TextBlob(post).sentiment.polarity for post in user_posts]
                sentiment_variance = np.var(sentiments)
                if sentiment_variance < 0.05:
                    score += 15
                elif sentiment_variance > 0.2:
                    score -= 10
                
                # Keyword Density using TF-IDF
                user_tfidf_vector = vectorizer.transform(user_posts)
                avg_tfidf_score = user_tfidf_vector.mean()
                score += 20 if avg_tfidf_score > 0.5 else -10
                
                # Emoji Analysis
                emoji_count = sum(len(emoji_pattern.findall(post)) for post in user_posts)
                avg_emoji_usage = emoji_count / len(user_posts)
                score += 10 if avg_emoji_usage > 3 else (-5 if avg_emoji_usage < 1 else 0)
                
                # Temporal Posting Pattern Analysis
                posting_times = [
                    datetime.strptime(post['created_at'], "%Y-%m-%dT%H:%M:%S.000Z")
                    for post in session_data.posts if post['author_id'] == user['id']
                ]
                if len(posting_times) > 1:
                    intervals = np.diff(sorted(posting_times)).astype('timedelta64[s]').astype(int)
                    avg_interval = np.mean(intervals) if intervals.size > 0 else 0
                    if avg_interval < 300:
                        score += 20
                    elif avg_interval > 300:
                        score -= 15
                else:
                    avg_interval = 0
            else:
                # User has no posts
                avg_word_count = 0
                vocab_diversity = 0
                sentiment_variance = 0
                avg_tfidf_score = 0
                avg_emoji_usage = 0
                avg_interval = 0
                score -= 10  # Penalize for no posts
            
            # Location Validity Check
            location = (user.get('location') or '').lower()
            if any(loc in location for loc in invalid_location_patterns):
                score += 10
            else:
                score -= 10

            
            # Anomaly Detection with Isolation Forest
            data_point = np.array([[
                user.get('tweet_count', 0),
                user.get('z_score', 0)
            ]])
            outlier = clf.predict(data_point)
            score += 30 if outlier[0] == -1 else -10
            
            # Normalize heuristic confidence
            score = max(20, score)  # Set minimum score to 20
            heuristic_bot_confidence = max(0, min(score, 100))
            heuristic_not_bot_confidence = 100 - heuristic_bot_confidence

            # Decide whether to call GPT
            if heuristic_bot_confidence >= 90:
                is_bot = True
                final_confidence = heuristic_bot_confidence
            else:
                # Proceed to call GPT
                sampled_posts = random.sample(user_posts, min(len(user_posts), 10)) if user_posts else []
                gpt_response = self.query_gpt_analysis(sampled_posts, heuristic_bot_confidence)
                gpt_is_bot_confidence = gpt_response["is_bot_true"]
                gpt_not_bot_confidence = gpt_response["is_bot_false"]

                # Average the confidences
                is_bot_true_avg = (heuristic_bot_confidence + gpt_is_bot_confidence) / 2
                is_bot_false_avg = (heuristic_not_bot_confidence + gpt_not_bot_confidence) / 2

                # Final decision
                is_bot = is_bot_true_avg > is_bot_false_avg
                final_confidence = max(is_bot_true_avg, is_bot_false_avg)

            # Constrain final confidence
            final_confidence = max(10, min(final_confidence, 100))  # Set minimum confidence to 10

            return DetectionMark(
                user_id=user['id'],
                confidence=int(final_confidence),
                bot=is_bot
            )
        except Exception as e:
            #print(f"Error processing user {user.get('username', 'N/A')}: {e}")
            return DetectionMark(user_id=user['id'], confidence=10, bot=False)

    def query_gpt_analysis(self, posts, heuristic_score):
        try:
            user_data = {
                "heuristic_score": heuristic_score,
                "posts": [{"text": post} for post in posts],
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that detects bot-like behavior in social media accounts. "
                        "Analyze the provided user data, which includes an initial heuristic suspicion score (between 0 and 100) and up to 10 recent posts. "
                        "Determine the likelihood that the user is a bot or not. "
                        "Provide two confidence scores between 0 and 100: 'is_bot_true' and 'is_bot_false'. "
                        "These scores are independent and do not need to sum to 100. "
                        "Respond **only** with a JSON object structured exactly as follows: "
                        '{"is_bot_true": <confidence>, "is_bot_false": <confidence>}'

                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Here is the user data:\n\n"
                        f"{json.dumps(user_data)}\n\n"
                        "Please provide your classification and confidence as JSON."
                    )
                }
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
                temperature=0
            )
            analysis_text = response.choices[0].message["content"].strip()
            #print(f"GPT response: {analysis_text}")
            result = json.loads(analysis_text)
            # After parsing result
            is_bot_true = float(result.get("is_bot_true", 50))
            is_bot_false = float(result.get("is_bot_false", 50))
            # Ensure values are between 0 and 1
            if not (0 <= is_bot_true <= 100 and 0 <= is_bot_false <= 100):
                raise ValueError("Confidence scores out of expected range.")
            return {"is_bot_true": is_bot_true, "is_bot_false": is_bot_false}
        except Exception as e:
            #print(f"Error parsing GPT response: {e}")
            return {"is_bot_true": 50, "is_bot_false": 50}

            
