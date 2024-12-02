from abc_classes import ADetector
from teams_classes import DetectionMark
from collections import Counter
import re
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

class Detector(ADetector):
    def detect_bot(self, session_data):
        accounts = []
        # Define stop words as a list instead of a set
        stop_words = ["the", "it", "by", "of", "that", "is", "with", "this", "for", "a", "to", "at", "and", "in", "on", "an"]

        # Initialize the vectorizer with stop words as a list
        vectorizer = TfidfVectorizer(stop_words=stop_words)

        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
        invalid_location_patterns = ["she/her", "he/him", "they/them", "planet", "moon", "universe"]

        # Machine Learning Model for Anomaly Detection
        clf = IsolationForest(contamination=0.1)

        # Preprocessing tweets to get TF-IDF features for keyword density analysis
        all_posts = [post['text'] for post in session_data.posts]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(all_posts)

        for user in session_data.users:
            score = 50  # Starting confidence level, adjustable based on bot-like behavior
            
            # Text Analysis for Consistency, Keyword Density, and Sentiment
            user_posts = [post['text'] for post in session_data.posts if post['author_id'] == user['id']]
            user_word_count = sum(len(post.split()) for post in user_posts)
            avg_word_count = user_word_count / len(user_posts) if user_posts else 0

            # Vocabulary Diversity (unique words per total words)
            unique_words = set(word for post in user_posts for word in post.split())
            vocab_diversity = len(unique_words) / user_word_count if user_word_count else 0
            if vocab_diversity < 0.1:
                score += 20
            elif vocab_diversity < 0.3:
                score += 10
            else:
                score -= 10  # Indicates human-like behavior

            # Sentiment Consistency
            sentiments = [TextBlob(post).sentiment.polarity for post in user_posts]
            sentiment_variance = np.var(sentiments)
            if sentiment_variance < 0.05:
                score += 15
            elif sentiment_variance > 0.2:
                score -= 10  # Varied sentiment is more human-like

            # Keyword Density using TF-IDF
            user_tfidf_vector = vectorizer.transform(user_posts)
            avg_tfidf_score = user_tfidf_vector.mean()
            if avg_tfidf_score > 0.5:
                score += 20
            else:
                score -= 10  # Less repetitive keyword use suggests human behavior

            # Emoji Analysis
            emoji_count = sum(len(emoji_pattern.findall(post)) for post in user_posts)
            avg_emoji_usage = emoji_count / len(user_posts) if user_posts else 0
            if avg_emoji_usage > 3:
                score += 10
            elif avg_emoji_usage < 1:
                score -= 5  # Limited emoji use is more human-like

            # Temporal Posting Pattern Analysis
            posting_times = [datetime.strptime(post['created_at'], "%Y-%m-%dT%H:%M:%S.000Z") for post in session_data.posts if post['author_id'] == user['id']]
            if len(posting_times) > 1:
                intervals = np.diff(sorted(posting_times)).astype('timedelta64[s]').astype(int)
                avg_interval = np.mean(intervals) if intervals.size > 0 else 0
                if avg_interval < 300:
                    score += 20
                if any(interval < 10 for interval in intervals):
                    score += 25
                else:
                    score -= 15  # Longer intervals imply non-bot behavior

            # Location Validity Check
            location = user.get('location', '').lower() if user.get('location') else ''
            if any(loc in location for loc in invalid_location_patterns) or emoji_pattern.search(location):
                score += 10
            else:
                score -= 10  # Valid location is more human-like

            # Abnormal Account Behavior with Isolation Forest
            tweet_count = user.get('tweet_count', 0)
            following = user.get('following_count', 0)
            followers = user.get('followers_count', 0)
            data_point = np.array([[tweet_count, following, followers]])
            outlier = clf.fit_predict(data_point)  # -1 indicates outlier
            if outlier[0] == -1:
                score += 30
            else:
                score -= 10  # Typical user behavior

            # Normalize score to be between 1 and 100
            final_confidence = max(1, min(score, 100))

            detection = DetectionMark(
                user_id=user['id'],
                confidence=final_confidence,
                bot=final_confidence > 60  # Threshold for bot detection
            )
            accounts.append(detection)

        return accounts
