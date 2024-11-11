from abc_classes import ADetector
from teams_classes import DetectionMark
from collections import Counter
import re
from datetime import datetime
from textblob import TextBlob

class Detector(ADetector):
    def detect_bot(self, session_data):
        accounts = []
        # stop words = common words to ignore
        stop_words = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "at", "by", "with", "an", "this", "that", "it"}

        # print(session_data.users)
        # print('/n')

        # grab text from posts and split into words (ignoring stop words)
        word_counter = Counter()
        for post in session_data.posts:
            words = re.findall(r'\b\w+\b', post['text'].lower())
            filtered_words = [word for word in words if word not in stop_words]
            word_counter.update(filtered_words)

        # find the most common word
        most_common_word, _ = word_counter.most_common(1)[0] if word_counter else ("", 0)

        # grab emojis
        emoji_pattern = re.compile("[\U0001F600-\U0001F64F"  
                                   "\U0001F300-\U0001F5FF"
                                   "\U0001F680-\U0001F6FF"
                                   "\U0001F1E0-\U0001F1FF"
                                   "\U00002702-\U000027B0"
                                   "\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        # invalid locations
        invalid_location_patterns = [
            "she/her", "he/him", "they/them", "pronouns", "blk", "rainbow", 
            "planet", "mars", "moon", "space", "anywhere", "nowhere", 
            "everywhere", "earth", "universe"
        ]
        
        # iterate through the users dataset (list of dict)
        for user in session_data.users:
            is_bot = False
            confidence = 50  # default confidence

            # Rule 1: Check if the username or description contains "bot"
            if "bot" in user.get('username', '').lower() or "bot" in user.get('description', '').lower():
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 2: Check if the user uses the most common word in their posts
            user_posts = [post['text'] for post in session_data.posts if post['author_id'] == user['id']]
            user_word_usage = any(most_common_word in post.lower() for post in user_posts)
            if not user_word_usage:
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 3: Check if the location is suspicious or empty
            location = user.get('location', '')  
            if location:  # Check if location is not None
                location = location.lower()  # Normalize case
                
                # Check if the location contains any invalid patterns
                for pattern in invalid_location_patterns:
                    if pattern in location:
                        is_bot = True
                        confidence = max(confidence, 100)
                        break  # No need to check further patterns if one matches
                
                # Check for emojis in the location
                if emoji_pattern.search(location):
                    is_bot = True
                    confidence = max(confidence, 100)

            # Rule 4: Analyze tweet count
            tweet_count = user.get('tweet_count', 0)
            if 100 <= tweet_count <= 1000:
                is_bot = True
                confidence = max(confidence, 100)
            # future = make it based on time that went by 

            # Rule 5: Analyze z_score for abnormal values
            z_score = user.get('z_score', 0)
            if z_score < -2 or z_score > 2:
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 6: Check language usage of posts
            user_languages = set(post['lang'] for post in session_data.posts if post['author_id'] == user['id'])
            if len(user_languages) > 2:
                is_bot = True
                confidence = max(confidence, 100)
            for lang in user_languages:
                if emoji_pattern.search(lang):
                    is_bot = True
                    confidence = max(confidence, 100)


            # Rule 7: Posting times (e.g., very frequent, non-human intervals)
            posting_times = [post['created_at'] for post in session_data.posts if post['author_id'] == user['id']]
            if len(posting_times) > 1:
                # check the time difference between consecutive posts
                time_differences = [
                    (datetime.strptime(posting_times[i], "%Y-%m-%dT%H:%M:%S.000Z") - datetime.strptime(posting_times[i-1], "%Y-%m-%dT%H:%M:%S.000Z")).total_seconds()
                    for i in range(1, len(posting_times))
                ]
                # if posts are too close together, flag as suspicious
                if any(diff < 10 for diff in time_differences):  # less than 10 seconds between posts
                    is_bot = True
                    confidence = max(confidence, 95)

            # Final classification
            detection = DetectionMark(
                user_id=user['id'],
                confidence=confidence if is_bot else 1,
                bot=is_bot
            )
            accounts.append(detection)

        # print("\nDetected Users and Their Bot Status:")
        # for detection in accounts:
        #     print(f"user_id='{detection.user_id}', confidence={detection.confidence}, bot={detection.bot}")
            

        return accounts
