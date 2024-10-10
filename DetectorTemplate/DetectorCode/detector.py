from abc_classes import ADetector
from teams_classes import DetectionMark
from collections import Counter
import re

class Detector(ADetector):
    def detect_bot(self, session_data):
        accounts = []
        # stop words = common words to ignore
        stop_words = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "at", "by", "with", "an", "this", "that", "it"}

        # grab text from posts and split into words (ignoring stop words)
        word_counter = Counter()
        for post in session_data.posts:
            words = re.findall(r'\b\w+\b', post['text'].lower())
            filtered_words = [word for word in words if word not in stop_words]
            word_counter.update(filtered_words)

        # find the most common word
        most_common_word, _ = word_counter.most_common(1)[0]
        #print(f"Most Common Word: '{most_common_word}'")

        # iterate through the users dataset (list of dict)
        for user in session_data.users:
            # basic detection logic
            is_bot = False
            # default confidence level for detected bots
            confidence = 50  

            # Rule 1: If "bot" is in the username or description, classify as a bot
            if "bot" in user.get('username', '').lower() or "bot" in user.get('description', '').lower():
                is_bot = True

            # Rule 2: Check if the user uses the most common word in their posts
            user_posts = [post['text'] for post in session_data.posts if post['author_id'] == user['id']]
            user_word_usage = any(most_common_word in post.lower() for post in user_posts)
            if not user_word_usage:
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 3: Check if the location is unusual or empty
            location = user.get('location', '')
            if location is None or location.lower() in ["", "unknown", "nowhere", "planet x"]:
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 4: Check tweet count (suspicious if very low or very high)
            tweet_count = user.get('tweet_count', 0)
            if tweet_count < 5 or tweet_count > 1000:
                is_bot = True
                confidence = max(confidence, 100)

            # Rule 5: Analyze z_score for extreme values
            z_score = user.get('z_score', 0)
            if z_score < -2 or z_score > 2:
                is_bot = True
                confidence = max(confidence, 100)

            # Create a DetectionMark object for each user
            detection = DetectionMark(
                user_id=user['id'],
                confidence=confidence if is_bot else 1,
                bot=is_bot
            )
            accounts.append(detection)
        
        # Print out the results in the desired format
        # print("\nDetected Users and Their Bot Status:")
        # for detection in accounts:
        #     print(f"user_id='{detection.user_id}', confidence={detection.confidence}, bot={detection.bot}")

        return accounts
