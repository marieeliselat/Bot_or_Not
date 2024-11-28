# Import necessary libraries
from abc_classes import ADetector
from teams_classes import DetectionMark
from collections import Counter
import re
from datetime import datetime
import openai
import os
from difflib import SequenceMatcher
import json

# Set your OpenAI API key
openai.api_key = 'sk-svcacct-hLryFGI6k7id2c8JNbQfOIBQku4N4FaI4ZNudgEHicS4kO5HhTcm5HKt3cPxi4RWKggfT3BlbkFJse802MFwaCvHoH_nSgEZPX-f2sQIfNh6e71_LtPsxnN5uE8milyoxoJBpVBrpioInWwA'  # Replace with your actual API key

# Define the Detector class
class Detector(ADetector):
    def detect_bot(self, session_data):
        accounts = []
        user_ids_processed = set()  # Keep track of processed user IDs (as strings)

        # Build a mapping from user IDs to user data
        user_id_to_user = {str(user['id']): user for user in session_data.users}

        # Collect posts for each user
        user_posts = {}
        for post in session_data.posts:
            user_id_str = str(post['author_id'])
            if user_id_str not in user_posts:
                user_posts[user_id_str] = []
            user_posts[user_id_str].append(post['text'])

        # Define the emoji pattern
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

        # First pass: Heuristic analysis
        for user in session_data.users:
            user_id_str = str(user['id'])
            if user_id_str in user_ids_processed:
                continue  # Skip if already processed

            is_bot = False
            confidence = 50  # Start with a neutral confidence

            # # Rule 1: Check if the username or description contains "bot"
            # if "bot" in user.get('username', '').lower() or "bot" in user.get('description', '').lower():
            #     confidence += 20  # Increase confidence moderately

            # Rule 2: Check if the user has repetitive tweets with similar content
            posts = user_posts.get(user_id_str, [])
            if len(posts) > 1:  # Ensure there are at least two posts to compare
                similarity_threshold = 0.8  # Threshold for similarity (can be adjusted)
                similar_pairs_count = 0
                total_pairs = 0

                # Compare each tweet with every other tweet
                for i in range(len(posts)):
                    for j in range(i + 1, len(posts)):
                        total_pairs += 1
                        similarity_ratio = SequenceMatcher(None, posts[i], posts[j]).ratio()
                        if similarity_ratio > similarity_threshold:
                            similar_pairs_count += 1

                # If the majority of the tweets are similar, classify as a bot
                if total_pairs > 0 and (similar_pairs_count / total_pairs) > 0.5:  # More than 50% of pairs are similar
                    is_bot = True
                    confidence = max(confidence, 100)

            # Rule 3: Check if the location is suspicious or empty
            location = user.get('location', '')
            if location:
                location = location.lower()
                invalid_location_patterns = [
                    "she/her", "he/him", "they/them", "pronouns", "blk", "rainbow",
                    "planet", "mars", "moon", "space", "anywhere", "nowhere",
                    "everywhere", "earth", "universe"
                ]
                for pattern in invalid_location_patterns:
                    if pattern in location:
                        confidence += 10  # Slight bot indicator
                        break
                if emoji_pattern.search(location):
                    confidence += 10  # Slight bot indicator
            else:
                # Empty location might be slightly suspicious
                confidence += 5

            # Rule 4: Analyze tweet count
            tweet_count = user.get('tweet_count', 0)
            if 100 <= tweet_count <= 1000:
                confidence += 5  # Slight bot indicator

            # Rule 5: Analyze z_score for abnormal values
            z_score = user.get('z_score', 0)
            if z_score < -2 or z_score > 2:
                confidence += 10  # Moderate bot indicator

            # Rule 6: Check language usage of posts
            user_languages = set(post['lang'] for post in session_data.posts if str(post['author_id']) == user_id_str)
            if len(user_languages) > 2:
                confidence += 10  # Moderate bot indicator
            for lang in user_languages:
                if emoji_pattern.search(lang):
                    confidence += 10  # Moderate bot indicator

            # Rule 7: Posting times (e.g., very frequent, non-human intervals)
            posting_times = [
                post['created_at'] for post in session_data.posts if str(post['author_id']) == user_id_str
            ]
            if len(posting_times) > 1:
                # Check the time difference between consecutive posts
                time_differences = [
                    (
                        datetime.strptime(posting_times[i], "%Y-%m-%dT%H:%M:%S.000Z") -
                        datetime.strptime(posting_times[i - 1], "%Y-%m-%dT%H:%M:%S.000Z")
                    ).total_seconds()
                    for i in range(1, len(posting_times))
                ]
                # If posts are too close together, flag as suspicious
                if any(diff < 10 for diff in time_differences):  # Less than 10 seconds between posts
                    confidence += 10  # Moderate bot indicator

            # Ensure confidence is within 0-100%
            confidence = max(0, min(confidence, 100))

            # Determine if the user is a bot based on confidence
            is_bot = confidence >= 70  # You can adjust this threshold as needed

            # Final classification for high confidence users
            detection = DetectionMark(
                user_id=user_id_str,  # User ID as string
                confidence=confidence,
                bot=is_bot
            )
            accounts.append(detection)
            user_ids_processed.add(user_id_str)

        # Print the results for the first few users only
        for detection in accounts[:5]:
            user = user_id_to_user.get(detection.user_id, {})
            print(f"User ID: {detection.user_id}")
            print(f"Username: {user.get('username', 'N/A')}")
            print(f"Confidence: {detection.confidence}%")
            print(f"Classified as bot: {detection.bot}")
            print("-" * 30)

        return accounts

    # Function to analyze a batch of users with ChatGPT
    def analyze_users_with_chatgpt(self, users, user_posts):
        # Prepare the prompt
        prompt = """
You are an expert in detecting social media bots and fake bots.

Analyze the following user profiles and their recent posts to determine the likelihood that each user is a bot.

For each user, provide a JSON object with the following keys:
- "user_id": the user's ID (as a string)
- "classification": "Bot" or "Not Bot"
- "confidence": a percentage between 0 and 100 (integer)

Here are the users:

"""

        for user in users:
            user_id_str = str(user.get('id', 'N/A'))
            username = user.get('username', 'N/A')
            description = user.get('description', 'N/A')
            location = user.get('location', 'N/A')
            tweet_count = user.get('tweet_count', 'N/A')
            z_score = user.get('z_score', 'N/A')
            posts = user_posts.get(user_id_str, [])
            posts = posts[:3]  # Limit number of posts per user
            formatted_posts = "\n".join(f"- {post}" for post in posts)

            user_info = f"""
User ID: {user_id_str}
Username: {username}
Description: {description}
Location: {location}
Tweet Count: {tweet_count}
Z-Score: {z_score}
Recent Posts:
{formatted_posts}

"""
            prompt += user_info

        prompt += """
Please analyze the users and provide the classifications in the following JSON array (ensure it is valid JSON and nothing else):

[
  {
    "user_id": "user_id_here",
    "classification": "Bot",
    "confidence": confidence_percentage
  },
  {
    "user_id": "another_user_id",
    "classification": "Not Bot",
    "confidence": confidence_percentage
  }
  // ... one entry per user
]

Do not include any explanations or additional text. Only provide the JSON array.
"""

        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant for detecting bots on social media platforms.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            max_tokens=1500,  # Adjust based on expected response size
            n=1,
            stop=None,
            temperature=0.3,
        )

        return response['choices'][0]['message']['content']

    # Function to parse ChatGPT's response
    def parse_chatgpt_response(self, response_text):
        # Extract JSON from the response
        try:
            # Use a regular expression to extract the JSON array
            json_match = re.search(r'\[\s*{.*?}\s*\]', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in the response.")

            json_str = json_match.group(0)

            # Clean up the JSON string to ensure it is valid
            json_str = json_str.replace('// ... one entry per user', '')
            json_str = re.sub(r'//.*?\n', '', json_str)  # Remove any comments
            json_str = re.sub(r',\s*]', ']', json_str)   # Remove trailing commas before closing bracket

            data = json.loads(json_str)
            results = {}
            for item in data:
                user_id_str = str(item.get('user_id'))
                classification = item.get('classification', '').lower()
                confidence = item.get('confidence', 50)
                # Ensure confidence is a number and within 0-100
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 50  # Default value if parsing fails
                confidence = min(max(confidence, 0), 100)
                is_bot = True if classification == 'bot' else False
                results[user_id_str] = {
                    'is_bot': is_bot,
                    'confidence': confidence
                }
            return results
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise Exception(f"Error parsing ChatGPT response: {e}")
