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

        # Step 1: Heuristic analysis
        heuristic_results = {}
        for user in session_data.users:
            user_id_str = str(user['id'])
            if user_id_str in user_ids_processed:
                continue

            is_bot = False
            confidence = 50

            # Rule 2: Check repetitive tweets
            posts = user_posts.get(user_id_str, [])
            if len(posts) > 1:
                similarity_threshold = 0.8
                similar_pairs_count = 0
                total_pairs = 0
                for i in range(len(posts)):
                    for j in range(i + 1, len(posts)):
                        total_pairs += 1
                        similarity_ratio = SequenceMatcher(None, posts[i], posts[j]).ratio()
                        if similarity_ratio > similarity_threshold:
                            similar_pairs_count += 1
                if total_pairs > 0 and (similar_pairs_count / total_pairs) > 0.5:
                    is_bot = True
                    confidence = max(confidence, 100)

            # Rule 3: Check location
            location = user.get('location', '')
            if location:
                location = location.lower()
                invalid_location_patterns = [
                    "she/her", "he/him", "they/them", "pronouns", "blk", "rainbow",
                    "planet", "mars", "moon", "space", "anywhere", "nowhere",
                    "everywhere", "earth", "universe"
                ]
                if any(pattern in location for pattern in invalid_location_patterns):
                    confidence += 10
            else:
                confidence += 5

            # Rule 4: Tweet count
            tweet_count = user.get('tweet_count', 0)
            if 100 <= tweet_count <= 1000:
                confidence += 5

            # Rule 5: Z-Score
            z_score = user.get('z_score', 0)
            if z_score < -2 or z_score > 2:
                confidence += 10

            # Rule 6: Language diversity
            user_languages = set(post['lang'] for post in session_data.posts if str(post['author_id']) == user_id_str)
            if len(user_languages) > 2:
                confidence += 10

            # Rule 7: Posting frequency
            posting_times = [
                post['created_at'] for post in session_data.posts if str(post['author_id']) == user_id_str
            ]
            if len(posting_times) > 1:
                time_differences = [
                    (
                        datetime.strptime(posting_times[i], "%Y-%m-%dT%H:%M:%S.000Z") -
                        datetime.strptime(posting_times[i - 1], "%Y-%m-%dT%H:%M:%S.000Z")
                    ).total_seconds()
                    for i in range(1, len(posting_times))
                ]
                if any(diff < 10 for diff in time_differences):
                    confidence += 10

            confidence = max(0, min(confidence, 100))
            is_bot = confidence >= 70

            heuristic_results[user_id_str] = {'is_bot': is_bot, 'confidence': confidence}
            user_ids_processed.add(user_id_str)

        # Step 2: Analyze users with ChatGPT
        chatgpt_analysis = self.analyze_users_with_chatgpt(session_data.users, user_posts)
        chatgpt_results = self.parse_chatgpt_response(chatgpt_analysis)

        # Step 3: Combine Heuristics and ChatGPT results
        for user_id, chatgpt_result in chatgpt_results.items():
            chatgpt_is_bot = chatgpt_result['is_bot']
            chatgpt_confidence = chatgpt_result['confidence']

            heuristic_result = heuristic_results.get(user_id, {'is_bot': False, 'confidence': 50})
            heuristic_is_bot = heuristic_result['is_bot']
            heuristic_confidence = heuristic_result['confidence']

            if chatgpt_is_bot == heuristic_is_bot:
                final_confidence = (chatgpt_confidence + heuristic_confidence) / 2
                final_is_bot = chatgpt_is_bot
            else:
                if chatgpt_confidence > heuristic_confidence:
                    final_is_bot = chatgpt_is_bot
                    final_confidence = chatgpt_confidence
                else:
                    final_is_bot = heuristic_is_bot
                    final_confidence = heuristic_confidence

            detection = DetectionMark(
                user_id=user_id,
                confidence=final_confidence,
                bot=final_is_bot
            )
            accounts.append(detection)

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
