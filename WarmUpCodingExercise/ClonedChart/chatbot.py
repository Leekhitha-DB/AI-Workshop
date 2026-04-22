
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError('GEMINI_API_KEY not found.')

client = genai.Client(api_key=API_KEY)

# Use a Gemma model (from your list)
MODEL_NAME = 'models/gemma-3-4b-it'   # or gemma-3-12b-it, etc.

# System prompt will be injected as a fake user message at start
SYSTEM_PROMPT = '''You are a helpful, friendly, and concise AI assistant.
You answer questions clearly and ask for clarification when needed.
Keep responses under 200 words unless a longer answer is necessary.'''

history = []

# Initialize history with system prompt (as a user message)
history.append(types.Content(
    role="user",
    parts=[types.Part.from_text(text=f"System: {SYSTEM_PROMPT}\n\nPlease follow these instructions.")]
))
# Add a dummy assistant acknowledgment (optional)
history.append(types.Content(
    role="model",
    parts=[types.Part.from_text(text="Understood. I will follow those guidelines.")]
))

def chat(user_input: str) -> str:
    global history

    # Add real user message
    history.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_input)]
    ))

    # Generate response (no system_instruction in config)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=history,
        config=types.GenerateContentConfig(temperature=0.7)  # no system_instruction
    )

    reply = response.text

    # Add assistant response
    history.append(types.Content(
        role="model",
        parts=[types.Part.from_text(text=reply)]
    ))

    return reply

def main():
    print('=' * 50)
    print(' Gemini AI Chatbot (Gemma) | Type exit to quit')
    print('=' * 50)
    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit', 'bye'):
            print('Bot: Goodbye! Have a great day.')
            break
        try:
            reply = chat(user_input)
            print(f'Bot: {reply}\n')
        except Exception as e:
            print(f'Bot: [Error] {e}\n')

if __name__ == '__main__':
    main()