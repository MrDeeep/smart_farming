import google.generativeai as genai
import config # Imports your key

genai.configure(api_key=config.apikey)

print("Searching for available models...")
for m in genai.list_models():
    # Only show models that can generate content (text/images)
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)