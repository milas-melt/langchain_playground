import os
from dotenv import load_dotenv

load_dotenv('.env.local')


# OPENAI
openai_api_key = os.environ.get("OPENAI_API_KEY")

if openai_api_key is None:
    print("OPENAI_API_KEY not found.")
else:
    print(f"API key: {openai_api_key}")

#  serpapi
serpapi_api_key = os.environ.get("SERPAPI_API_KEY")
if serpapi_api_key is None:
    print("SERPAPI_API_KEY not found.")
else:
    print(f"API key: {serpapi_api_key}")