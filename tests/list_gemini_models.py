from google import genai
import os
from dotenv import load_dotenv


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found.")

    client = genai.Client(api_key=api_key)

    print("Available Gemini models")
   

    count = 0

    for model in client.models.list():
        name = model.name or ""

        if "gemini" in name.lower():
            print(name)
            count += 1

  
    print(f"Found {count} Gemini model(s).")


if __name__ == "__main__":
    main()