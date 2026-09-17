from google import genai
from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found.")

    client = genai.Client(api_key=api_key)

    models = [
        "gemini-3.5-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
    ]

    print("DocDev AI - Gemini Model Test")
    print("=" * 70)

    for model in models:
        print(f"\nTesting: {model}")
        print("-" * 70)

        try:
            response = client.models.generate_content(
                model=model,
                contents="Explain retrieval augmented generation in one simple sentence.",
            )

            print("[OK] Response received")
            print(response.text)

        except Exception as exc:
            print("[FAILED]")
            print(exc)

   
    print("MODEL TEST COMPLETE")
  


if __name__ == "__main__":
    main()