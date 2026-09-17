from models.llm import GeminiLLM


def main():
    print("DocDev AI - Gemini LLM Test")
    

    print("\n[1] Creating Gemini client...")

    llm = GeminiLLM()

    print("[OK] Gemini client created.")

    print("\n[2] Sending test request...")

    prompt = """
Answer this question in one short sentence:

What is artificial intelligence?
"""

    response = llm.generate(
        prompt=prompt,
    )

    if not response:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    print("[OK] Gemini response received.")

    print("\n--- GEMINI RESPONSE ---")
    print(response)

   
    print("GEMINI LLM TEST PASSED!")



if __name__ == "__main__":
    main()