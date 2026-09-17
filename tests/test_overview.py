from analysis.overview import generate_overview


def main():
    print("DocDev AI - Document Overview Test")
   
    file_path = "data/uploads/test.txt"

    print("\n[1] Generating overview...")

    overview = generate_overview(
        file_path
    )

    for key, value in overview.items():
        print(
            f"{key}: {value}"
        )

    print("\n[2] Validating overview...")

    if overview["filename"] != "test.txt":
        raise RuntimeError(
            "Incorrect filename."
        )

    if overview["file_type"] != ".txt":
        raise RuntimeError(
            "Incorrect file type."
        )

    if overview["words"] <= 0:
        raise RuntimeError(
            "Word count should be greater than zero."
        )

    if overview["characters"] <= 0:
        raise RuntimeError(
            "Character count should be greater than zero."
        )

    if overview["empty"]:
        raise RuntimeError(
            "Document should not be empty."
        )

    print("[OK] Overview is valid.")

   
    print("DOCUMENT OVERVIEW TEST PASSED!")
   


if __name__ == "__main__":
    main()