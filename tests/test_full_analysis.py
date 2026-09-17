from analysis.analyzer import DocumentAnalyzer


def main():

    print("DocDev AI - Full Document Analysis Test")
    
    file_path = "data/uploads/test.txt"

    analyzer = DocumentAnalyzer()

    print("\n[1] SUMMARY")
   
    print(analyzer.summary(file_path))

    print("\n[2] IMPORTANT POINTS")
   
    print(analyzer.important_points(file_path))

    print("\n[3] KNOWLEDGE")
   
    print(analyzer.knowledge(file_path))

    print("\n[4] TOPICS")
    print("-" * 60)
    print(analyzer.topics(file_path))

    print("\n[5] TERMS")
    print("-" * 60)
    print(analyzer.terms(file_path))

    print("\n[6] ENTITIES")
    print("-" * 60)
    print(analyzer.entities(file_path))

    print("\n[7] QUESTIONS")
    print("-" * 60)
    print(analyzer.questions(file_path))

    print("\n[8] LINKS")
    print("-" * 60)

    links = analyzer.links(file_path)

    if links:
        for link in links:
            print(link)
    else:
        print("No links found.")

   
    print("FULL DOCUMENT ANALYSIS TEST PASSED!")
   


if __name__ == "__main__":
    main()