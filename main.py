from tools.search import search  # change this to the actual filename where the `search` tool is defined

def main():
    # Hardcoded input
    query = "absorption"
    file_path = r"C:\Users\lenovo\Desktop\DocuMentor\test.pdf"  # Use raw string to avoid path issues

    print(f"🔍 Query: {query}")
    print(f"📁 File Path: {file_path}\n")

    try:
        result = search.invoke({
            "query": query,
            "file_path": file_path  # ⚠️ snake_case
        })
        print("✅ Result:\n")
        print(result)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()