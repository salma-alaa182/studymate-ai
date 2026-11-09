from src.vectorstore import load_vectorstore

vs = load_vectorstore()
query = input("Question: ")
for doc, score in vs.similarity_search_with_score(query, k=4):
    print(f"\n[distance={score:.3f}] {doc.metadata['source']} (page {doc.metadata['page']})")
    print(doc.page_content[:300])