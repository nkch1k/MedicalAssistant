"""Debug script to see what chunks are being retrieved."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from part_a.main import RAGApplication
from part_a.config import load_config

# Initialize
config = load_config()
pdf_files = list(config.data_dir.glob("*.pdf"))
document_path = pdf_files[0]

app = RAGApplication()
app.initialize_system(document_path)

# Test questions
questions = [
    "כמה יעלה לי ביטוח טיפולי אקופונקטורה?",
    "ממתי ניתן לקבל החזר על הטיפול באקופונקטורה?",
    "האם מקבלים החזר מלא על הטיפולים?",
    "כמה טיפולים מכוסים?"
]

print("=" * 80)
print("RETRIEVAL ANALYSIS")
print("=" * 80)

for i, question in enumerate(questions, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}: {question}")
    print('='*80)

    # Get retrieved chunks
    chunks = app.rag_chain.retriever.retrieve(
        query=question,
        k=4,
        similarity_threshold=0.0
    )

    print(f"\nRetrieved {len(chunks)} chunks:")

    for j, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {j} (similarity: {chunk.similarity_score:.3f}) ---")
        print(f"Content length: {len(chunk.content)} chars")
        print(f"Content:\n{chunk.content[:500]}...")

        # Check for expected keywords
        keywords = {
            1: ['8.22', '21.86', '₪', '20'],
            2: ['90', 'ימים', 'אכשרה'],
            3: ['השתתפות עצמית', '47', '72', '50%', '140'],
            4: ['20', 'טיפולים', 'שנה']
        }

        found = []
        for kw in keywords[i]:
            if kw in chunk.content:
                found.append(kw)

        if found:
            print(f"\n✓ Keywords found in this chunk: {', '.join(found)}")

    print("\n" + "="*80)
