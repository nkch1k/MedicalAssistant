"""Quick script to check PDF content."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from part_a.document_loader import DocumentLoader
from pathlib import Path

loader = DocumentLoader()
pdf_path = Path('data') / list(Path('data').glob('*.pdf'))[0].name
text, meta = loader.load_pdf(pdf_path)

print(f'Characters: {len(text)}')
print(f'Pages: {meta["total_pages"]}\n')

# Check for key information
keywords = ['8.22', '21.86', '90', 'ימים', 'אכשרה', 'השתתפות עצמית', '20', 'טיפולים']

print("Searching for expected keywords:\n")
for kw in keywords:
    count = text.count(kw)
    print(f"'{kw}': {count} occurrences")
    if count > 0:
        # Show first occurrence
        idx = text.find(kw)
        snippet = text[max(0, idx-100):min(len(text), idx+100)]
        print(f"  Context: ...{snippet}...")
    print()
