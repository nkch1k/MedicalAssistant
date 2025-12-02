# Answer Accuracy Validation Method

## Overview

Automated validation system that tests the RAG system with predefined questions and verifies answers contain expected information.

## Validation Approach

**Method:** Keyword-based validation with scoring system

The system compares generated answers against expected keywords to verify correctness.

## Test Cases

Four mandatory test questions from requirements:

1. **כמה יעלה לי ביטוח טיפולי אקופונקטורה?**
   Expected keywords: `8.22`, `21.86`, `₪`, `20`

2. **ממתי ניתן לקבל החזר על הטיפול באקופונקטורה?**
   Expected keywords: `90`, `ימים`, `אכשרה`

3. **האם מקבלים החזר מלא על הטיפולים?**
   Expected keywords: `השתתפות עצמית`, `47`, `72`, `50%`, `140`

4. **כמה טיפולים מכוסים?**
   Expected keywords: `20`, `טיפולים`, `שנה`

## Scoring System

**Formula:**
```
score = found_keywords / total_expected_keywords
```

**Pass Criteria:**
- Score ≥ 50% (at least half of keywords present)
- All keywords ideally should be found

## Running Validation

```bash
python validate_qa.py
```

Output:
- Each question with generated answer
- Found vs missing keywords
- Score per question
- Overall pass/fail status
- JSON report saved to `validation_report.json`

## Validation Metrics

**Per Question:**
- Found keywords list
- Missing keywords list
- Accuracy score (0-100%)
- Pass/fail status

**Overall:**
- Tests passed/failed count
- Average accuracy score
- Timestamp and document info

## Limitations

- Keyword matching is case-sensitive for numbers/symbols
- Doesn't validate semantic correctness, only presence of key information
- May flag correct answers as incorrect if wording differs

## Future Improvements

- Semantic similarity scoring using embeddings
- LLM-based answer evaluation
- Reference answer comparison
- User feedback integration
