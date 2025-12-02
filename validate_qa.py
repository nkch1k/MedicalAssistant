#!/usr/bin/env python3
"""
Validation script for RAG system.
Tests with predefined questions and expected keywords.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from part_a.main import RAGApplication
from part_a.config import load_config


# Test questions with expected keywords
TEST_CASES = [
    {
        "question": "כמה יעלה לי ביטוח טיפולי אקופונקטורה?",
        "expected_keywords": ["8.22", "21.86", "₪", "20"],
        "description": "Cost of acupuncture treatment"
    },
    {
        "question": "ממתי ניתן לקבל החזר על הטיפול באקופונקטורה?",
        "expected_keywords": ["90", "ימים", "אכשרה"],
        "description": "Waiting period for acupuncture"
    },
    {
        "question": "האם מקבלים החזר מלא על הטיפולים?",
        "expected_keywords": ["השתתפות עצמית", "47", "72", "50%", "140"],
        "description": "Full reimbursement question"
    },
    {
        "question": "כמה טיפולים מכוסים?",
        "expected_keywords": ["20", "טיפולים", "שנה"],
        "description": "Number of covered treatments"
    }
]


def validate_answer(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Validate answer against expected keywords.

    Args:
        answer: Generated answer
        expected_keywords: List of keywords that should appear

    Returns:
        Validation results
    """
    found_keywords = [kw for kw in expected_keywords if kw in answer]
    missing_keywords = [kw for kw in expected_keywords if kw not in answer]

    score = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0

    return {
        "score": score,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "pass": score >= 0.5  # At least 50% of keywords should be present
    }


def run_validation(document_path: Path) -> Dict[str, Any]:
    """
    Run validation tests.

    Args:
        document_path: Path to PDF document

    Returns:
        Validation report
    """
    print("=" * 80)
    print("RAG SYSTEM VALIDATION")
    print("=" * 80)
    print()

    # Initialize application
    print("Initializing RAG system...")
    app = RAGApplication()
    pages, chunks = app.initialize_system(document_path)
    print()

    # Run tests
    results = []
    total_score = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        print("-" * 80)
        print(f"TEST {i}/{len(TEST_CASES)}: {test_case['description']}")
        print("-" * 80)
        print(f"Question: {test_case['question']}")
        print()

        # Get answer
        try:
            answer = app.answer_question(test_case['question'])
            print(f"Answer: {answer}")
            print()

            # Validate
            validation = validate_answer(answer, test_case['expected_keywords'])

            print(f"✓ Found keywords: {validation['found_keywords']}")
            if validation['missing_keywords']:
                print(f"✗ Missing keywords: {validation['missing_keywords']}")
            print(f"Score: {validation['score']:.2%}")
            print(f"Status: {'PASS' if validation['pass'] else 'FAIL'}")
            print()

            results.append({
                "test_number": i,
                "question": test_case['question'],
                "description": test_case['description'],
                "answer": answer,
                "validation": validation
            })

            total_score += validation['score']

        except Exception as e:
            print(f"ERROR: {e}")
            print()
            results.append({
                "test_number": i,
                "question": test_case['question'],
                "description": test_case['description'],
                "error": str(e)
            })

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.get('validation', {}).get('pass', False))
    total = len(results)
    avg_score = total_score / total if total > 0 else 0

    print(f"Tests passed: {passed}/{total}")
    print(f"Average score: {avg_score:.2%}")
    print()

    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "document": str(document_path),
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "average_score": avg_score,
        "test_results": results
    }

    return report


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save validation report to JSON file.

    Args:
        report: Validation report
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point."""
    # Find PDF document
    config = load_config()
    pdf_files = list(config.data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"ERROR: No PDF files found in {config.data_dir}")
        print("Please place your insurance document PDF in the 'data' directory.")
        return 1

    # Use the first PDF found
    document_path = pdf_files[0]
    print(f"Using document: {document_path.name}\n")

    try:
        # Run validation
        report = run_validation(document_path)

        # Save report
        output_path = Path("validation_report.json")
        save_report(report, output_path)

        # Return exit code based on results
        return 0 if report['passed'] == report['total_tests'] else 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
