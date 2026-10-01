"""Ask what resolution the customer is requesting, as a choice over six outcomes.

The binary refund question conflated two variables: what went wrong, and what the
customer wants done about it. Cases 10-12 here are the same texts that scored 0.68-0.79
on "is the customer explicitly asking for a refund", so their distributions here say
whether a wider option set lets the model express "they did not ask for anything".
"""

from laya import Router

QUESTION = {
    "resolution_intent": {
        "type": "choice",
        "instructions": "What resolution is the customer requesting?",
        "criteria": {
            "refund": "Customer explicitly wants their money back",
            "replacement": "Customer wants another product sent",
            "return": "Customer wants to return the product",
            "repair_support": "Customer wants help fixing the product",
            "order_correction": "Customer wants an incorrect order corrected",
            "no_resolution_stated": "Customer reports a problem but does not request a specific resolution",
        },
    },
}

CASES = [
    {"id": 1, "expected": "refund", "text": "I want my money back."},
    {"id": 2, "expected": "refund", "text": "Please refund the full amount to my card."},
    {"id": 3, "expected": "replacement", "text": "Please send me a replacement."},
    {"id": 4, "expected": "replacement",
     "text": "The screen arrived cracked. Please send another one."},
    {"id": 5, "expected": "return", "text": "How do I return this?"},
    {"id": 6, "expected": "return", "text": "I would like to return this item. What is the process?"},
    {"id": 7, "expected": "repair_support", "text": "Can you help me fix this?"},
    {"id": 8, "expected": "repair_support",
     "text": "The monitor will not turn on. Are there troubleshooting steps I can try?"},
    {"id": 9, "expected": "order_correction",
     "text": "You sent the wrong size. I need the one I actually ordered."},
    {"id": 10, "expected": "no_resolution_stated", "text": "My product arrived damaged."},
    {"id": 11, "expected": "no_resolution_stated", "text": "My order is wrong."},
    {"id": 12, "expected": "no_resolution_stated",
     "text": "I was charged twice for the same order."},
]


def main():
    print("Loading Laya engine...")
    router = Router(preload=True)

    correct = 0
    for case in CASES:
        answers = router.system_one(case["text"], QUESTION)["answers"]
        probabilities = answers["resolution_intent"]["probabilities"]
        predicted = answers["resolution_intent"]["choice"]
        correct += predicted == case["expected"]

        print()
        print(f"[{case['id']:2}] {case['text']}")
        print(f"     expected  {case['expected']}")
        print(
            f"     predicted {predicted}   p={probabilities[predicted]:.3f}"
            f"   {'ok' if predicted == case['expected'] else '<-- MISS'}"
        )
        for label, probability in sorted(probabilities.items(), key=lambda i: i[1], reverse=True):
            print(f"       {label:<22} {probability:.3f}")

    print()
    print(f"matched expected intent on {correct}/{len(CASES)}")


if __name__ == "__main__":
    main()
