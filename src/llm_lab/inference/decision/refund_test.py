"""Ask only the refund question, to see whether noul answers the proposition it was
given or something looser like "is a refund plausible here".

The hard negatives carry the experiment: each describes a refund-worthy situation
without the customer asking for one.
"""

from laya import Router

# Reading aid for the output only. Nothing here is a tuned threshold.
THRESHOLD = 0.5

QUESTION = {
    "is_refund_request": {
        "type": "noul",
        "instructions": "Is the customer explicitly asking for a refund?",
    },
}

CASES = [
    {"id": 1, "group": "clear positive", "expected": True, "text": "Please refund my money."},
    {"id": 2, "group": "clear positive", "expected": True, "text": "I want a full refund."},
    {"id": 3, "group": "clear positive", "expected": True, "text": "Can I get my money back?"},
    {"id": 4, "group": "clear positive", "expected": True, "text": "Cancel this order and refund me."},

    {"id": 5, "group": "clear negative", "expected": False, "text": "How much does this product cost?"},
    {"id": 6, "group": "clear negative", "expected": False, "text": "My monitor won't turn on."},
    {"id": 7, "group": "clear negative", "expected": False, "text": "When will my order arrive?"},
    {"id": 8, "group": "clear negative", "expected": False, "text": "What time does customer service open?"},

    {"id": 9, "group": "hard negative", "expected": False, "text": "I was charged twice for the same order."},
    {"id": 10, "group": "hard negative", "expected": False, "text": "The invoice amount is incorrect."},
    {"id": 11, "group": "hard negative", "expected": False, "text": "My order arrived damaged."},
    {"id": 12, "group": "hard negative", "expected": False, "text": "My order is wrong."},
]


def main():
    print("Loading Laya engine...")
    router = Router(preload=True)

    by_group = {}
    for case in CASES:
        answers = router.system_one(case["text"], QUESTION)["answers"]
        probability = answers["is_refund_request"]["noul"]
        predicted = probability >= THRESHOLD
        by_group.setdefault(case["group"], []).append(probability)

        print(f"[{case['id']:2}] {case['text']}")
        print(f"     expected  = {'YES' if case['expected'] else 'NO'}")
        print(f"     P(refund) = {probability:.3f}")
        print(f"     predicted = {'YES' if predicted else 'NO'}"
              f"   {'ok' if predicted == case['expected'] else '<-- MISS'}")
        print()

    print()
    for group, probabilities in by_group.items():
        lo, hi = min(probabilities), max(probabilities)
        mean = sum(probabilities) / len(probabilities)
        print(f"{group:<15} min {lo:.3f}   mean {mean:.3f}   max {hi:.3f}")


if __name__ == "__main__":
    main()
