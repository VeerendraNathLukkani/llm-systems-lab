"""Run one fixed set of triage questions across a small ticket set and print the
probabilities, to see where Laya is confident and where it is not.

The department answer goes through an abstain rule: a flat distribution is reported as
UNCERTAIN rather than forced into a label.
"""

from dataclasses import dataclass

from laya import Router

UNCERTAIN = "UNCERTAIN"

# Eyeballed from ten tickets, so they are illustrative only -- not validated, and not
# thresholds to run anything on.
MIN_TOP_PROBABILITY = 0.60
MIN_MARGIN = 0.20

# expected is the label a human would assign. None means the ticket is genuinely
# ambiguous and we want to look at the distribution rather than score a hit.
TICKETS = [
    {"id": 1, "expected": "billing",
     "text": "My credit card was charged twice for the same order."},
    {"id": 2, "expected": "technical_support",
     "text": "The monitor powers on but the screen stays completely black."},
    {"id": 3, "expected": "returns_and_refunds",
     "text": "I don't want this product anymore. Please refund my money."},
    {"id": 4, "expected": "general_inquiry",
     "text": "What are your customer service opening hours on weekends?"},
    {"id": 5, "expected": "returns_and_refunds",
     "text": "The monitor arrived with a cracked screen. I want my money back."},
    {"id": 6, "expected": "technical_support",
     "text": "The monitor arrived with a cracked screen. Is there anything I can do to fix it?"},
    {"id": 7, "expected": "billing",
     "text": "The invoice says 249 but the product page said 199. The amount is wrong."},
    {"id": 8, "expected": "returns_and_refunds",
     "text": "Can I return this?"},
    {"id": 9, "expected": "general_inquiry",
     "text": "This is absolutely the worst company I have ever dealt with. Unbelievable."},
    {"id": 10, "expected": None,
     "text": "My order is wrong."},
]

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this support ticket?",
        "criteria": {
            "billing": "Payment, invoice or charge problems",
            "technical_support": "The product does not work as expected",
            "returns_and_refunds": "The customer wants a replacement, return or refund",
            "general_inquiry": "Anything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently does this ticket need a human response?",
        "criteria": [
            "No urgency, informational only",
            "Low, can wait several days",
            "Moderate, respond within a day",
            "High, respond within hours",
            "Critical, respond immediately",
        ],
    },
    "is_refund_request": {
        "type": "noul",
        "instructions": "Is the customer explicitly asking for a refund?",
    },
    "contains_profanity": {
        "type": "noul",
        "instructions": "Does the message contain profanity?",
    },
}


@dataclass(frozen=True)
class Decision:
    """The top two candidates for one question, and what we act on."""

    final: str
    top_label: str
    top_probability: float
    second_label: str
    second_probability: float

    @property
    def margin(self) -> float:
        return self.top_probability - self.second_probability


def decide(probabilities):
    """Take the top label only when it is both confident and clearly ahead."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    (top_label, top_probability), (second_label, second_probability) = ranked[0], ranked[1]
    confident = top_probability >= MIN_TOP_PROBABILITY
    clear = top_probability - second_probability >= MIN_MARGIN
    return Decision(
        final=top_label if confident and clear else UNCERTAIN,
        top_label=top_label,
        top_probability=top_probability,
        second_label=second_label,
        second_probability=second_probability,
    )


def main():
    print("Loading Laya engine...")
    router = Router(preload=True)

    correct = wrong = abstained = 0
    for ticket in TICKETS:
        answers = router.system_one(ticket["text"], QUESTIONS)["answers"]
        probabilities = answers["department"]["probabilities"]
        decision = decide(probabilities)
        expected = ticket["expected"]

        if expected is None:
            outcome = "ambiguous by design"
        elif decision.final == UNCERTAIN:
            abstained += 1
            was = "right" if decision.top_label == expected else "WRONG"
            outcome = f"abstained (top-1 was {decision.top_label}, {was})"
        elif decision.final == expected:
            correct += 1
            outcome = "correct"
        else:
            wrong += 1
            outcome = f"WRONG, expected {expected}"

        print()
        print(f"[{ticket['id']:2}] {ticket['text']}")
        print(f"      expected {expected or '(ambiguous)'}")
        for label, probability in sorted(probabilities.items(), key=lambda i: i[1], reverse=True):
            print(f"        {label:<22} {probability:.2f}")
        print(
            f"      top_prob = {decision.top_probability:.2f}"
            f"   second = {decision.second_probability:.2f}"
            f"   margin = {decision.margin:.2f}"
        )
        print(f"      FINAL = {decision.final}   -> {outcome}")

    print()
    print(
        f"decided correctly {correct}   decided wrongly {wrong}   abstained {abstained}"
        f"   (of {len(TICKETS)} tickets, 1 ambiguous by design)"
    )


if __name__ == "__main__":
    main()
