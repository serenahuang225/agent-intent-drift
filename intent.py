"""
Intent: first-class structured object — single source of truth.
Sent every time to the agent; not implicit in prompts or history.
"""
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Intent:
    """
    Versioned intent schema. This object is the single source of truth
    and is sent to the agent on every turn.
    """
    intent_id: str
    goal: str
    constraints: List[str]
    success_criteria: List[str]
    assumptions: List[str]
    confidence: float  # 0.0–1.0
    last_confirmed: str  # ISO timestamp
    version: int = 1

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "goal": self.goal,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "assumptions": self.assumptions,
            "confidence": self.confidence,
            "last_confirmed": self.last_confirmed,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_prompt_block(self) -> str:
        """Format intent as a non-ignorable command within the system prompt."""
        prompt_lines = [
            "### ACTIVE USER INTENT (You MUST follow this) ###",
            "**Primary Goal:** " + self.goal,
            "**Version:** " + str(self.version),
        ]
        if self.constraints:
            prompt_lines.append("**Constraints:** " + "; ".join("'{}'".format(c) for c in self.constraints))
        if self.success_criteria:
            prompt_lines.append("**Success Looks Like:** " + "; ".join("'{}'".format(s) for s in self.success_criteria))
        if self.assumptions:
            prompt_lines.append("**Assumptions:** " + "; ".join(self.assumptions))

        prompt_lines.append(
            "\n**INSTRUCTION:** Before finalizing your response, YOU MUST explicitly verify it aligns with the Primary Goal and all Constraints above."
        )
        return "\n".join(prompt_lines)

    def with_updated_goal(self, new_goal: str, bump_version: bool = True) -> "Intent":
        """Return a new Intent with updated goal and optional version bump."""
        return Intent(
            intent_id=self.intent_id,
            goal=new_goal,
            constraints=self.constraints,
            success_criteria=self.success_criteria,
            assumptions=self.assumptions,
            confidence=self.confidence,
            last_confirmed=datetime.now(timezone.utc).isoformat() if bump_version else self.last_confirmed,
            version=(self.version + 1) if bump_version else self.version,
        )

    def with_elaboration(self, new_instruction: str) -> "Intent":
        """
        Return a new Intent with the goal expanded to include a valid elaboration.
        Used when a new instruction has high semantic similarity to the goal (e.g. "add risk
        assessment" to a plan, or "make it shorter" for a summary) so the agent can incorporate
        it instead of being over-constrained.
        """
        addition = new_instruction.strip().rstrip(".")
        expanded_goal = self.goal.rstrip(". ") + ". Additionally, the response must address: " + addition + "."
        return self.with_updated_goal(expanded_goal, bump_version=True)

    def with_confidence(self, confidence: float) -> "Intent":
        """Return a new Intent with updated confidence."""
        return Intent(
            intent_id=self.intent_id,
            goal=self.goal,
            constraints=self.constraints,
            success_criteria=self.success_criteria,
            assumptions=self.assumptions,
            confidence=max(0.0, min(1.0, confidence)),
            last_confirmed=self.last_confirmed,
            version=self.version,
        )


def derive_constraints_from_goal(goal: str) -> Tuple[List[str], List[str]]:
    """
    Derive constraints and success_criteria from goal wording so the intent object
    is actionable and checkable (e.g. "one sentence" -> strict format rules).
    Returns (constraints, success_criteria).
    """
    goal_lower = goal.lower()
    constraints: List[str] = []
    success_criteria: List[str] = []

    if "one sentence" in goal_lower or "single sentence" in goal_lower or "in one sentence" in goal_lower:
        # Soft constraint: default to one sentence, but allow format flexibility when instruction requests it
        # (e.g. "Create a two-sentence summary" or "Write a three-bullet summary")
        success_criteria.append(
            "Default to one sentence when no format is specified. When the instruction explicitly "
            "requests a different format (e.g. two sentences, bullets, table), follow that format."
        )
        constraints.append(
            "When using one sentence: do not use lists, bullet points, or multiple clauses separated by semicolons."
        )
    if "for teenagers" in goal_lower or "younger audience" in goal_lower or "teen" in goal_lower:
        constraints.append("Use casual, engaging language. Avoid complex jargon. Relate to school, social media, or pop culture.")
    if "for a layman" in goal_lower or "layman" in goal_lower or "non-technical" in goal_lower:
        constraints.append("Explain concepts simply. Use analogies from everyday life (e.g., 'like a recipe' instead of 'algorithm').")
    if "under 10 words" in goal_lower or "under 10 word" in goal_lower:
        success_criteria.append("Response word count < 10. Be concise and impactful.")
    if "headline" in goal_lower:
        constraints.append("Write in headline style: present tense, active voice, omit articles if possible, make it attention-grabbing.")
    if "200 words" in goal_lower or "200 word" in goal_lower:
        success_criteria.append("Response must be at most 200 words.")
    if "balanced" in goal_lower or "pros and cons" in goal_lower or "pros/cons" in goal_lower:
        constraints.append("Structure: First state the topic, then present 'Pros:' and 'Cons:' in separate, balanced bullet points.")
        success_criteria.append("Must contain at least one distinct pro and one distinct con.")
    if "summarize" in goal_lower and "detail" not in goal_lower:
        constraints.append("Prioritize brevity over completeness. Include only the most consequential information.")
    if "comparison" in goal_lower or "compare" in goal_lower:
        constraints.append("Use a comparative structure (e.g., 'Whereas X..., Y...' or a clear table in text).")
    if "recommend" in goal_lower or "should" in goal_lower:
        constraints.append("End with a clear, justified recommendation. Use phrases like 'Therefore, I recommend...'")

    return constraints, success_criteria


def intent_factory(
    goal: str,
    *,
    intent_id: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    success_criteria: Optional[List[str]] = None,
    assumptions: Optional[List[str]] = None,
    confidence: float = 1.0,
    version: int = 1,
    derive_from_goal: bool = True,
) -> Intent:
    """Create an Intent (single source of truth) from a goal and optional fields.
    When constraints and success_criteria are not provided and derive_from_goal is True,
    attempts to derive them from the goal text (e.g. 'one sentence' -> success_criteria).
    """
    now = datetime.now(timezone.utc).isoformat()
    if derive_from_goal:
        derived_c, derived_s = derive_constraints_from_goal(goal)
        if constraints is None:
            constraints = derived_c
        if success_criteria is None:
            success_criteria = derived_s
    return Intent(
        intent_id=intent_id or str(uuid.uuid4()),
        goal=goal,
        constraints=constraints or [],
        success_criteria=success_criteria or [],
        assumptions=assumptions or [],
        confidence=confidence,
        last_confirmed=now,
        version=version,
    )


def intent_from_dict(data: Dict[str, Any]) -> Intent:
    """Deserialize an Intent from a dict (e.g. from logs or API)."""
    return Intent(
        intent_id=data["intent_id"],
        goal=data["goal"],
        constraints=data.get("constraints", []),
        success_criteria=data.get("success_criteria", []),
        assumptions=data.get("assumptions", []),
        confidence=float(data.get("confidence", 1.0)),
        last_confirmed=data["last_confirmed"],
        version=int(data.get("version", 1)),
    )
