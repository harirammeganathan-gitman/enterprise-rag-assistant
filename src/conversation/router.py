"""Deterministic routing for conversational and document-oriented messages."""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    SOCIAL = "social"
    ACKNOWLEDGEMENT = "acknowledgement"
    DOCUMENT_QUESTION = "document_question"
    FOLLOW_UP = "follow_up"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    response: str | None = None


def _normalize(message: str) -> str:
    return " ".join(re.findall(r"[\w']+", message.lower()))


def _has_substantive_request(message: str, normalized: str) -> bool:
    if "?" in message:
        return True
    return bool(
        re.match(
            r"^(?:what|why|how|when|where|who|which|whose|whom|"
            r"can|could|would|should|do|does|did|is|are|was|were|"
            r"will|has|have)\b",
            normalized,
        )
        or re.match(
            r"^(?:please\s+)?(?:explain|describe|summari[sz]e|tell me|show me|"
            r"list|compare|find|define|give me|help me|walk me through|"
            r"elaborate on|provide|outline|identify)\b",
            normalized,
        )
    )


def _follow_up_response(normalized: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:why(?: is that)?|how so|tell me more|go on|continue|"
            r"explain (?:that|it)|what about (?:that|it|the .+)|"
            r"which one|can you elaborate|elaborate(?: more)?|"
            r"(?:can|could|would) you (?:please )?"
            r"(?:explain|clarify|elaborate on|expand on) (?:that|it|this)|"
            r"what does (?:that|it|this) mean|"
            r"(?:tell me|say) more(?: about (?:that|it|this))?|"
            r"then what|okay so)",
            normalized,
        )
    )


def _references_prior_context(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:that|it|this|they|them|those|these|above|previous|"
            r"former|latter|second|first)\b",
            normalized,
        )
    )


def route_message(message: str, has_history: bool) -> RouteDecision:
    """Choose a deterministic route before cache lookup or document retrieval."""
    normalized = _normalize(message)

    if _follow_up_response(normalized):
        if has_history:
            return RouteDecision(Intent.FOLLOW_UP)
        return RouteDecision(
            Intent.CLARIFY,
            "What would you like me to clarify? Please include "
            "the question or topic.",
        )

    if re.fullmatch(
        r"(?:hi|hello|hey|hiya|yo|greetings)(?: there)?|"
        r"good (?:morning|afternoon|evening|day)",
        normalized,
    ):
        return RouteDecision(Intent.SOCIAL, "Hello! How can I help you today?")
    if re.fullmatch(r"how are you(?: doing)?|how's it going|what's up", normalized):
        return RouteDecision(
            Intent.SOCIAL, "I'm doing well, thanks! What can I help you with?"
        )
    if re.fullmatch(r"bye|goodbye|see you(?: later)?", normalized):
        return RouteDecision(
            Intent.SOCIAL, "Goodbye! Feel free to come back if you need anything."
        )

    if _has_substantive_request(message, normalized):
        if _references_prior_context(normalized):
            if has_history:
                return RouteDecision(Intent.FOLLOW_UP)
            return RouteDecision(
                Intent.CLARIFY,
                "What would you like me to clarify? Please include "
                "the question or topic.",
            )
        return RouteDecision(Intent.DOCUMENT_QUESTION)

    courtesy = (
        r"(?:ok(?:ay)?|alright|good|great|fantastic|perfect|awesome|cool|nice|"
        r"thanks(?: so much)?|thank you(?: so much)?)"
    )
    feedback = (
        r"(?:i (?:really )?(?:like|liked|love|loved|enjoyed) "
        r"(?:it|that|this|the answer|the response|your answer|your response)|"
        r"(?:that|this|it) (?:was|is) (?:really )?"
        r"(?:good|great|fantastic|perfect|awesome|helpful|useful)|"
        r"(?:that|this) helps|i appreciate (?:it|that)|"
        r"(?:that's|that is|it's|it is|sounds) "
        rf"{courtesy}(?: {courtesy})*|"
        rf"{courtesy}(?: {courtesy})*)"
    )
    if re.fullmatch(feedback, normalized):
        return RouteDecision(
            Intent.ACKNOWLEDGEMENT,
            "Glad it helped! Let me know if you have another question.",
        )

    if _references_prior_context(normalized):
        if has_history:
            return RouteDecision(Intent.FOLLOW_UP)
        return RouteDecision(
            Intent.CLARIFY,
            "What would you like me to clarify? Please include the question or topic.",
        )

    return RouteDecision(Intent.DOCUMENT_QUESTION)
