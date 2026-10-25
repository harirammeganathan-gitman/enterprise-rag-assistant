from src.conversation.router import Intent, route_message


def test_standalone_feedback_is_handled_without_rag() -> None:
    for message in (
        "I like the response",
        "I like it",
        "okay perfect",
        "that was fantastic",
        "thanks",
    ):
        decision = route_message(message, has_history=True)

        assert decision.intent == Intent.ACKNOWLEDGEMENT
        assert decision.response


def test_substantive_request_takes_precedence_over_feedback() -> None:
    decision = route_message(
        "I like the response, can you explain step two?",
        has_history=True,
    )

    assert decision.intent == Intent.DOCUMENT_QUESTION
    assert decision.response is None


def test_contextual_question_routes_to_rag_as_follow_up() -> None:
    decision = route_message("I like it, can you explain step two?", has_history=True)

    assert decision.intent == Intent.FOLLOW_UP
    assert decision.response is None


def test_positive_word_inside_question_routes_to_rag() -> None:
    decision = route_message(
        "What's a good way to configure the model?",
        has_history=False,
    )

    assert decision.intent == Intent.DOCUMENT_QUESTION


def test_referential_question_without_context_requests_clarification() -> None:
    decision = route_message("Can you explain that?", has_history=False)

    assert decision.intent == Intent.CLARIFY
    assert decision.response


def test_follow_up_requires_conversation_context() -> None:
    assert route_message("Why?", has_history=True).intent == Intent.FOLLOW_UP

    decision = route_message("Why?", has_history=False)

    assert decision.intent == Intent.CLARIFY
    assert decision.response
