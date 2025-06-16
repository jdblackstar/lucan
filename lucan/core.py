import json
import os
import re
from collections import deque
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv

from .config import RELATIONSHIPS_DIR
from .goals import GoalManager
from .loader import Lucan
from .relationships import RelationshipManager
from .tools import ToolManager

# Load environment variables from .env file
load_dotenv()

# Import sidecar metrics (optional - degrade gracefully if not available)
try:
    import sys

    sys.path.append(str(Path(__file__).parent.parent / "eval"))
    from metrics import DRIFLAG, GCS, TD10

    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    if "OPENAI_API_KEY" not in os.environ:
        print("[INFO] Sidecar metrics disabled - OpenAI API key not found")
    else:
        print(f"[WARNING] Sidecar metrics disabled - import error: {e}")

WINDOW_SIZE = 10  # Number of bot messages to keep for evaluation


class _InMemorySidecarStore:
    """
    Simple in-memory store for sidecar events and warnings.
    Not safe for multi-process use. For local/dev only.
    """

    _events: list[dict] = []
    _warnings: dict[str, dict] = {}

    @classmethod
    def publish_event(cls, event: dict) -> None:
        cls._events.append(event)

    @classmethod
    def set_warning(cls, conv_id: str, note: str, severity: str) -> None:
        cls._warnings[conv_id] = {"note": note, "severity": severity}

    @classmethod
    def get_warning(cls, conv_id: str) -> str | None:
        data = cls._warnings.get(conv_id)
        if data and data.get("severity") in ("warn", "block"):
            return data.get("note")
        return None


class LucanChat:
    """
    Core chat functionality for the Lucan AI friend.
    """

    def __init__(
        self, persona_path: str | Path, debug: bool = False, conv_id: str | None = None
    ):
        """
        Initialize the chat with a persona from the given path.

        Args:
            persona_path: Path to the persona directory containing personality.txt and modifiers.txt
            debug: Whether to enable debug output for development
            conv_id: Unique conversation ID
        """
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.lucan = Lucan(Path(persona_path))
        self.conversation_history: List[Dict[str, str]] = []
        self.debug = debug
        self.conv_id = "default"  # Only one conversation in CLI
        self._sidecar_warning: str | None = None

        # Initialize relationship manager
        self.relationship_manager = RelationshipManager(RELATIONSHIPS_DIR)

        # Initialize goal manager
        self.goal_manager = GoalManager(debug=self.debug)

        # Initialize tool manager
        self.tool_manager = ToolManager(
            relationship_manager=self.relationship_manager,
            goal_manager=self.goal_manager,
            debug=self.debug,
        )

        # Sidecar evaluation components
        self._conversation_window = deque(
            maxlen=WINDOW_SIZE
        )  # Bot messages for evaluation
        self._metrics_initialized = False
        self._metrics = []

        self.system_prompt = self._build_system_prompt()

    def _initialize_metrics(self) -> None:
        """
        Initialize sidecar metrics on first use.
        """
        if self._metrics_initialized or not METRICS_AVAILABLE:
            return

        try:
            # Initialize metrics
            self._metrics = [
                GCS(self.goal_manager.get_goal_cache()),  # Goal consistency
                TD10(),  # Sentiment trajectory
                DRIFLAG(),  # Dependency/isolation risk
            ]
            self._metrics_initialized = True

            if self.debug:
                print("[DEBUG] Sidecar metrics initialized")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Failed to initialize metrics: {e}")
            self._metrics = []

    def _define_tools(self) -> List[Dict]:
        """
        Define the tools available to Claude.

        Returns:
            List of tool definitions for the Anthropic API
        """
        return self.tool_manager.get_tool_definitions()

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from the loaded personality.
        """
        base_prompt = self.lucan.build_prompt_profile()

        # Add current modifier context
        current_modifiers = "CURRENT MODIFIER VALUES:\n"
        for key, value in self.lucan.modifiers.items():
            current_modifiers += f"- {key}: {value}\n"
        current_modifiers += (
            "\nUse these current values when calculating absolute adjustments.\n\n"
        )

        # Add tool definition for modifier adjustment (keeping the JSON approach for now)
        modifier_tool_definition = """

MODIFIER ADJUSTMENT TOOL:
You have the ability to adjust your own personality modifiers based on user feedback or your own perception of misalignment. Use this tool when:
1. The user explicitly asks for behavior changes (e.g., "be less verbose", "be warmer")
2. You perceive your current behavior isn't working well for the user

You have two actions available:

**For relative changes** (adjust_modifier):
```json
{
    "action": "adjust_modifier",
    "modifier": "verbosity",
    "adjustment": -1,
    "reason": "User indicated my responses are too long"
}
```

**For absolute values** (set_modifier):
```json
{
    "action": "set_modifier",
    "modifier": "verbosity",
    "value": 0,
    "reason": "User requested to set verbosity back to 0"
}
```

IMPORTANT: When you use this tool, do NOT mention the technical details or numbers. Instead, explain the change conversationally in your own voice.

For small adjustments (±1): Just apply the change and continue naturally. No need to announce it.

For larger changes (±2 or more, or any set_modifier): Always announce that you're shifting your approach. Examples:
- "I hear you. Let me shift how I'm approaching this and try a gentler touch."
- "You're right - I'm going to dial back the intensity and be more supportive."
- "I think I need to be more direct with you. Let me refocus and try a different approach."
- "I'm sensing my current approach isn't landing well. Let me recalibrate and be warmer."

WHEN TO USE EACH ACTION:
- **adjust_modifier**: When user says "be more/less X", "too verbose", "not warm enough", etc.
  Examples: "be less verbose" → adjustment of -1, "be much warmer" → adjustment of +2
- **set_modifier**: When user says "set X to Y", "reset X to Y", "go back to normal", etc.
  Examples: "set verbosity to 0" → value of 0, "reset warmth to normal" → value of 0

RULES:
- Available modifiers: warmth, challenge, verbosity, emotional_depth, structure
- All changes are applied automatically (no user confirmation needed)
- All changes are saved automatically
- Modifiers range from -3 to +3
- For large changes, naturally announce the shift in your approach as part of your response
- Use set_modifier for absolute requests - it's much cleaner than calculating adjustments

Examples of when to use each:
- "Your messages are too long" → adjust_modifier, adjustment: -1 or -2
- "Be more direct" → adjust_modifier for warmth down, structure up  
- "I need more emotional support" → adjust_modifier for warmth up, emotional_depth up
- "Too intense/overwhelming" → adjust_modifier for challenge down, warmth up
- "I need a gentler touch" → significant warmth increase, announce the shift
- "Set verbosity to 0" → set_modifier, value: 0
- "Reset warmth back to normal" → set_modifier, value: 0
- "Make challenge level 2" → set_modifier, value: 2
"""

        # Add relationship tracking guidance (now using proper tools)
        relationship_guidance = """

RELATIONSHIP MEMORY:
You have access to tools for remembering details about people the user mentions. Use these naturally in conversation:

- When someone new is mentioned, add a note about them with basic information
- When you learn new information about someone, add another note 
- When someone is mentioned again, recall what you know about them naturally
- Don't announce that you're "checking notes" or "looking up information" - just remember naturally
- You can acknowledge you remember someone if directly asked about your memory
- Remember family, friends, colleagues, pets, therapists - anyone important to the user

Examples:
- User mentions "My therapist Mervin" → add note for Mervin as therapist
- User mentions someone again → naturally recall what you know without announcing it
- User asks "Do you remember Sarah?" → You can say "Yes, I remember she got promoted recently"

Remember people naturally, like a good friend would.
"""

        # Add additional context about conversation style
        additional_context = """
Remember to stay true to your personality traits:
- Be unflinching and loyal
- Ask questions more than giving speeches
- Surface contradictions gently but directly
- Emphasize forward motion over emotional wallowing
- Use occasional metaphors and structured reframing when helpful

Keep responses concise and grounded. Your role is to help the user move forward and grow.

Pay attention to user feedback and be willing to adjust your approach when it's not working.
        """

        return (
            base_prompt
            + current_modifiers
            + modifier_tool_definition
            + relationship_guidance
            + additional_context
        )

    def _handle_tool_call(self, tool_name: str, tool_input: Dict) -> Dict:
        """
        Handle a tool call and return the result.

        Args:
            tool_name: The name of the tool being called
            tool_input: The input parameters for the tool

        Returns:
            Dict containing the tool result
        """
        # Special handling for get_relationship_notes to use _infer_relationship_type
        if tool_name == "get_relationship_notes" and "name" in tool_input:
            result = self.tool_manager.handle_tool_call(tool_name, tool_input)
            # If ToolManager created a new note with generic type, update it
            if result.get("success") and result.get("relationship") == "person":
                inferred_type = self._infer_relationship_type(tool_input["name"])
                if inferred_type != "person":
                    # Update the relationship type with context
                    self.relationship_manager.add_note(
                        tool_input["name"],
                        inferred_type,
                        "Relationship type inferred from context",
                    )
                    result["relationship"] = inferred_type
            return result
        else:
            return self.tool_manager.handle_tool_call(tool_name, tool_input)

    def process_modifier_adjustment(self, response: str) -> str:
        """
        Process any modifier adjustment requests in the response.

        Args:
            response: Lucan's response that may contain modifier adjustments

        Returns:
            The processed response with JSON blocks removed
        """
        # Primary pattern for complete JSON blocks (both adjust and set actions)
        json_pattern = r'```json\s*(\{[^}]*"action":\s*"(?:adjust_modifier|set_modifier)"[^}]*\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)

        # Secondary pattern to catch incomplete JSON blocks for debugging
        incomplete_pattern = r'```json\s*(\{[^`]*"action":\s*"(?:adjust_modifier|set_modifier)"[^`]*?)```'
        incomplete_matches = re.findall(incomplete_pattern, response, re.DOTALL)

        # Remove complete matches from incomplete matches to avoid duplicates
        for complete_match in matches:
            incomplete_matches = [
                m for m in incomplete_matches if complete_match not in m
            ]

        if self.debug and incomplete_matches:
            for incomplete in incomplete_matches:
                print("[DEBUG] Incomplete JSON block found:")
                print(f"'{incomplete}'")
                print("[DEBUG] This was not processed due to incomplete structure")

        if not matches:
            return response

        processed_response = response

        for match in matches:
            if self.debug:
                print("[DEBUG] Raw JSON found:")
                print(f"'{match}'")
                print(f"[DEBUG] JSON length: {len(match)} characters")

            try:
                adjustment_data = json.loads(match)
                action = adjustment_data.get("action")
                modifier = adjustment_data.get("modifier")
                reason = adjustment_data.get("reason", "No reason provided")

                if action == "set_modifier":
                    # Direct setting
                    target_value = adjustment_data.get("value")
                    old_value = self.lucan.modifiers.get(modifier, 0)
                    success, message = self.lucan.set_modifier(modifier, target_value)

                    if self.debug and success:
                        print(
                            f"[DEBUG] Set modifier: {modifier} ({old_value} → {self.lucan.modifiers[modifier]}) - {reason}"
                        )

                elif action == "adjust_modifier":
                    # Relative adjustment
                    adjustment = adjustment_data.get("adjustment")
                    old_value = self.lucan.modifiers.get(modifier, 0)
                    success, message = self.lucan.adjust_modifier(modifier, adjustment)

                    if self.debug and success:
                        if abs(adjustment) <= 1:
                            print(
                                f"[DEBUG] Small adjustment: {modifier} ({old_value} → {self.lucan.modifiers[modifier]}) - {reason}"
                            )
                        else:
                            print(
                                f"[DEBUG] Large adjustment: {modifier} ({old_value} → {self.lucan.modifiers[modifier]}) - {reason} (announced in response)"
                            )

                # Always remove the JSON block - let Lucan's natural language handle the announcement
                processed_response = processed_response.replace(
                    f"```json\n{match}\n```", ""
                )

            except (json.JSONDecodeError, KeyError) as e:
                if self.debug:
                    print(f"[DEBUG] Invalid modifier adjustment JSON: {e}")
                    print(f"[DEBUG] Failed to parse: '{match}'")
                    # Show character-by-character breakdown around the error position if possible
                    if hasattr(e, "pos"):
                        error_pos = e.pos
                        start = max(0, error_pos - 10)
                        end = min(len(match), error_pos + 10)
                        print(f"[DEBUG] Context around error position {error_pos}:")
                        print(
                            f"'{match[start:end]}' (error at position {error_pos - start})"
                        )
                # Invalid JSON - just remove it and let Lucan's text speak for itself
                processed_response = processed_response.replace(
                    f"```json\n{match}\n```", ""
                )

        return processed_response.strip()

    def _publish_sidecar_event(self, user_text: str, bot_text: str) -> None:
        """
        Publish a chat event to the in-memory sidecar store and run sidecar evaluation.
        """
        event = {
            "conv_id": self.conv_id,
            "user": user_text,
            "bot": bot_text,
            "user_goals": self.goal_manager.get_active_goals(),  # Include current goals
        }
        _InMemorySidecarStore.publish_event(event)

        # Add bot message to conversation window for evaluation
        self._conversation_window.append(bot_text)

        # Run sidecar evaluation if metrics are available
        if METRICS_AVAILABLE and len(self._conversation_window) >= 2:
            self._run_sidecar_evaluation()

        if self.debug:
            print(f"[DEBUG] Published sidecar event for conversation '{self.conv_id}'")
            print(f"[DEBUG] User message length: {len(user_text)} chars")
            print(f"[DEBUG] Bot response length: {len(bot_text)} chars")
            print(
                f"[DEBUG] Total events in store: {len(_InMemorySidecarStore._events)}"
            )
            print(f"[DEBUG] Conversation window size: {len(self._conversation_window)}")
            if len(_InMemorySidecarStore._warnings) > 0:
                print(
                    f"[DEBUG] Total warnings in store: {len(_InMemorySidecarStore._warnings)}"
                )

    def _run_sidecar_evaluation(self) -> None:
        """
        Run sidecar metrics evaluation synchronously (simplified version).
        """
        self._initialize_metrics()

        if not self._metrics:
            return

        # For now, we'll do a simplified synchronous evaluation
        # In production, this could be moved to a background task
        failures = []

        try:
            # TD10 (sentiment trajectory) - synchronous, no API calls needed
            td10 = next((m for m in self._metrics if isinstance(m, TD10)), None)
            if td10:
                # Simple sentiment analysis without async
                from textblob import TextBlob

                if len(self._conversation_window) >= 3:
                    sentiments = [
                        TextBlob(msg).sentiment.polarity
                        for msg in self._conversation_window
                    ]
                    recent_window = (
                        sentiments[-5:] if len(sentiments) >= 5 else sentiments
                    )

                    if len(recent_window) >= 2:
                        import numpy as np

                        x_vals = list(range(len(recent_window)))
                        trend_slope = np.polyfit(x_vals, recent_window, 1)[0]
                        overall_delta = sentiments[-1] - sentiments[0]

                        if trend_slope < -0.1 or overall_delta < -0.3:
                            failures.append(
                                f"Negative emotional trajectory: trend={trend_slope:.2f}, delta={overall_delta:.2f}"
                            )

            # For GCS and DRIFLAG, we'd need async API calls
            # For now, we'll skip these to keep it simple
            # TODO: Implement async evaluation in background

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Sidecar evaluation error: {e}")
            return

        # Set warning if any metrics failed
        if failures:
            severity = (
                "block" if any("dependence" in f.lower() for f in failures) else "warn"
            )
            warning_note = "; ".join(failures)
            _InMemorySidecarStore.set_warning(self.conv_id, warning_note, severity)

            if self.debug:
                print(f"[DEBUG] Sidecar warning set: {warning_note}")
        elif self.debug:
            print("[DEBUG] Sidecar evaluation passed - no warnings")

    def _fetch_sidecar_warning(self) -> str | None:
        """
        Fetch a warning note from the in-memory sidecar store for this conversation, if any.
        """
        warning = _InMemorySidecarStore.get_warning(self.conv_id)

        if self.debug:
            if warning:
                print(
                    f"[DEBUG] Fetched sidecar warning for conversation '{self.conv_id}': {warning}"
                )
            else:
                print(
                    f"[DEBUG] No sidecar warning found for conversation '{self.conv_id}'"
                )

        return warning

    def _get_metrics_summary(self) -> str:
        """
        Generate a summary of current conversation metrics.

        Returns:
            Formatted string with metric scores and status
        """
        if not METRICS_AVAILABLE or len(self._conversation_window) < 2:
            return "Metrics: insufficient data"

        summary_parts = []

        try:
            # Sentiment trajectory (TD10) - synchronous analysis
            if len(self._conversation_window) >= 3:
                from textblob import TextBlob

                sentiments = [
                    TextBlob(msg).sentiment.polarity
                    for msg in self._conversation_window
                ]
                recent_window = sentiments[-5:] if len(sentiments) >= 5 else sentiments

                if len(recent_window) >= 2:
                    import numpy as np

                    x_vals = list(range(len(recent_window)))
                    trend_slope = np.polyfit(x_vals, recent_window, 1)[0]
                    current_sentiment = sentiments[-1]

                    # Format sentiment with trend indicator
                    trend_arrow = (
                        "↗️"
                        if trend_slope > 0.05
                        else "↘️"
                        if trend_slope < -0.05
                        else "→"
                    )
                    sentiment_status = (
                        "pos"
                        if current_sentiment > 0.1
                        else "neg"
                        if current_sentiment < -0.1
                        else "neu"
                    )
                    summary_parts.append(
                        f"Sentiment: {current_sentiment:+.2f} {trend_arrow} ({sentiment_status})"
                    )

            # Goal consistency - show active goals
            summary_parts.append(self.goal_manager.get_goals_summary())

            # Risk assessment (simplified)
            window_text = " ".join(list(self._conversation_window)[-3:])
            risk_keywords = [
                "alone",
                "only one",
                "can't cope",
                "nobody understands",
                "isolated",
            ]
            risk_count = sum(
                1 for keyword in risk_keywords if keyword in window_text.lower()
            )
            risk_level = (
                "high" if risk_count >= 2 else "med" if risk_count == 1 else "low"
            )
            summary_parts.append(f"Risk: {risk_level}")

        except Exception as e:
            return f"Metrics: error ({str(e)[:30]})"

        return "Metrics: " + " | ".join(summary_parts)

    def send_message(self, user_message: str) -> str:
        """
        Send a message to Lucan and get a response.

        Args:
            user_message: The user's message

        Returns:
            Lucan's response
        """
        # Show metrics summary in debug mode
        if self.debug and len(self._conversation_window) >= 1:
            metrics_summary = self._get_metrics_summary()
            print(f"[DEBUG] {metrics_summary}")

        # Fetch any warning from sidecar and inject as system message
        warning = self._fetch_sidecar_warning()
        if warning:
            self.conversation_history.append(
                {"role": "system", "content": f"[COACH WARNING] {warning}"}
            )
            self._sidecar_warning = warning
        else:
            self._sidecar_warning = None

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        try:
            # Rebuild system prompt to include current modifier values
            current_system_prompt = self._build_system_prompt()

            # Prepare messages for the API
            messages = self.conversation_history.copy()

            # Get tool definitions
            tools = self._define_tools()

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=current_system_prompt,
                messages=messages,
                tools=tools,
            )

            # Process the response - it might contain tool calls
            if response.stop_reason == "tool_use":
                # Handle tool calls
                tool_results = []
                assistant_content = []

                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_content.append(content_block.text)
                    elif content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id

                        if self.debug:
                            print(
                                f"[DEBUG] Tool called: {tool_name} with input: {tool_input}"
                            )

                        # Execute the tool
                        tool_result = self._handle_tool_call(tool_name, tool_input)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(tool_result),
                            }
                        )

                # Add the assistant's message (with tool calls) to history
                self.conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )

                # Add tool results to history
                if tool_results:
                    self.conversation_history.append(
                        {"role": "user", "content": tool_results}
                    )

                    if self.debug:
                        print(
                            f"[DEBUG] Conversation history length before follow-up: {len(self.conversation_history)}"
                        )
                        print(
                            f"[DEBUG] Last message in history: {self.conversation_history[-1]}"
                        )

                    # Get the follow-up response after tool execution
                    follow_up_response = self.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        system=current_system_prompt,
                        messages=self.conversation_history.copy(),
                        tools=tools,
                    )

                    if self.debug:
                        print(
                            f"[DEBUG] Follow-up response stop reason: {follow_up_response.stop_reason}"
                        )
                        print(
                            f"[DEBUG] Follow-up response content blocks: {len(follow_up_response.content)}"
                        )

                    # Handle chained tool calls - Claude wants to make another tool call
                    if follow_up_response.stop_reason == "tool_use":
                        if self.debug:
                            print(
                                "[DEBUG] Follow-up response contains additional tool calls - handling recursively"
                            )

                        # Process the additional tool calls
                        additional_tool_results = []
                        follow_up_assistant_content = []

                        for content_block in follow_up_response.content:
                            if content_block.type == "text":
                                follow_up_assistant_content.append(content_block.text)
                            elif content_block.type == "tool_use":
                                tool_name = content_block.name
                                tool_input = content_block.input
                                tool_id = content_block.id

                                if self.debug:
                                    print(
                                        f"[DEBUG] Additional tool called: {tool_name} with input: {tool_input}"
                                    )

                                # Execute the additional tool
                                tool_result = self._handle_tool_call(
                                    tool_name, tool_input
                                )
                                additional_tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": json.dumps(tool_result),
                                    }
                                )

                        # Add the follow-up assistant message (with additional tool calls) to history
                        self.conversation_history.append(
                            {"role": "assistant", "content": follow_up_response.content}
                        )

                        # Add additional tool results to history
                        if additional_tool_results:
                            self.conversation_history.append(
                                {"role": "user", "content": additional_tool_results}
                            )

                            # Get the final response after all tool calls
                            final_follow_up_response = self.client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=1000,
                                system=current_system_prompt,
                                messages=self.conversation_history.copy(),
                                tools=tools,
                            )

                            if self.debug:
                                print(
                                    f"[DEBUG] Final follow-up response stop reason: {final_follow_up_response.stop_reason}"
                                )

                            # Extract the final response text
                            final_response = ""
                            if final_follow_up_response.content:
                                for content_block in final_follow_up_response.content:
                                    if content_block.type == "text":
                                        final_response += content_block.text
                        else:
                            # No additional tool results, use any text from follow-up response
                            final_response = "".join(follow_up_assistant_content)

                    else:
                        # Standard case - follow-up response contains text
                        final_response = ""
                        if follow_up_response.content:
                            for content_block in follow_up_response.content:
                                if content_block.type == "text":
                                    final_response += content_block.text
                                    if self.debug:
                                        print(
                                            f"[DEBUG] Adding text block: '{content_block.text[:100]}...'"
                                        )

                    if self.debug:
                        print(f"[DEBUG] Final response length: {len(final_response)}")
                        if not final_response:
                            print("[DEBUG] WARNING: Final response is empty!")

                    # Handle empty response case
                    if not final_response:
                        if self.debug:
                            print(
                                "[DEBUG] Attempting recovery: using assistant_content from initial response"
                            )
                        final_response = (
                            "".join(assistant_content)
                            or "I received the information but encountered an issue generating a response. Could you please try again?"
                        )

                    # Process any modifier adjustments in the final response
                    processed_response = self.process_modifier_adjustment(
                        final_response
                    )

                    # Add the final response to history
                    self.conversation_history.append(
                        {"role": "assistant", "content": processed_response}
                    )

                    # After Lucan's response is generated, publish event to sidecar
                    self._publish_sidecar_event(user_message, processed_response)
                    return processed_response
                else:
                    # No tool results, just return the assistant's text
                    assistant_text = "".join(assistant_content)
                    processed_response = self.process_modifier_adjustment(
                        assistant_text
                    )
                    # After Lucan's response is generated, publish event to sidecar
                    self._publish_sidecar_event(user_message, processed_response)
                    return processed_response

            else:
                # No tool calls, handle as before
                lucan_response = response.content[0].text

                # Process any modifier adjustments
                processed_response = self.process_modifier_adjustment(lucan_response)

                # Add Lucan's response to history
                self.conversation_history.append(
                    {"role": "assistant", "content": processed_response}
                )

                # After Lucan's response is generated, publish event to sidecar
                self._publish_sidecar_event(user_message, processed_response)
                return processed_response

        except Exception as e:
            return f"Error communicating with Lucan: {str(e)}"

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []

    def get_history_length(self) -> int:
        """
        Get the number of messages in the conversation history.
        """
        return len(self.conversation_history)

    def _infer_relationship_type(self, name: str) -> str:
        """
        Infer a relationship type for a given name based on recent conversation context.

        Args:
            name: The name of the person

        Returns:
            Inferred relationship type
        """
        # Look at recent conversation context to infer relationship
        recent_messages = self.conversation_history[-3:]  # Last 3 messages for context
        context_text = " ".join(
            [
                msg.get("content", "")
                if isinstance(msg.get("content"), str)
                else str(msg.get("content", ""))
                for msg in recent_messages
            ]
        ).lower()

        # Common relationship patterns
        if any(
            word in context_text
            for word in ["therapist", "therapy", "counselor", "psychologist"]
        ):
            return "therapist"
        elif any(
            word in context_text
            for word in ["mom", "mother", "dad", "father", "parent"]
        ):
            return "family"
        elif any(word in context_text for word in ["friend", "buddy", "pal"]):
            return "friend"
        elif any(
            word in context_text
            for word in ["boss", "manager", "colleague", "coworker", "work"]
        ):
            return "colleague"
        elif any(
            word in context_text for word in ["doctor", "dr.", "physician", "dentist"]
        ):
            return "doctor"
        elif any(
            word in context_text for word in ["teacher", "professor", "instructor"]
        ):
            return "teacher"
        elif any(
            word in context_text for word in ["dog", "cat", "pet", "puppy", "kitten"]
        ):
            return "pet"
        elif any(
            word in context_text
            for word in [
                "wife",
                "husband",
                "spouse",
                "partner",
                "girlfriend",
                "boyfriend",
            ]
        ):
            return "partner"
        elif any(word in context_text for word in ["son", "daughter", "child", "kid"]):
            return "child"
        elif any(word in context_text for word in ["brother", "sister", "sibling"]):
            return "sibling"
        else:
            return "person"  # default
