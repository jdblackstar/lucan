import json
import os
import re
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv

from .loader import Lucan
from .relationships import RelationshipManager

# Load environment variables from .env file
load_dotenv()


class LucanChat:
    """
    Core chat functionality for the Lucan AI friend.
    """

    def __init__(self, persona_path: str | Path, debug: bool = False):
        """
        Initialize the chat with a persona from the given path.

        Args:
            persona_path: Path to the persona directory containing personality.txt and modifiers.txt
            debug: Whether to enable debug output for development
        """
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.lucan = Lucan(Path(persona_path))
        self.conversation_history: List[Dict[str, str]] = []
        self.debug = debug

        # Initialize relationship manager
        relationships_dir = Path(persona_path).parent.parent / "relationships"
        self.relationship_manager = RelationshipManager(relationships_dir)

        self.system_prompt = self._build_system_prompt()

    def _define_tools(self) -> List[Dict]:
        """
        Define the tools available to Claude.

        Returns:
            List of tool definitions for the Anthropic API
        """
        return [
            {
                "name": "add_relationship_note",
                "description": "Add or update information about someone the user mentions. Use this tool when the user shares important information about people in their life, such as: relationship changes (breakups, marriages), life updates (new jobs, moves, health issues), new people they mention, or any significant details worth remembering. Examples:'My girlfriend and I broke up', 'My mom got a new job', 'I have a new therapist named Dr. Smith', 'My friend Sarah is getting married'. Don't announce when you're using this tool - just naturally remember the information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person's name"},
                        "relationship_type": {
                            "type": "string",
                            "description": "Their relationship to the user (e.g., friend, family, colleague, therapist, pet, partner, etc.)",
                        },
                        "note": {
                            "type": "string",
                            "description": "What to remember about this person (updates, context, interests, concerns, relationship changes, etc.)",
                        },
                    },
                    "required": ["name", "relationship_type", "note"],
                },
            },
            {
                "name": "get_relationship_notes",
                "description": "Look up information about someone the user asks about. Use this tool when the user asks questions like 'Do you know my mom?', 'Tell me about Sarah', 'Do you remember my therapist?', 'What do you know about my friend John?', etc. You can search by either a person's name (like 'Sarah') or by relationship type (like 'mom', 'therapist', 'friend'). Don't announce when you're using this tool - just naturally recall the information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The person's name OR their relationship type (e.g., 'Sarah', 'mom', 'therapist', 'friend', 'dog')",
                        }
                    },
                    "required": ["name"],
                },
            },
        ]

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
        if tool_name == "add_relationship_note":
            name = tool_input.get("name", "")
            relationship_type = tool_input.get("relationship_type", "")
            note = tool_input.get("note", "")

            if name and note:
                success = self.relationship_manager.add_note(
                    name, relationship_type, note
                )
                if self.debug:
                    print(
                        f"[DEBUG] Added note for {name} ({relationship_type}): {note}"
                    )
                return {"success": success, "message": f"Added note for {name}"}
            else:
                return {"success": False, "message": "Missing required fields"}

        elif tool_name == "get_relationship_notes":
            name = tool_input.get("name", "")
            if name:
                # First try direct name lookup
                notes = self.relationship_manager.get_notes(name)
                if self.debug:
                    if notes:
                        print(
                            f"[DEBUG] Retrieved {len(notes['notes'])} notes for {name}"
                        )
                    else:
                        print(f"[DEBUG] No notes found for {name}")

                if notes:
                    return {
                        "success": True,
                        "name": notes["name"],
                        "relationship": notes["relationship"],
                        "notes": notes["notes"],
                    }
                else:
                    # If no direct name match, try searching by relationship type
                    if self.debug:
                        print(f"[DEBUG] Trying relationship type search for '{name}'")

                    relationship_results = (
                        self.relationship_manager.find_by_relationship_type(name)
                    )

                    if relationship_results:
                        if self.debug:
                            names = [r["name"] for r in relationship_results]
                            print(
                                f"[DEBUG] Found {len(relationship_results)} people with relationship '{name}': {names}"
                            )

                        # Return the first match (could be enhanced to return all matches)
                        first_match = relationship_results[0]
                        return {
                            "success": True,
                            "name": first_match["name"],
                            "relationship": first_match["relationship"],
                            "notes": first_match["notes"],
                            "found_by": "relationship_type",  # Indicate how it was found
                        }
                    else:
                        if self.debug:
                            print(
                                f"[DEBUG] No relationship type matches found for '{name}'"
                            )

                        # If no notes found anywhere, create a basic note for this person
                        relationship_type = self._infer_relationship_type(name)
                        success = self.relationship_manager.add_note(
                            name, relationship_type, "First mentioned in conversation"
                        )
                        if self.debug and success:
                            print(
                                f"[DEBUG] Created initial note for {name} as {relationship_type}"
                            )
                        return {
                            "success": True,
                            "name": name,
                            "relationship": relationship_type,
                            "notes": ["First mentioned in conversation"],
                        }
            else:
                return {"success": False, "message": "Name is required"}

        return {"success": False, "message": f"Unknown tool: {tool_name}"}

    def _process_modifier_adjustment(self, response: str) -> str:
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

    def send_message(self, user_message: str) -> str:
        """
        Send a message to Lucan and get a response.

        Args:
            user_message: The user's message

        Returns:
            Lucan's response
        """
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

                    # Get the follow-up response after tool execution
                    follow_up_response = self.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        system=current_system_prompt,
                        messages=self.conversation_history.copy(),
                        tools=tools,
                    )

                    # Extract the final response text
                    final_response = ""
                    if follow_up_response.content:
                        for content_block in follow_up_response.content:
                            if content_block.type == "text":
                                final_response += content_block.text

                    # Process any modifier adjustments in the final response
                    processed_response = self._process_modifier_adjustment(
                        final_response
                    )

                    # Add the final response to history
                    self.conversation_history.append(
                        {"role": "assistant", "content": processed_response}
                    )

                    return processed_response
                else:
                    # No tool results, just return the assistant's text
                    assistant_text = "".join(assistant_content)
                    processed_response = self._process_modifier_adjustment(
                        assistant_text
                    )
                    return processed_response

            else:
                # No tool calls, handle as before
                lucan_response = response.content[0].text

                # Process any modifier adjustments
                processed_response = self._process_modifier_adjustment(lucan_response)

                # Add Lucan's response to history
                self.conversation_history.append(
                    {"role": "assistant", "content": processed_response}
                )

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
