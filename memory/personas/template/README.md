# Persona Template

This template makes it easy to create new personas for Lucan. The goal for this system is to eventually be able to swap between multiple user-defined personas automatically to be of most use to the user.

## Quick Start

1. **Copy the template:**
   ```bash
   cp -r memory/personas/template memory/personas/your_persona_name
   ```

2. **Edit the files:**
   - Open `personality.txt` and replace all the placeholder text with your persona's details
   - Adjust `modifiers.txt` if your persona needs different starting values (most should stay at 0)
   - Delete this README.md file when you're done

3. **Test your persona:**
   ```bash
   python main.py --persona memory/personas/your_persona_name
   ```

## What Each File Does

- **`personality.txt`** - Defines who the persona is, how they communicate, and what they focus on
- **`modifiers.txt`** - Sets the starting values for personality adjustments (warmth, challenge, etc.)

## Tips for Good Personas

1. **Start with purpose** - What specific need does this persona fill?
2. **Be specific** - "analytical" is better than "smart"
3. **Keep it coherent** - All traits should work together
4. **Test mentally** - Imagine having a conversation with this persona

## Examples of Different Persona Types

- **Coach**: Motivational, action-oriented, holds you accountable
- **Therapist**: Empathetic, reflective, explores emotions
- **Analyst**: Logical, data-driven, breaks down problems
- **Creative**: Imaginative, experimental, suggests alternatives

## Need Help?

Check the examples and comments in the template files - they include detailed guidance and examples of different persona types. 