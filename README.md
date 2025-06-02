# Lucan - Your AI Friend

A smart CLI chat application featuring an AI friend who adapts to your communication style and remembers your relationships. Lucan is your loyal, clear-eyed AI companion who helps you move forward without judgment.

## Features

**Adaptive Personality** - Lucan can adjust their communication style based on your feedback  
**Relationship Memory** - Remembers details about your friends, family, and pets  
**Forward-Focused** - Helps you make progress rather than dwelling on problems  
**Customizable** - Fine-tune personality traits through simple text files  
**Debug Mode** - See how the AI processes your conversations

## Quick Start

1. **Set up your API key:**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

2. **Install dependencies:**
```bash
uv sync
```

3. **Start chatting:**
```bash
python main.py
```

## Setup

### Environment Variables

Create a `.env` file with your Anthropic API key:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
ANTHROPIC_KEY=your_actual_api_key_here
```

**Note:** The project requires an Anthropic API key. OpenAI keys are not currently used, but might be implemented in the future for more robust usage of tools.

### Requirements

- Python 3.13+
- UV package manager (recommended) or pip
- Anthropic API key

## Usage

### Basic Commands

- Type messages to chat with Lucan
- `/help` - Show available commands  
- `/clear` - Reset conversation history
- `quit`, `exit`, `bye` - Exit the chat

### Adaptive Features

**Dynamic Style Adjustment:**
- Tell Lucan to be "more direct", "less verbose", or "warmer"
- Lucan will adjust their communication style and remember your preferences
- Changes are applied automatically and saved for future conversations

**Relationship Memory:**
- Mention friends, family, or pets in conversation
- Lucan will remember details about them naturally
- No need to explicitly ask - this happens automatically in conversation

### Debug Mode

Run with debug mode to see how Lucan processes your conversations:
```bash
python main.py --debug
```

This shows:
- Current personality modifier values
- The system prompt being used
- How personality adjustments are being processed

## Personality System

The AI personality is built from multiple components:

### Core Files

- `memory/personas/lucan/personality.txt` - Core traits and communication style
- `memory/personas/lucan/modifiers.txt` - Numeric adjustments for behavior

### Dynamic Modifiers

Lucan can adjust these aspects of their personality in real-time:

- **Warmth** (-3 to +3) - How supportive vs. direct
- **Challenge** (-3 to +3) - How much they push you vs. accept
- **Verbosity** (-3 to +3) - Response length and detail
- **Emotional Depth** (-3 to +3) - How much they explore feelings
- **Structure** (-3 to +3) - How organized vs. conversational

### Relationship Tracking

Relationship notes are stored in `memory/relationships/` as .txt files:
- Automatic note-taking about people you mention
- Remembers relationship types (friend, family, colleague, pet, etc.)
- Tracks updates and changes over time

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Dependencies

The project uses:
- **anthropic** - Claude AI integration
- **rich** - Beautiful terminal output  
- **pyyaml** - Configuration file parsing
- **python-dotenv** - Environment variable management
- **ruff** - Code formatting and linting

## How It Works

1. **Personality Loading** - Loads base personality from text files
2. **Dynamic Adjustment** - Processes user feedback to adjust communication style
3. **Relationship Tracking** - Automatically notes important people mentioned
4. **Context Integration** - Combines personality, relationships, and conversation history
5. **Adaptive Response** - Generates responses that match your preferred communication style

Lucan learns your communication preferences and remembers your relationships to provide increasingly personalized and helpful conversations over time.

---

*Built with Claude AI and designed to be your thoughtful, adaptive AI companion.*
