# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    # === CUSTOM: Handle models with baked-in system prompts (e.g., OLMo3, Gemma-3) ===
    # Detect system prompt by searching for where the user message starts.
    # Use tools=None to match the fixed agent loop behavior (pass None, not empty list)
    try:
        # Use tools=None to match fixed agent loop behavior
        # We now pass None instead of [] when there are no tools (to avoid OLMo3's long prompt)
        kwargs = {"tools": None}
        kwargs.update(apply_chat_template_kwargs)

        # Generate a sample prompt with auto-generated system prompt
        full_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            add_generation_prompt=True,
            tokenize=True,
            **kwargs
        )

        # Search for the <|im_start|>user pattern to find where system prompt ends
        user_start_pattern = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        for i in range(len(full_tokens) - len(user_start_pattern) + 1):
            if full_tokens[i:i + len(user_start_pattern)] == user_start_pattern:
                # Found it! Return everything before the user message
                return full_tokens[:i]

        # If not found, return empty (no system prompt detected)
        return []
    except Exception:
        return []
    # === END CUSTOM ===


def extract_system_prompt_and_generation(tokenizer):
    # === CUSTOM: Handle models with strict role alternation requirements ===
    # IMPORTANT: Use tools=None to match fixed agent loop behavior (see initialize_system_prompt)
    try:
        kwargs = {"tools": None}  # Use None to avoid triggering OLMo3's long system prompt
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            **kwargs
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            **kwargs
        )
        # get system prompt tokens
        system_prompt = token1[: -(len(token2) - len(token1))] if len(token2) > len(token1) else []
        # get generate prompt tokens
        token3 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
            **kwargs
        )
        generate_prompt = token3[len(token1) :]
        return system_prompt, generate_prompt
    except Exception:
        # Fallback for models that don't support this pattern
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
        generate_prompt = token3[len(token1) :]
        return [], generate_prompt
    # === END CUSTOM ===


def remove_system_prompt_from_text(formatted_prompt: str, tokenizer) -> str:
    """
    Remove system prompt from a formatted chat template string.
    Useful for evaluation scripts that use apply_chat_template(tokenize=False).

    Args:
        formatted_prompt: The formatted prompt string from apply_chat_template
        tokenizer: The tokenizer used to format the prompt

    Returns:
        The prompt with system prompt removed, or original if no system prompt detected
    """
    try:
        # Get system prompt tokens
        sys_prompt_tokens = initialize_system_prompt(tokenizer)
        if not sys_prompt_tokens:
            return formatted_prompt

        # Decode system prompt to find what text to remove
        sys_prompt_text = tokenizer.decode(sys_prompt_tokens, skip_special_tokens=False)

        # Remove system prompt from the beginning of the formatted prompt
        if formatted_prompt.startswith(sys_prompt_text):
            return formatted_prompt[len(sys_prompt_text):]

        # Fallback: search for the system prompt pattern
        # Look for <|im_start|>user as the start of the actual user message
        user_start = formatted_prompt.find("<|im_start|>user")
        if user_start > 0:
            return formatted_prompt[user_start:]

        return formatted_prompt
    except Exception:
        return formatted_prompt
