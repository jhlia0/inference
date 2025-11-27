import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("gpt-oss")
class GptOssToolParser(ToolParser):
    """
    Tool parser implementation for GPT-OSS model.

    This parser handles the specific format used by GPT-OSS for tool calls,
    which uses special channel-based tags for tool calls and responses.

    The format is:
    <|start|>assistant to=functions.{function_name}<|channel|>commentary json<|message|>{arguments}<|call|>
    """

    def __init__(self):
        """
        Initialize the GPT-OSS tool parser.

        Sets up the special tokens and regex patterns used for parsing
        GPT-OSS model outputs containing tool calls.
        """
        super().__init__()

        # Sentinel tokens for streaming mode
        self.tool_call_start_pattern = r" to=functions\."
        self.tool_call_end_token = "<|call|>"
        self.message_start_token = "<|message|>"

        # Regex patterns for parsing tool calls
        # Pattern matches: assistantcommentary to=functions.{name} json{args}
        # Note: [\w-]+ allows function names with hyphens
        self.tool_call_complete_regex = re.compile(
            r" to=functions\.([\w-]+)\s+json(\{.*?\})(?:\s*\\?\s*$|\s*\\?\s*(?=assistantcommentary))",
            re.DOTALL,
        )
        # Pattern for incomplete tool calls (for streaming)
        self.tool_call_incomplete_regex = re.compile(
            r" to=functions\.([\w-]+)\s+json(\{.*?)$",
            re.DOTALL,
        )

    def _parse_tool_call_content(
        self, function_name: str, arguments_str: str
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse a single tool call's content.

        Args:
            function_name (str): The name of the function.
            arguments_str (str): The arguments as a JSON string.

        Returns:
            Tuple of (content, function_name, arguments).
            On error, returns (original_str, None, None).
        """
        try:
            # Clean up the arguments string
            arguments_str = arguments_str.strip()
            if not arguments_str:
                # Empty arguments
                return (None, function_name, {})

            # Parse JSON arguments
            arguments = json.loads(arguments_str, strict=False)
            return (None, function_name, arguments)
        except Exception as e:
            logger.error(
                "Failed to parse gpt-oss tool call. Function: %s, Args: %s, Error: %s",
                function_name,
                arguments_str,
                e,
            )
            return (f"functions.{function_name}: {arguments_str}", None, None)

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Parses the model output to find tool calls in the GPT-OSS format,
        extracting function names and arguments.

        Args:
            model_output (str): The complete output string from the model.

        Returns:
            List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A list of tuples where each tuple contains:
            - content (str or None): Raw content if parsing failed, None if successful
            - function_name (str or None): Name of the function to call
            - arguments (dict or None): Function arguments

        Example:
            >>> parser = GptOssToolParser()
            >>> output = '<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "Beijing"}<|call|>'
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        print("======= gptoss tool parser ==========")
        print(model_output)
        # Check if there are any tool calls in the output
        if not re.search(self.tool_call_start_pattern, model_output):
            return [(model_output, None, None)]

        try:
            # Find all complete tool calls
            matches = self.tool_call_complete_regex.findall(model_output)

            if not matches:
                # No complete tool calls found, return original output
                return [(model_output, None, None)]

            results: List[
                Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
            ] = []

            for function_name, arguments_str in matches:
                result = self._parse_tool_call_content(function_name, arguments_str)
                results.append(result)

            return results

        except Exception as e:
            logger.error(
                "Can't parse gpt-oss tool call output: %s. Error: %s",
                model_output,
                e,
            )
            return [(model_output, None, None)]

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from streaming output.

        Processes streaming model output to detect and extract tool calls
        as they are being generated. Handles incomplete tool calls and
        determines when a complete tool call is available.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.

        Returns:
            Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A tuple containing:
            - content (str or None): Text content or None for tool calls
            - function_name (str or None): Name of the function to call
            - arguments (dict or None): Function arguments
            Returns None if no complete tool call is ready.

        Note:
            This method is designed to work with GPT-OSS's streaming output format
            and handles partial tool calls during generation.
        """
        print("======= gptoss stream tool parser ==========")
        print("previous_text:", previous_text, "current_text:", current_text, "delta_text:", delta_text)
        try:
            # Check if current output contains tool call pattern
            if not re.search(self.tool_call_start_pattern, current_text):
                # No tool call, return delta as regular content
                print("f1")
                return (delta_text, None, None)

            # Check for complete tool calls
            complete_matches = self.tool_call_complete_regex.findall(current_text)
            if complete_matches:
                # Get the last complete tool call
                function_name, arguments_str = complete_matches[-1]

                # Check if this is a new tool call (not already processed)
                # by checking if the previous text had this complete tool call
                if previous_text:
                    prev_complete = self.tool_call_complete_regex.findall(
                        previous_text[-1]
                    )
                    if (
                        prev_complete
                        and (function_name, arguments_str) == prev_complete[-1]
                    ):
                        print("f2")
                        # This tool call was already processed
                        return None

                # Parse and return the tool call
                result = self._parse_tool_call_content(function_name, arguments_str)
                print("f3")
                return result

            # If we have incomplete tool calls, don't return anything yet
            # (wait for completion)
            if re.search(
                self.tool_call_start_pattern, current_text
            ) and self.tool_call_incomplete_regex.search(current_text):
                print("f4")
                return None
            print("f5")
            # Default: return delta as content
            return (delta_text, None, None)

        except Exception as e:
            logger.error("Error in gpt-oss streaming tool call extraction: %s", e)
            return (delta_text, None, None)
