import re
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple, Union

from .harmony import HarmonyStreamParser
from ...types import (
    ChatCompletionChunk,
    ChatCompletionChunkDelta,
    CompletionChoice,
    CompletionChunk,
)


class GptOssReasoningParser:
    """Reasoning parser for reasoning model."""

    def __init__(
        self,
        reasoning_content: bool = False,
        reasoning_start_tag: str = "",
        reasoning_end_tag: str = "",
        enable_thinking: bool = True,
    ):
        self.reasoning_content = reasoning_content
        self.reasoning_start_tag = reasoning_start_tag
        self.reasoning_end_tag = reasoning_end_tag
        self.reasoning_regex = re.compile(
            rf"{self.reasoning_start_tag}(.*?){self.reasoning_end_tag}", re.DOTALL
        )
        # enable_thinking can be set to False only for hybrid model
        # e.g. qwen3, which can support both thinking and non-thinking
        self.enable_thinking = enable_thinking
        # GPT-OSS streaming support
        self.harmony_parser: Optional[HarmonyStreamParser] = None
        self.is_gpt_oss_mode: bool = False

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> ChatCompletionChunkDelta:
        """Extract reasoning content from DeepSeek-R1 model output in a streaming fashion.

        Args:
            previous_text (str): The previous accumulated text content.
            current_text (Union[str, ChatCompletionChunk]): The current text chunk or completion chunk.

        Yields:
            str: Extracted reasoning content chunks.
        """
        print("extract_reasoning_content_streaming, previous_text:", previous_text, "current_text:", current_text, "delta_text:", delta_text)
        delta = ChatCompletionChunkDelta()

        # GPT-OSS format detection and handling
        # Check if we're in GPT-OSS mode or if content starts with "analysis"
        if not self.is_gpt_oss_mode:
            if previous_text.startswith("analysis") or delta_text.startswith("analysis"):
                self.is_gpt_oss_mode = True
                self.harmony_parser = HarmonyStreamParser()

        if self.is_gpt_oss_mode:
            if self.harmony_parser is None:
                self.harmony_parser = HarmonyStreamParser()

            # Use HarmonyStreamParser for GPT-OSS format
            reasoning_content = ""
            content = ""

            for seg in self.harmony_parser.feed(delta_text):
                ch, c = seg["channel"], seg["content"]
                print("====== seg channel, content =========")
                print(ch, c)
                if ch == "final":
                    content += c
                elif ch == "analysis":
                    reasoning_content += c
                elif ch == "tool":
                    if content:
                        content += c
                    else:
                        if c != "final":
                            content += c
                    # if not c:
                    #     content += "\n\n"

            delta["reasoning_content"] = reasoning_content if reasoning_content else None
            delta["content"] = content if content else None
            return delta

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.reasoning_start_tag and self.reasoning_start_tag in previous_text:
            if self.reasoning_end_tag and self.reasoning_end_tag in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            elif self.reasoning_end_tag and self.reasoning_end_tag in previous_text:
                # <think> in previous, </think> in previous,
                # <think> in previous, </think> in previous,
                # reasoning content ends
                delta["reasoning_content"] = None
                delta["content"] = delta_text
                return delta
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                delta["reasoning_content"] = delta_text
                delta["content"] = None
                return delta
        elif self.reasoning_start_tag and self.reasoning_start_tag in delta_text:
            start_idx = delta_text.find(self.reasoning_start_tag)
            if self.reasoning_end_tag and self.reasoning_end_tag in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[
                    start_idx + len(self.reasoning_start_tag) : end_idx
                ]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                reasoning_content = delta_text[
                    start_idx + len(self.reasoning_start_tag) :
                ]
                delta["reasoning_content"] = reasoning_content
                delta["content"] = None
                return delta
        else:
            # No <think> in previous or delta, also need to check for </think>.
            # Because the model may have generated </think> without <think>
            # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
            if self.reasoning_end_tag and self.reasoning_end_tag in delta_text:
                # </think> in delta with more tokens,
                # extract reasoning content and content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            elif self.reasoning_end_tag and self.reasoning_end_tag in previous_text:
                # </think> in previous, thinking content ends
                delta["reasoning_content"] = None
                delta["content"] = delta_text
                return delta
            else:
                # no </think> in previous or delta, reasoning content continues
                delta["reasoning_content"] = delta_text
                delta["content"] = None
                return delta

    def extract_reasoning_content(
        self, model_output: Union[str, CompletionChoice]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning content from DeepSeek-R1 model output.

        Args:
            content (str): The model output content to parse.

        Returns:
            Optional[str]: Extracted reasoning content, or None if no reasoning content found.
        """
        if not isinstance(model_output, str):
            model_output = model_output["text"]

        print("model_output:", model_output)

        # GPT-OSS
        if "assistantfinal" in model_output or "assistantcommentary" in model_output:
            parser = HarmonyStreamParser()
            msg = {}

            # Reset fields before parsing
            msg["content"] = ""
            msg["reasoning_content"] = ""
            msg.setdefault("tool_calls", [])

            # Feed original content
            for seg in parser.feed(model_output):
                ch, c = seg["channel"], seg["content"]
                if ch == "final":
                    msg["content"] += c
                elif ch == "analysis":
                    msg["reasoning_content"] += c
                elif ch == "tool":
                    print("add tool from vllm content")
                    msg["content"].append(c)
            return msg["reasoning_content"], msg["content"]
        elif model_output and model_output.startswith("final"):
            model_output = model_output[len("final"):]
            return None, model_output

        # Handle None or empty model_output
        if model_output is None or model_output == "":
            return None, ""

        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if not self.reasoning_end_tag or self.reasoning_end_tag not in model_output:
            return model_output, ""
        else:
            # Add a start token if it's missing to keep compatibility.
            if not self.reasoning_start_tag or self.reasoning_start_tag not in model_output:
                model_output = f"{self.reasoning_start_tag}{model_output}"
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]

            end_index = len(
                f"{self.reasoning_start_tag}{reasoning_content}{self.reasoning_end_tag}"
            )
            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return reasoning_content, ""
            return reasoning_content, final_output

    def check_content_parser(self) -> bool:
        """Check if the parser should extract reasoning content.

        Returns:
            bool: True if reasoning content should be extracted, False otherwise
        """
        if self.is_enable_thinking():
            return self.reasoning_content
        return False

    def _create_chat_completion_chunk(
        self, chunk: Union[Dict[str, Any], CompletionChunk], content: str
    ) -> ChatCompletionChunk:
        """Helper method to create a ChatCompletionChunk with specified content.

        Args:
            chunk: The original chunk to copy metadata from
            content: The content to include in the chunk

        Returns:
            ChatCompletionChunk: A new chat completion chunk
        """
        return ChatCompletionChunk(
            id="chat" + chunk["id"],
            model=chunk["model"],
            created=chunk["created"],
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        )

    def _create_completion_chunk(
        self, chunk: Union[Dict[str, Any], CompletionChunk], text: str
    ) -> CompletionChunk:
        """Helper method to create a CompletionChunk with specified text.

        Args:
            chunk: The original chunk to copy metadata from
            text: The text to include in the chunk

        Returns:
            CompletionChunk: A new completion chunk
        """
        return CompletionChunk(
            id=chunk["id"],
            model=chunk["model"],
            created=chunk["created"],
            object="text_completion",
            choices=[
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        )

    def is_enable_thinking(self):
        from .core import chat_context_var

        context = chat_context_var.get({})
        return context.get("enable_thinking", self.enable_thinking)

    async def prepare_reasoning_content_streaming(
        self, chunks: AsyncGenerator[CompletionChunk, None]
    ):
        """Process the chunks from model output, check if the first chunk contains reasoning_start_tag,
        if not, add a chunk with the tag at the beginning.

        Args:
            chunks (AsyncGenerator[CompletionChunk, None]): Chunks from model output

        Yields:
            AsyncGenerator[CompletionChunk, None]: Processed chunks
        """

        # If reasoning_start_tag is not set, or disable thinking for hybrid model like qwen3,
        # yield chunks as is
        if not self.reasoning_start_tag or not self.is_enable_thinking():
            async for chunk in chunks:
                yield chunk
            return

        # If chunks is empty, return
        if not chunks:
            return

        # Flag to identify the first chunk
        is_first_chunk = True

        async for chunk in chunks:
            if is_first_chunk:
                # Reset the flag after processing the first chunk
                is_first_chunk = False
                choices = chunk.get("choices")
                if not choices or not choices[0]:
                    continue
                if (
                    chunk.get("object") == "chat.completion.chunk"
                    and "delta" in choices[0]
                ):
                    # For chat completion chunks with delta format
                    delta = choices[0].get("delta")
                    if delta is None:
                        continue
                    assert isinstance(delta, dict)
                    text = delta.get("content")
                    if not text:
                        continue
                    # If the first chunk doesn't contain the reasoning_start_tag
                    if self.reasoning_start_tag not in text:
                        # Create and yield chunks with reasoning_start_tag and newline
                        yield self._create_chat_completion_chunk(
                            chunk, f"{self.reasoning_start_tag}\n"
                        )
                else:
                    # For standard completion chunks
                    text = choices[0].get("text")
                    if not text:
                        continue
                    # If the first chunk doesn't contain the reasoning_start_tag
                    if self.reasoning_start_tag not in text:
                        # Create and yield chunks with reasoning_start_tag and newline
                        yield self._create_completion_chunk(
                            chunk, f"{self.reasoning_start_tag}\n"
                        )
                # Yield the original first chunk
                yield chunk
            else:
                # For non-first chunks, yield directly
                yield chunk

    def prepare_reasoning_content_sync(self, chunks: Iterator[CompletionChunk]):
        """Process the chunks from model output, check if the first chunk contains reasoning_start_tag,
        if not, add a chunk with the tag at the beginning. This is a synchronous version of
        prepare_reasoning_content_streaming.

        Args:
            chunks (Iterator[CompletionChunk]): Chunks from model output

        Returns:
            Iterator[CompletionChunk]: Processed chunks
        """
        # If reasoning_start_tag is not set, or disable thinking for hybrid model like qwen3,
        # yield chunks as is
        if not self.reasoning_start_tag or not self.is_enable_thinking():
            for chunk in chunks:
                yield chunk
            return

        # Flag to identify the first chunk
        is_first_chunk = True

        for chunk in chunks:
            if is_first_chunk:
                # Reset the flag after processing the first chunk
                is_first_chunk = False
                choices = chunk.get("choices")
                if not choices or not choices[0]:
                    continue
                if (
                    chunk.get("object") == "chat.completion.chunk"
                    and "delta" in choices[0]
                ):
                    # For chat completion chunks with delta format
                    delta = choices[0].get("delta")
                    if delta is None:
                        continue
                    assert isinstance(delta, dict)
                    text = delta.get("content")
                    if text is None:
                        continue
                    # If the first chunk doesn't contain the reasoning_start_tag
                    if self.reasoning_start_tag not in text:
                        # Create and yield chunks with reasoning_start_tag and newline
                        yield self._create_chat_completion_chunk(
                            chunk, f"{self.reasoning_start_tag}\n"
                        )
                else:
                    # For standard completion chunks
                    text = choices[0].get("text")
                    if text is None:
                        continue
                    # If the first chunk doesn't contain the reasoning_start_tag
                    if self.reasoning_start_tag not in text:
                        # Create and yield chunks with reasoning_start_tag and newline
                        yield self._create_completion_chunk(
                            chunk, f"{self.reasoning_start_tag}\n"
                        )
                # Yield the original first chunk
                yield chunk
            else:
                # For non-first chunks, yield directly
                yield chunk

    def prepare_reasoning_content(self, completion):
        """Ensures that the model output string starts with the reasoning_start_tag.

        If the model_output is not a string (e.g., CompletionChoice), it extracts
        the text content. If the reasoning_start_tag is not found in the text,
        it prepends the tag to the text.

        Args:
            completion: The completion object containing model output,
                which can be either a chat completion or a standard completion.
        """
        if not self.reasoning_start_tag or not self.is_enable_thinking():
            return completion

        if completion.get("object") == "chat.completion" and completion.get("choices"):
            text = completion["choices"][0]["message"]["content"]
            if self.reasoning_start_tag not in text:
                text = f"{self.reasoning_start_tag}\n{text}"
            completion["choices"][0]["message"]["content"] = text
            return completion

        text = completion["choices"][0]["text"]
        if self.reasoning_start_tag not in text:
            text = f"{self.reasoning_start_tag}\n{text}"
        completion["choices"][0]["text"] = text
        return completion

    def prepare_first_reasoning_content_chunk(
        self,
        chunk: CompletionChunk,
    ) -> List[ChatCompletionChunk]:
        """Prepares the first chunk of a completion by adding reasoning_start_tag if needed.

        This function checks if the first chunk contains the reasoning_start_tag. If not,
        it creates two new chunks containing the reasoning_start_tag and a newline character
        that will be inserted before the original chunk.

        Args:
            chunk (CompletionChunk): The first chunk of a completion to check and possibly modify

        Returns:
            List[ChatCompletionChunk]: A list of new chunks to insert before the original chunk,
                or an empty list if no modification is needed
        """
        chunks: List[ChatCompletionChunk] = []
        if not self.reasoning_start_tag or not self.is_enable_thinking():
            return chunks

        choices = chunk.get("choices")
        if not choices or not choices[0]:
            return chunks
        text = choices[0].get("text")
        if not text:
            return chunks

        if self.reasoning_start_tag not in text:
            # Create chunks with reasoning_start_tag and newline
            chunks.append(
                self._create_chat_completion_chunk(
                    chunk, f"{self.reasoning_start_tag}\n"
                )
            )

        return chunks
