# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import TYPE_CHECKING, AsyncGenerator, Dict, Union

if TYPE_CHECKING:
    from ...types import ChatCompletion, ChatCompletionChunk


class HarmonyStreamParser:
    MARKER = "assistantfinal"
    COMMENTARY_MARKER = "assistantcommentary"

    def __init__(self):
        # Current channel: either 'analysis', 'final', or None if not started yet
        self.current_channel = None
        # Buffer for accumulating text when looking for 'assistantfinal' marker
        self.buffer = ""

    def _find_partial_marker_length(self, text):
        """
        Find the length of partial marker at the end of text.
        Returns the length of the longest partial match, or 0 if no match.
        """
        # Check both markers
        for marker in [self.MARKER, self.COMMENTARY_MARKER]:
            # Check from longest to shortest prefix
            for i in range(len(marker), 0, -1):
                prefix = marker[:i]
                if text.endswith(prefix):
                    return i
        return 0

    def _emit_safe_content(self, segments, channel):
        """
        Emit safe content from buffer (everything except potential partial marker at end).
        Keep only the partial marker portion in buffer.
        """
        if not self.buffer:
            return

        partial_len = self._find_partial_marker_length(self.buffer)
        if partial_len > 0:
            # Emit everything before the partial marker
            safe_content = self.buffer[:-partial_len]
            if safe_content:
                segments.append({"channel": channel, "content": safe_content})
            # Keep only the partial marker in buffer
            self.buffer = self.buffer[-partial_len:]
        else:
            # No partial marker, emit everything
            segments.append({"channel": channel, "content": self.buffer})
            self.buffer = ""

    def feed(self, text):
        """
        Feed a chunk of text into the parser and return parsed segments.

        Each segment is a dict:
        {
            "channel": "analysis" | "final",
            "content": <string>
        }

        The parser detects 'assistantfinal' markers inside reasoning text,
        splits the reasoning and final content correctly, and switches the channel.
        """
        segments = []

        # If we are currently in 'analysis' mode
        if self.current_channel == "analysis":
            # Add text to buffer and check for 'assistantfinal' marker
            self.buffer += text
            if self.MARKER in self.buffer:
                # Split reasoning and final content
                before, after = self.buffer.split(self.MARKER, 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                # Switch to final channel
                self.current_channel = "final"
                self.buffer = ""
                if after:
                    segments.append({"channel": "final", "content": after})
                return segments
            elif self.COMMENTARY_MARKER in self.buffer:
                # Split at commentary marker
                before, after = self.buffer.split(self.COMMENTARY_MARKER, 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                self.current_channel = "tool"
                self.buffer = ""
                return segments
            else:
                # Emit safe content and keep potential partial marker in buffer
                self._emit_safe_content(segments, "analysis")
                return segments

        # if self.current_channel == "commentary":


        # If we are currently in 'final' mode
        if self.current_channel == "final":
            # Check if this is actually a new message starting with 'analysis'
            if text.startswith("analysis"):
                # Reset parser state for new message
                self.current_channel = None
                self.buffer = ""
                # Re-process this text with the new state
                return self.feed(text)
            else:
                segments.append({"channel": "final", "content": text})
                return segments
        
        # If we are currently in 'tool' mode
        if self.current_channel == "tool":
            # Check if this is actually a new message starting with 'analysis'
            if text.startswith("analysis"):
                # Reset parser state for new message
                self.current_channel = None
                self.buffer = ""
                # Re-process this text with the new state
                return self.feed(text)
            else:
                segments.append({"channel": "tool", "content": text})
                return segments

        # If no channel has been started yet
        if text.startswith("analysis"):
            self.current_channel = "analysis"
            rest = text[len("analysis"):]
            if self.MARKER in rest:
                # Split immediately if marker is found in the first chunk
                before, after = rest.split(self.MARKER, 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                self.current_channel = "final"
                if after:
                    segments.append({"channel": "final", "content": after})
            elif self.COMMENTARY_MARKER in rest:
                # Split immediately if marker is found in the first chunk
                before, after = rest.split(self.COMMENTARY_MARKER, 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                self.current_channel = "tool"
            else:
                # Start buffering for potential marker
                self.buffer = rest
                # Emit safe content, keep potential partial marker in buffer
                self._emit_safe_content(segments, "analysis")
        elif text.startswith(self.MARKER):
            self.current_channel = "final"
            rest = text[len(self.MARKER):]
            if rest:
                segments.append({"channel": "final", "content": rest})
        elif text.startswith("final"):
            self.current_channel = "final"
            rest = text[len("final"):]
            if rest:
                segments.append({"channel": "final", "content": rest})

        return segments


async def async_stream_harmony_chat_completion(
    chunks: Union[
        "ChatCompletion",
        AsyncGenerator["ChatCompletionChunk", None],
    ],
) -> AsyncGenerator["ChatCompletion", None]:
    """
    Parse Harmony-formatted content from either a full ChatCompletion (non-streaming)
    or an async stream of ChatCompletionChunk (streaming), using the HarmonyStreamParser defined in this file.

    Yields parsed objects incrementally.
    """

    # --- Non-streaming: ChatCompletion ---
    if isinstance(chunks, dict) and chunks.get("object") == "chat.completion":
        out_data = deepcopy(chunks)
        for choice in out_data["choices"]:
            parser = HarmonyStreamParser()
            msg = choice["message"]

            # Backup original content & reasoning
            original_content = msg.get("content") or ""
            original_reasoning = msg.get("reasoning_content") or ""

            # Reset fields before parsing
            msg["content"] = ""
            msg["reasoning_content"] = ""
            msg.setdefault("tool_calls", [])

            # Feed original content
            for seg in parser.feed(original_content):
                ch, c = seg["channel"], seg["content"]
                if ch == "final":
                    msg["content"] += c
                elif ch == "analysis":
                    msg["reasoning_content"] += c
                elif ch == "tool":
                    msg["tool_calls"].append(c)

            # Feed original reasoning_content
            for seg in parser.feed(original_reasoning):
                ch, c = seg["channel"], seg["content"]
                if ch == "analysis":
                    msg["reasoning_content"] += c
                elif ch == "tool":
                    msg["tool_calls"].append(c)
                elif ch == "final":
                    msg["content"] += c

            # Clean up reasoning_content: set to None if no reasoning content was parsed
            if not msg["reasoning_content"] and not original_reasoning:
                msg["reasoning_content"] = None  # type: ignore

        yield out_data

    else:
        # Streaming: handle async generator
        parsers_per_choice = {}
        async for chunk in chunks:  # type: ignore
            out_chunk = {  # type: ignore
                "id": chunk["id"],
                "model": chunk["model"],
                "object": chunk["object"],
                "created": chunk["created"],
                "choices": [],
            }
            for i, choice in enumerate(chunk["choices"]):
                delta = choice.get("delta", {})
                # Read from both content and reasoning_content fields
                # because upstream parsers may put text in reasoning_content
                text = delta.get("content") or delta.get("reasoning_content") or ""  # type: ignore

                if i not in parsers_per_choice:
                    parsers_per_choice[i] = HarmonyStreamParser()

                # Feed text to parser and collect current delta only
                curr_delta: Dict[str, object] = {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                }

                for seg in parsers_per_choice[i].feed(text):
                    ch = seg["channel"]
                    c = seg["content"]
                    if ch == "final":
                        curr_delta["content"] += c  # type: ignore
                    elif ch == "analysis":
                        curr_delta["reasoning_content"] += c  # type: ignore
                    elif ch == "tool":
                        curr_delta["tool_calls"].append(c)  # type: ignore

                # Only include tool_calls if there are actual tool calls
                if not curr_delta["tool_calls"]:
                    del curr_delta["tool_calls"]

                if curr_delta["reasoning_content"]:
                    if not curr_delta["content"]:
                        curr_delta["content"] = None

                elif curr_delta["content"]:
                    if not curr_delta["reasoning_content"]:
                        curr_delta["reasoning_content"] = None

                elif (
                    choice.get("finish_reason") is not None
                    and not curr_delta["reasoning_content"]
                ):
                    # For the final chunk, if there's no new reasoning content,
                    # don't include empty reasoning_content to avoid clearing existing state
                    curr_delta["reasoning_content"] = None

                out_chunk["choices"].append(  # type: ignore
                    {
                        "index": i,
                        "delta": curr_delta,
                        "finish_reason": choice.get("finish_reason"),
                    }
                )

            # Only yield if we have either content or reasoning_content
            has_content = any(
                choice["delta"].get("content")  # type: ignore
                or choice["delta"].get("reasoning_content")  # type: ignore
                or choice.get("finish_reason") is not None  # type: ignore
                for choice in out_chunk["choices"]  # type: ignore
            )
            if has_content:
                yield out_chunk  # type: ignore
