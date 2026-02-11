"""
JSON encoding/decoding and file operations.

This module provides utilities for JSON processing including compact
encoding and caching for efficient file operations.
"""

import json
import json5
import logging
import os

import orjson

from ..types import Conversation


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 7000
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 1000
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"

        # ensure keys are converted to strings
        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{json.dumps(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )

        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        self.indentation_level -= 1

        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        if isinstance(o, list):
            if all(isinstance(el, str) for el in o) and len(str(o)) > 200:
                return False

        # we allow lists of ints to be printed on a single line, no matter how long,
        # otherwise containers are put on multiple lines if they are too long.
        # Usually ints are mainly used for token ids, which become very long for some prompts.
        return (
            self._primitives_only(o)
            and (
                all(isinstance(el, int) for el in o)
                or (len(o) <= self.MAX_ITEMS and len(str(o)) - 2 <= self.MAX_WIDTH)
            )
        )

    def _primitives_only(self, o: list | tuple | dict):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )


JSON_CACHE = {}


def cached_json_load(path):
    mod_time = os.path.getmtime(path)
    if path in JSON_CACHE:
        if JSON_CACHE[path][0] == mod_time:
            return JSON_CACHE[path][1]
        del JSON_CACHE[path]
    # Get the last modification time of the file
    # Return both the data and the modification time
    data = orjson.loads(open(path, "rb").read())
    JSON_CACHE[path] = (mod_time, data)
    return data


# basic json parsing

def safe_parse_json_responses(convs: list[Conversation], attempt: int, dummy_result: None | dict = None):
    """ if dummy_result is None, None will be returned on parse error"""

    responses = [conv[-1]["content"] for conv in convs]
    jsons: list[dict | None] = [None] * len(responses)
    parse_unsuccessful_map = [False] * len(responses)
    error_messages = []

    for i, response in enumerate(responses):
        try:
            jsons[i] = parse_json_response(response)
        except Exception as e:
            logging.error(f"Error in parsing JSON attempt {attempt}: {e} Response: {response} Conv: {convs[i]}")
            parse_unsuccessful_map[i] = True
            error_messages.append(str(e))
            if dummy_result:
                logging.info("Falling back to dummy result.")
                jsons[i] = dummy_result

    return jsons, parse_unsuccessful_map, error_messages


def parse_json_response(output: str) -> dict:
    """Parse a single JSON response, tolerating extra text around the object."""
    if "{" in output:
        if "}" not in output:
            output += "}"
        start = output.index("{")
        end = output.rindex("}")
        output = output[start : end + 1]
    return json5.loads(output)


def batch_parse_json_response(output: list[str]) -> list[dict]:
    """Parse a list of JSON responses."""
    return [parse_json_response(item) for item in output]