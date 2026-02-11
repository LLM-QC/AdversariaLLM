from typing import TypeAlias

Conversation = list[dict[str, str]]

Json: TypeAlias = str | int | float | bool | None | dict[str, "Json"] | list["Json"]
JsonSchema: TypeAlias = dict[str, Json]