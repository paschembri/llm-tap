# -*- coding: utf-8 -*-
"""Core module for llm-tap, providing functionalities for interacting with LLMs.

This module includes:
    - Conversion of Python data classes to JSON schemas.
    - Adapters for various LLM backends (HTTP, llama.cpp).
    - Helper functions for preparing prompts and parsing responses.
"""
import time
import os
import typing
import types
import json
import logging

from functools import lru_cache
from inspect import isclass
from enum import Enum, StrEnum
from textwrap import dedent

from dataclasses import (
    make_dataclass,
    dataclass,
    fields,
    is_dataclass,
    MISSING,
)

import requests
import llama_cpp
from contextlib import redirect_stdout, redirect_stderr


# Initialize logger
logger = logging.getLogger(__name__)

class_names_mapping = {}


@lru_cache
def convert_field(cls, field_type):
    """Converts a Python type to its JSON schema representation.

    This function handles basic types, dataclasses, Enums, and generic types
    like list, dict, Union, etc. It uses a mapping for basic types and
    recursively calls `to_json_schema` for nested dataclasses.

    Args:
        cls: The class containing the field (used for handling self-references).
        field_type: The Python type to convert.

    Returns:
        dict: The JSON schema representation of the field_type.

    Raises:
        NotImplementedError: If a container type with multiple type arguments
                             is encountered (e.g., Dict[str, int, str]).
        ValueError: If an unknown field type string is encountered.
    """
    class_name = cls.__name__

    mapping = {
        int: {"type": "integer"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        float: {"type": "number"},
        complex: {"type": "string", "format": "complex-number"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }

    if field_type in mapping:
        return mapping[field_type]
    else:
        if is_dataclass(field_type):
            return to_json_schema(field_type)

        elif (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is typing.Union
        ) or isinstance(field_type, types.UnionType):
            available_types = field_type.__args__
            augmented_types = [
                make_dataclass(
                    f"Choice{_cls.__name__}",
                    [
                        (
                            "name",
                            StrEnum(f"Enum{_cls.__name__}", [_cls.__name__]),
                        ),
                        ("arguments", _cls),
                    ],
                )
                for _cls in available_types
            ]

            return {"anyOf": [convert_field(cls, t) for t in augmented_types]}

        elif isinstance(field_type, types.GenericAlias):
            container_type = field_type.__origin__
            items_type = field_type.__args__

            if len(items_type) != 1:
                # For now, only support container types with a single type argument
                # e.g. list[str], not dict[str, int]
                raise NotImplementedError(
                    f"Annotation not supported for {field_type}[{items_type}]"
                )

            items_type = items_type[0]  # Get the inner type

            items = convert_field(cls, items_type)

            container_mapping = {
                list: {"type": "array", "items": items},
                tuple: {"type": "array", "items": items},
                dict: {"type": "object", "additionalProperties": items},
                set: {"type": "array", "uniqueItems": True, "items": items},
                frozenset: {
                    "type": "array",
                    "uniqueItems": True,
                    "items": items,
                },
            }
            return container_mapping[container_type]

        elif isclass(field_type) and issubclass(field_type, Enum):
            return {
                "type": "string",
                "enum": list(field_type.__members__.keys()),
            }

        elif type(field_type) is str:
            if field_type == class_name:
                # Handle self-referencing types
                return {"$ref": "#"}
            else:
                # This case might occur if a string literal for a type hint
                # is used for a class not yet defined or not in class_names_mapping
                raise ValueError(f"Unknown field type: {field_type}")

        else:
            # Default to object type if no other match
            return {
                "type": "object",
            }


@lru_cache
def to_json_schema(data_class):
    """Converts a Python dataclass to its JSON schema representation.

    The schema includes title, description (from docstring), properties,
    and required fields.

    Args:
        data_class: The Python dataclass to convert.

    Returns:
        dict: The JSON schema representation of the dataclass.
    """
    properties = {}
    required_fields = []

    class_names_mapping[data_class.__name__] = data_class

    for f in fields(data_class):
        properties[f.name] = convert_field(data_class, f.type)

        no_default = f.default == MISSING
        no_default_factory = f.default_factory == MISSING
        required = no_default and no_default_factory

        if required:
            required_fields.append(f.name)

    data_class_name = data_class.__name__
    data_class_docstring = (
        dedent(data_class.__doc__).strip()
        if data_class.__doc__
        else data_class.__name__
    )

    schema = {
        "type": "object",
        "title": data_class_name,
        "description": data_class_docstring,
        "properties": properties,
        "required": required_fields,
    }
    logger.debug(
        f"Generated JSON schema for {data_class_name}: {json.dumps(schema, indent=2)}"
    )
    return schema


def from_dict(cls, attrs):
    """Recursively converts a dictionary to an instance of a given class.

    This function is the inverse of `to_json_schema` in a way,
    reconstructing Python objects from a dictionary representation,
    presumably parsed from a JSON. It handles dataclasses,
    unions, container types (list, tuple, set, frozenset), and Enums.

    Args:
        cls: The target class or type to convert the dictionary into.
             Can be a dataclass, a type hint (like Union, list), or an Enum.
        attrs: The dictionary of attributes to use for creating the instance.

    Returns:
        An instance of `cls` populated with data from `attrs`.

    Raises:
        ValueError: If `cls` is a string and not found in `class_names_mapping`.
        KeyError: If a 'name' key in `attrs` for a Union type does not
                  correspond to any of the Union's arguments.
    """
    containers = (
        list,
        tuple,
        set,
        frozenset,
    )
    if isinstance(cls, str):
        cls = class_names_mapping.get(cls)
        if cls is None:
            raise ValueError(f"Unknown class name: {cls}")

    if is_dataclass(cls):
        field_types = {f.name: f.type for f in fields(cls)}
        return cls(
            **{k: from_dict(field_types[k], v) for k, v in attrs.items()}
        )

    elif (
        isinstance(cls, types.UnionType)
        or hasattr(cls, "__origin__")
        and cls.__origin__ is typing.Union
    ):
        try:
            # For Union types, attrs is expected to have a 'name' field
            # indicating which class in the Union to instantiate,
            # and an 'arguments' field for its constructor arguments.
            target_cls = next(
                filter(
                    lambda target_cls: target_cls.__name__ == attrs["name"],
                    cls.__args__,
                )
            )
        except StopIteration:
            raise KeyError(f"Class {attrs['name']} not found in Union {cls}")

        instance = target_cls(
            **attrs["arguments"]
        )  # Instantiate the chosen class

        return instance

    elif hasattr(cls, "__origin__") and cls.__origin__ in containers:
        return cls.__origin__([from_dict(cls.__args__[0], v) for v in attrs])

    elif hasattr(cls, "__name__") and cls.__name__ == "list":
        return [from_dict(cls.__args__[0], v) for v in attrs]

    elif hasattr(cls, "__name__") and cls.__name__ == "set":
        return set([from_dict(cls.__args__[0], v) for v in attrs])

    elif isclass(cls) and issubclass(cls, Enum):
        return getattr(cls, attrs)

    else:
        return attrs


def as_tool(json_schema):
    """Formats a JSON schema into an OpenAI tool specification.

    Args:
        json_schema (dict): The JSON schema of the tool's parameters.
                            Typically generated by `to_json_schema`.

    Returns:
        dict: An OpenAI tool specification dictionary.
    """
    return {
        "type": "function",
        "function": {
            "name": json_schema["title"],
            "description": json_schema["description"],
            "parameters": json_schema,
        },
    }


def as_tool_choice(json_schema):
    """Creates an OpenAI tool_choice dictionary for a given JSON schema.

    This is used to specify which tool (function) the LLM should call.

    Args:
        json_schema (dict): The JSON schema of the tool, typically from
                            `to_json_schema`. The 'title' field of the
                            schema is used as the function name.

    Returns:
        dict: An OpenAI tool_choice dictionary.
    """
    return {"type": "function", "function": {"name": json_schema["title"]}}


@lru_cache
def make_helper(data_class):
    """Generates a string representation of a dataclass for LLM prompts.

    This helper string includes the class name, its description (from docstring),
    and a list of its fields with their types and requirement status.
    This is used to guide the LLM in generating structured output.

    Args:
        data_class: The dataclass to generate the helper string for.

    Returns:
        str: A formatted string describing the dataclass structure.
    """
    class_name = f"name: {data_class.__name__}"
    class_description = f"description: {dedent(data_class.__doc__.strip()) if data_class.__doc__ else data_class.__name__}"

    inputs_schema = []

    for f in fields(data_class):
        field_name = f.name
        field_type_name = getattr(
            f.type, "__name__", getattr(f.type, "_name", "")
        )
        is_required = f.default == MISSING and f.default_factory == MISSING
        requirement_status = "(required)" if is_required else ""
        field_representation = (
            f"{field_name} ({field_type_name}) {requirement_status}"
        )
        inputs_schema.append(field_representation)

    formatted_inputs_schema = "\n".join(inputs_schema)

    helper_description = "\n".join(
        (class_name, class_description, "inputs:", formatted_inputs_schema)
    )

    return helper_description


@lru_cache
def prepare(data_class, prompt, system_prompt, model=None):
    """Prepares the payload for an LLM API call.

    This function constructs the messages list, tool specifications,
    and other parameters required for an LLM API request.

    Args:
        data_class: The dataclass representing the expected structured output.
        prompt (str): The user's prompt for the LLM.
        system_prompt (str): The system prompt to guide the LLM's behavior.
        model (str, optional): The specific model to use for the API call.
                               Defaults to None.

    Returns:
        dict: The payload dictionary ready to be sent to an LLM API.
    """
    data_class_schema = to_json_schema(data_class)
    data_class_tool = as_tool(data_class_schema)
    data_class_tool_choice = as_tool_choice(data_class_schema)
    data_class_helper = make_helper(data_class)

    user_prompt = "\n\n".join((data_class_helper, prompt))

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "tools": [data_class_tool],
        "tool_choice": data_class_tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0.0,
    }

    if model is not None:
        payload.update({"model": model})

    logger.debug(
        f"Prepared payload for {data_class.__name__} with prompt '{prompt[:50]}...': {json.dumps(payload, indent=2)}"
    )
    return payload


def prepare_for_tool_use(all_tools, helper_prompt, user_prompt, system_prompt, model=None):
    """Prepares the payload for an LLM API call with multiple tools.

    Args:
        all_tools (list): A list of tool schemas.
        helper_prompt (str): A helper prompt describing the tools or general instructions.
        user_prompt (str): The user's prompt for the LLM.
        system_prompt (str): The system prompt to guide the LLM's behavior.
        model (str, optional): The specific model to use for the API call.
                               Defaults to None.

    Returns:
        dict: The payload dictionary ready to be sent to an LLM API.
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": helper_prompt, # Helper prompt added as a user message
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    payload = {
        "messages": messages,
        "tools": all_tools,
        "tool_choice": "auto",  # LLM decides which tool to call
        "parallel_tool_calls": False, # Assuming we don't want parallel calls for now
        "temperature": 0.0,
    }

    if model is not None:
        payload.update({"model": model})

    logger.debug(
        f"Prepared payload for tool use with user prompt '{user_prompt[:50]}...': {json.dumps(payload, indent=2)}"
    )
    return payload


def parse_response(data_class, attributes):
    """Parses the LLM's response and converts it to a dataclass instance.

    Assumes the response contains a tool call with function arguments
    in JSON format.

    Args:
        data_class: The target dataclass to instantiate with the parsed data.
        attributes (dict): The raw response dictionary from the LLM API.
                           Expected to follow OpenAI's chat completion format.

    Returns:
        An instance of `data_class` populated with data from the LLM's response.
    """
    try:
        # Extract the tool call from the response
        tool_call = attributes["choices"][0]["message"]["tool_calls"][0]
        # Extract the JSON string of arguments from the tool call
        arguments_json = tool_call["function"]["arguments"]
        arguments_dict = json.loads(arguments_json)
        instance = from_dict(data_class, arguments_dict)
        logger.info(
            f"Successfully parsed response into an instance of {data_class.__name__}"
        )
        return instance
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(
            f"Error parsing LLM response for {data_class.__name__}: {e}. Response attributes: {attributes}"
        )
        raise ValueError(
            f"LLM output malformed or missing expected tool_call/arguments: {attributes}"
        ) from e


@dataclass
class HTTP:
    """HTTP adapter for interacting with OpenAI-compatible LLM APIs.

    This class provides a convenient way to send requests to an LLM API
    endpoint using HTTP POST requests. It handles request preparation,
    sending the request, and parsing the response.

    Attributes:
        base_url (str): The base URL of the LLM API endpoint.
                        Defaults to the value of the "ENDPOINT" environment variable.
        api_key (str): The API key for authentication.
                       Defaults to the value of the "API_KEY" environment variable.
        model (str): The default model to use for requests.
                     Defaults to the value of the "DEFAULT_MODEL" environment variable.
        session (requests.Session): The requests session object used for making
                                    HTTP requests.
    """

    base_url: str = os.getenv("ENDPOINT")
    api_key: str = os.getenv("API_KEY")
    model: str = os.getenv("DEFAULT_MODEL")

    def __post_init__(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "api-key": f"{self.api_key}",
        }
        self.session = requests.Session()
        self.session.headers.update(headers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def parse(self, data_class, prompt="", system_prompt="Answer in JSON"):
        """Sends a prompt to the LLM API and parses the structured response.

        Args:
            data_class: The dataclass to structure the LLM's response into.
            prompt (str, optional): The user prompt. Defaults to "".
            system_prompt (str, optional): The system prompt.
                                         Defaults to "Answer in JSON".

        Returns:
            An instance of `data_class` populated with the LLM's response.

        Raises:
            requests.exceptions.HTTPError: If the API returns an HTTP error status,
                                           except for 429 (Too Many Requests),
                                           which triggers a retry after a delay.
        """
        payload = prepare(data_class, prompt, system_prompt, self.model)
        response = self.session.post(self.base_url, json=payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(
                    f"Rate limit exceeded (429) for {self.base_url}. Retrying in 10 seconds..."
                )
                time.sleep(10)
                return self.parse(
                    data_class, prompt, system_prompt
                )  # Pass system_prompt for retry
            logger.error(f"HTTP error during API call to {self.base_url}: {e}")
            raise e
        except (
            requests.exceptions.RequestException
        ) as e:  # Catch other request errors like ConnectionError
            logger.error(
                f"Request exception during API call to {self.base_url}: {e}"
            )
            raise e
        attributes = response.json()
        return parse_response(data_class, attributes)

    def execute_tool_interaction(self, all_tools, helper_prompt, user_prompt, system_prompt="You are a helpful assistant that can use tools.", model=None):
        """Sends a prompt to the LLM API for tool interaction and returns the raw response.

        Args:
            all_tools (list): A list of tool schemas.
            helper_prompt (str): A helper prompt describing the tools or general instructions.
            user_prompt (str): The user's prompt for the LLM.
            system_prompt (str, optional): The system prompt.
                                         Defaults to "You are a helpful assistant that can use tools.".
            model (str, optional): The specific model to use. Defaults to class's self.model.

        Returns:
            dict: The JSON response from the LLM API.

        Raises:
            requests.exceptions.HTTPError: If the API returns an HTTP error status,
                                           except for 429 (Too Many Requests),
                                           which triggers a retry after a delay.
        """
        current_model = model if model is not None else self.model
        payload = prepare_for_tool_use(all_tools, helper_prompt, user_prompt, system_prompt, current_model)
        response = self.session.post(self.base_url, json=payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(
                    f"Rate limit exceeded (429) for {self.base_url}. Retrying in 10 seconds..."
                )
                time.sleep(10)
                # Retry with the same parameters
                return self.execute_tool_interaction(
                    all_tools, helper_prompt, user_prompt, system_prompt, current_model
                )
            logger.error(f"HTTP error during API call to {self.base_url}: {e}")
            raise e
        except requests.exceptions.RequestException as e:  # Catch other request errors
            logger.error(
                f"Request exception during API call to {self.base_url}: {e}"
            )
            raise e
        return response.json()


@dataclass
class LLamaCPP:
    """Adapter for interacting with local LLMs using llama.cpp via llama-cpp-python.

    This class allows running inference on GGUF models locally. It handles
    loading the model, preparing the prompt, running inference, and parsing
    the structured response.

    Attributes:
        model (str): Path to the GGUF model file.
                     Defaults to "/path/to/any/gguf/model".
                     It's recommended to change this to an actual model path.
        n_ctx (int): The context size for the model. Defaults to 4000.
        n_gpu_layers (int): Number of layers to offload to GPU.
                            Defaults to 100 (may need adjustment based on GPU VRAM).
        n_threads (int): Number of threads to use for generation. Defaults to 1.
    """

    model: str = (
        "/path/to/any/gguf/model"  # Placeholder, user should change this
    )
    n_ctx: int = 4_000
    n_gpu_layers: int = 100
    n_threads: int = 1

    def __post_init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def parse(self, data_class, prompt="", system_prompt="Answer in JSON"):
        """Runs inference with a local llama.cpp model and parses the structured response.

        Args:
            data_class: The dataclass to structure the LLM's response into.
            prompt (str, optional): The user prompt. Defaults to "".
            system_prompt (str, optional): The system prompt.
                                         Defaults to "Answer in JSON".

        Returns:
            An instance of `data_class` populated with the LLM's response.
        """
        model_path = os.path.expanduser(self.model)
        payload = prepare(data_class, prompt, system_prompt)

        try:
            # Suppress llama.cpp stdout/stderr messages during model loading and inference
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    self._llm = llama_cpp.Llama(
                        model_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_threads=self.n_threads,
                        verbose=False,  # Disable verbose logging from llama_cpp.Llama
                    )

                    response = self._llm.create_chat_completion(
                        messages=payload["messages"],
                        temperature=payload[
                            "temperature"
                        ],  # Typically 0.0 for structured output
                        tools=payload["tools"],
                        tool_choice=payload["tool_choice"],
                    )
                    del self._llm  # Release the model resources
            return parse_response(data_class, response)
        except Exception as e:  # Catch-all for llama.cpp related errors
            logger.error(
                f"Error during LLamaCPP parsing for model {self.model}: {e}"
            )
            raise e

    def execute_tool_interaction(self, all_tools, helper_prompt, user_prompt, system_prompt="You are a helpful assistant that can use tools.", model=None):
        """Runs inference with a local llama.cpp model for tool interaction.

        Args:
            all_tools (list): A list of tool schemas.
            helper_prompt (str): A helper prompt describing the tools or general instructions.
            user_prompt (str): The user's prompt for the LLM.
            system_prompt (str, optional): The system prompt.
                                         Defaults to "You are a helpful assistant that can use tools.".
            model (str, optional): The specific model string to pass to prepare_for_tool_use.
                                   Note: LLamaCPP uses the model path from its constructor.

        Returns:
            dict: The response from the LLM.
        """
        model_path = os.path.expanduser(self.model) # self.model is the path for LlamaCPP
        
        # The 'model' parameter here is for the payload, prepare_for_tool_use can use it if needed
        # but LlamaCPP itself will use model_path for loading.
        payload = prepare_for_tool_use(all_tools, helper_prompt, user_prompt, system_prompt, model)

        try:
            # Suppress llama.cpp stdout/stderr messages
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    self._llm = llama_cpp.Llama(
                        model_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_threads=self.n_threads,
                        verbose=False,
                    )
                    response = self._llm.create_chat_completion(
                        messages=payload["messages"],
                        temperature=payload.get("temperature", 0.0),
                        tools=payload["tools"],
                        tool_choice=payload["tool_choice"],
                    )
                    del self._llm  # Release the model resources
            return response
        except Exception as e:  # Catch-all for llama.cpp related errors
            logger.error(
                f"Error during LLamaCPP tool interaction for model {self.model}: {e}"
            )
            raise e
