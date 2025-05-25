import unittest
import typing
from enum import StrEnum
from dataclasses import dataclass

from llm_tap.llm import (
    convert_field,
    to_json_schema,
    from_dict,
    make_helper,
    prepare_for_tool_use,
)


class SimpleEnum(StrEnum):
    A = "A_val"
    B = "B_val"


@dataclass
class SimpleModel:
    """A simple model for testing."""

    name: str
    age: int
    is_active: bool = True


@dataclass
class NestedModel:
    """A model nested within another model."""

    value: str
    count: float


@dataclass
class ComplexModel:
    """A complex model with various field types."""

    simple: SimpleModel
    nested_list: list[NestedModel]
    optional_int: int = None
    enum_field: SimpleEnum = SimpleEnum.A
    # dictionary: typing.Dict[str, int] # Dictionary support might need specific checks


@dataclass
class UnionModelChildA:
    """First child for Union testing."""

    a_field: str


@dataclass
class UnionModelChildB:
    """Second child for Union testing."""

    b_field: int


@dataclass
class ModelWithUnion:
    """Model with a Union field."""

    choice: typing.Union[UnionModelChildA, UnionModelChildB]


# Required for from_dict to find these classes by name
# This would typically be populated as schemas are generated,
# but for testing from_dict directly, we might need to pre-populate or ensure
# to_json_schema is called on these types first within the test setup.
# For now, let's rely on to_json_schema being called in tests before from_dict.


class TestLLMUtils(unittest.TestCase):
    def setUp(self):
        # Clear the mapping for each test to ensure independence
        # This is important because to_json_schema populates this global mapping.
        from llm_tap.llm import class_names_mapping

        class_names_mapping.clear()

    # --- Tests for convert_field ---

    def test_convert_field_primitives(self):
        """Test convert_field with primitive types."""
        self.assertEqual(convert_field(SimpleModel, int), {"type": "integer"})
        self.assertEqual(convert_field(SimpleModel, str), {"type": "string"})
        self.assertEqual(convert_field(SimpleModel, bool), {"type": "boolean"})
        self.assertEqual(convert_field(SimpleModel, float), {"type": "number"})
        self.assertEqual(
            convert_field(SimpleModel, bytes),
            {"type": "string", "contentEncoding": "base64"},
        )

    def test_convert_field_enum(self):
        """Test convert_field with Enum type."""
        expected = {"type": "string", "enum": ["A", "B"]}
        self.assertEqual(convert_field(ComplexModel, SimpleEnum), expected)

    def test_convert_field_list(self):
        """Test convert_field with list type."""
        expected = {"type": "array", "items": {"type": "string"}}
        self.assertEqual(convert_field(SimpleModel, list[str]), expected)

        expected_nested = {
            "type": "array",
            "items": to_json_schema(NestedModel),
        }
        # We need to ensure NestedModel's schema is generated and it's added to class_names_mapping
        # for the $ref to work if it were self-referential, but here it's direct embedding.
        # Calling to_json_schema ensures it's "known"
        to_json_schema(NestedModel)  # Ensure NestedModel is processed
        self.assertEqual(
            convert_field(ComplexModel, list[NestedModel]),
            expected_nested,
        )

    def test_convert_field_dataclass(self):
        """Test convert_field with a dataclass type."""
        # This will generate the schema for SimpleModel and cache it.
        expected_schema = to_json_schema(SimpleModel)
        self.assertEqual(
            convert_field(ComplexModel, SimpleModel), expected_schema
        )

    def test_convert_field_union(self):
        """Test convert_field with Union type."""
        # Ensure children are processed
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)

        result = convert_field(
            ModelWithUnion, typing.Union[UnionModelChildA, UnionModelChildB]
        )
        self.assertIn("anyOf", result)
        self.assertEqual(len(result["anyOf"]), 2)
        # Check if the generated structures for union choices are present
        # The exact structure depends on the make_dataclass augmentation in convert_field
        self.assertTrue(
            any(
                "ChoiceUnionModelChildA" in item.get("title", "")
                for item in result["anyOf"]
            )
        )
        self.assertTrue(
            any(
                "ChoiceUnionModelChildB" in item.get("title", "")
                for item in result["anyOf"]
            )
        )

    # --- Tests for to_json_schema ---

    def test_to_json_schema_simple_model(self):
        """Test to_json_schema with a simple dataclass."""
        schema = to_json_schema(SimpleModel)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["title"], "SimpleModel")
        self.assertEqual(schema["description"], "A simple model for testing.")
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])
        self.assertIn("is_active", schema["properties"])
        self.assertEqual(schema["properties"]["name"], {"type": "string"})
        self.assertEqual(schema["properties"]["age"], {"type": "integer"})
        self.assertEqual(
            schema["properties"]["is_active"], {"type": "boolean"}
        )
        self.assertIn("name", schema["required"])
        self.assertIn("age", schema["required"])
        self.assertNotIn(
            "is_active", schema["required"]
        )  # Because it has a default

    def test_to_json_schema_complex_model(self):
        """Test to_json_schema with a more complex dataclass."""
        # Ensure sub-models are processed and in class_names_mapping
        to_json_schema(SimpleModel)
        to_json_schema(NestedModel)

        schema = to_json_schema(ComplexModel)
        self.assertEqual(schema["title"], "ComplexModel")
        self.assertIn("simple", schema["properties"])
        self.assertEqual(
            schema["properties"]["simple"]["title"], "SimpleModel"
        )  # Checks if nested schema is embedded
        self.assertIn("nested_list", schema["properties"])
        self.assertEqual(schema["properties"]["nested_list"]["type"], "array")
        self.assertEqual(
            schema["properties"]["nested_list"]["items"]["title"],
            "NestedModel",
        )
        self.assertIn("optional_int", schema["properties"])

        self.assertIn("enum_field", schema["properties"])
        self.assertEqual(
            schema["properties"]["enum_field"],
            {"type": "string", "enum": ["A", "B"]},
        )

        self.assertIn("simple", schema["required"])
        self.assertIn("nested_list", schema["required"])
        self.assertNotIn("optional_int", schema["required"])  # Has default
        self.assertNotIn("enum_field", schema["required"])  # Has default

    def test_to_json_schema_model_with_union(self):
        """Test to_json_schema with a dataclass containing a Union field."""
        # Ensure union member types are processed
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)

        schema = to_json_schema(ModelWithUnion)
        self.assertEqual(schema["title"], "ModelWithUnion")
        self.assertIn("choice", schema["properties"])

        choice_schema = schema["properties"]["choice"]
        self.assertIn("anyOf", choice_schema)
        self.assertEqual(len(choice_schema["anyOf"]), 2)

        # Verify that the titles of the generated choice dataclasses are in the anyOf list
        # (e.g., "ChoiceUnionModelChildA", "ChoiceUnionModelChildB")
        any_of_titles = [item.get("title") for item in choice_schema["anyOf"]]
        self.assertIn("ChoiceUnionModelChildA", any_of_titles)
        self.assertIn("ChoiceUnionModelChildB", any_of_titles)
        self.assertIn("choice", schema["required"])

    # --- Tests for from_dict ---
    def test_from_dict_simple_model(self):
        """Test from_dict with a simple model."""
        to_json_schema(
            SimpleModel
        )  # Ensure SimpleModel is in class_names_mapping
        data = {"name": "Test", "age": 30, "is_active": False}
        instance = from_dict(SimpleModel, data)
        self.assertIsInstance(instance, SimpleModel)
        self.assertEqual(instance.name, "Test")
        self.assertEqual(instance.age, 30)
        self.assertEqual(instance.is_active, False)

    def test_from_dict_complex_model(self):
        """Test from_dict with a complex model."""
        # Ensure all involved models are "known"
        to_json_schema(SimpleModel)
        to_json_schema(NestedModel)
        to_json_schema(ComplexModel)

        data = {
            "simple": {"name": "Nested Test", "age": 25, "is_active": True},
            "nested_list": [
                {"value": "val1", "count": 1.0},
                {"value": "val2", "count": 2.5},
            ],
            "optional_int": 123,
            "enum_field": "B",
        }
        instance = from_dict(ComplexModel, data)
        self.assertIsInstance(instance, ComplexModel)
        self.assertIsInstance(instance.simple, SimpleModel)
        self.assertEqual(instance.simple.name, "Nested Test")
        self.assertEqual(len(instance.nested_list), 2)
        self.assertIsInstance(instance.nested_list[0], NestedModel)
        self.assertEqual(instance.nested_list[0].value, "val1")
        self.assertEqual(instance.nested_list[1].count, 2.5)
        self.assertEqual(instance.optional_int, 123)
        self.assertEqual(
            instance.enum_field, SimpleEnum.B
        )  # from_dict should handle enum string to member

    def test_from_dict_model_with_union_child_a(self):
        """Test from_dict with a Union field, choosing ChildA."""
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)
        to_json_schema(ModelWithUnion)

        # Data structure for Union as per convert_field's augmentation
        data = {
            "choice": {
                "name": "UnionModelChildA",  # This specifies which class in the Union
                "arguments": {"a_field": "hello"},
            }
        }
        instance = from_dict(ModelWithUnion, data)
        self.assertIsInstance(instance, ModelWithUnion)
        self.assertIsInstance(instance.choice, UnionModelChildA)
        self.assertEqual(instance.choice.a_field, "hello")

    def test_from_dict_model_with_union_child_b(self):
        """Test from_dict with a Union field, choosing ChildB."""
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)
        to_json_schema(ModelWithUnion)

        data = {
            "choice": {
                "name": "UnionModelChildB",
                "arguments": {"b_field": 123},
            }
        }
        instance = from_dict(ModelWithUnion, data)
        self.assertIsInstance(instance, ModelWithUnion)
        self.assertIsInstance(instance.choice, UnionModelChildB)
        self.assertEqual(instance.choice.b_field, 123)

    def test_from_dict_optional_field_present(self):
        """Test from_dict with an Optional field that is present."""
        to_json_schema(
            ComplexModel
        )  # Ensure ComplexModel and its fields are processed
        data = {
            "simple": {"name": "Test", "age": 1},
            "nested_list": [],
            "optional_int": 42,  # Present
        }
        instance = from_dict(ComplexModel, data)
        self.assertEqual(instance.optional_int, 42)

    def test_from_dict_optional_field_absent(self):
        """Test from_dict with an Optional field that is absent (should use default)."""
        to_json_schema(ComplexModel)
        # Note: from_dict expects all fields listed in the dataclass unless they have defaults.
        # If 'optional_int' is missing from 'data', and it has a default in dataclass,
        # from_dict should ideally allow this. However, current from_dict might require it if
        # it's not smart enough about dataclass defaults vs. presence in dict.
        # Let's assume the data provides 'None' explicitly if it's optional and not set,
        # or relies on the default value if key is missing.
        # The current `from_dict` will fail if a key for a non-default field is missing.
        # For optional fields with defaults, if the key is missing in `attrs`, `from_dict`
        # will not include it in `**{k: from_dict(...) ...}`.
        # The dataclass constructor `cls(**{...})` will then use its default.
        data_missing_optional = {
            "simple": {"name": "Test", "age": 1},
            "nested_list": [],
            # "optional_int": is missing, ComplexModel.optional_int defaults to None
        }
        instance = from_dict(ComplexModel, data_missing_optional)
        self.assertIsNone(instance.optional_int)  # Relies on dataclass default

    def test_from_dict_error_unknown_union_name(self):
        """Test from_dict error handling for unknown class name in Union."""
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)
        to_json_schema(ModelWithUnion)
        data = {
            "choice": {
                "name": "UnknownChild",  # Invalid name
                "arguments": {"field": "data"},
            }
        }
        with self.assertRaisesRegex(
            KeyError, "Class UnknownChild not found in Union"
        ):
            from_dict(ModelWithUnion, data)

    def test_from_dict_error_missing_required_field(self):
        """Test from_dict error handling for missing required field in nested model."""
        to_json_schema(SimpleModel)  # name and age are required
        data = {"name": "TestOnly"}  # Missing 'age'
        with self.assertRaises(
            TypeError
        ):  # Dataclass constructor will complain
            from_dict(SimpleModel, data)

    # --- Tests for make_helper ---

    def test_make_helper_simple_model(self):
        """Test make_helper with a simple dataclass."""
        helper_str = make_helper(SimpleModel)
        self.assertIn("name: SimpleModel", helper_str)
        self.assertIn("description: A simple model for testing.", helper_str)
        self.assertIn("inputs:", helper_str)
        self.assertIn("name (str) (required)", helper_str)
        self.assertIn("age (int) (required)", helper_str)
        self.assertIn(
            "is_active (bool)", helper_str
        )  # Not required due to default

    def test_make_helper_complex_model(self):
        """Test make_helper with a complex dataclass."""
        # Ensure sub-models are processed if their names are part of the output
        to_json_schema(SimpleModel)
        to_json_schema(NestedModel)

        helper_str = make_helper(ComplexModel)
        self.assertIn("name: ComplexModel", helper_str)
        self.assertIn(
            "description: A complex model with various field types.",
            helper_str,
        )
        self.assertIn("simple (SimpleModel) (required)", helper_str)
        self.assertIn(
            "nested_list (list) (required)", helper_str
        )  # Note: type is 'list' not list[NestedModel]
        self.assertIn(
            "optional_int (int)", helper_str
        )  # Note: type is 'Optional' not Optional[int]
        self.assertIn("enum_field (SimpleEnum)", helper_str)

    def test_make_helper_model_with_union(self):
        """Test make_helper with a model containing a Union field."""
        to_json_schema(UnionModelChildA)  # Process to get names if needed
        to_json_schema(UnionModelChildB)
        helper_str = make_helper(ModelWithUnion)
        self.assertIn("name: ModelWithUnion", helper_str)
        self.assertIn("choice (Union) (required)", helper_str)


class TestPrepareForToolUse(unittest.TestCase):
    def setUp(self):
        self.sample_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        self.helper_prompt = "You have access to the following tools. Use them if applicable."
        self.user_prompt = "What's the weather like in Boston?"
        self.system_prompt = "You are a helpful assistant."
        self.model_name = "gpt-4-test"

    def test_prepare_for_tool_use_basic(self):
        """Test basic payload construction for tool use."""
        payload = prepare_for_tool_use(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
        )

        expected_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.helper_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        self.assertEqual(payload["messages"], expected_messages)
        self.assertEqual(payload["tools"], self.sample_tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["parallel_tool_calls"], False)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("model", payload) # Model is None by default

    def test_prepare_for_tool_use_with_model(self):
        """Test payload construction with a specified model."""
        payload = prepare_for_tool_use(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            model=self.model_name,
        )
        self.assertEqual(payload["model"], self.model_name)

    def test_prepare_for_tool_use_no_model(self):
        """Test payload construction when model is explicitly None."""
        payload = prepare_for_tool_use(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            model=None,
        )
        self.assertNotIn("model", payload) # Model should not be in payload if None


if __name__ == "__main__":
    # We need to import prepare_for_tool_use here or at the top of the file
    # For the sake of this diff, assuming it's imported at the top of the file like other functions.
    # from llm_tap.llm import prepare_for_tool_use # Ensure this is at the top
    unittest.main(argv=["first-arg-is-ignored"], exit=False)


# Imports for adapter tests
import requests 
import time 
from unittest.mock import patch, MagicMock
from llm_tap.llm import HTTP # Assuming LLamaCPP will be tested separately or imported here too


class TestHTTPAdapterToolInteraction(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://fakeapi.com/chat"
        self.api_key = "fake_key"
        self.default_model_name = "gpt-http-default"
        # Initialize adapter with a default model
        self.adapter = HTTP(base_url=self.base_url, api_key=self.api_key, model=self.default_model_name)

        self.sample_tools = [{"type": "function", "function": {"name": "test_tool"}}]
        self.helper_prompt = "Use tools."
        self.user_prompt = "Call test_tool."
        self.system_prompt = "System message for tools."
        # This is what prepare_for_tool_use is expected to return
        self.prepared_payload_default_model = {
            "messages": [], "tools": self.sample_tools, "tool_choice": "auto", 
            "model": self.default_model_name, "temperature": 0.0, "parallel_tool_calls": False
        }
        self.prepared_payload_override_model = {
            "messages": [], "tools": self.sample_tools, "tool_choice": "auto",
            "model": "gpt-override", "temperature": 0.0, "parallel_tool_calls": False
        }
        self.llm_response_json = {"id": "chatcmpl-123", "choices": [{"message": {"role": "assistant"}}]}

    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('requests.Session.post')
    def test_execute_tool_interaction_success_with_default_model(self, mock_post, mock_prepare):
        mock_prepare.return_value = self.prepared_payload_default_model
        mock_response = MagicMock()
        mock_response.json.return_value = self.llm_response_json
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        response = self.adapter.execute_tool_interaction(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt
            # No model specified, should use adapter's default
        )

        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, self.default_model_name
        )

# Need to import LLamaCPP and llama_cpp (for mocking)
from llm_tap.llm import LLamaCPP
import llama_cpp # Mocked
import os # For mocking os.path.expanduser

class TestLLamaCPPAdapterToolInteraction(unittest.TestCase):
    def setUp(self):
        self.model_path = "/fake/path/to/model.gguf"
        self.adapter = LLamaCPP(model=self.model_path, n_ctx=2048, n_gpu_layers=10, n_threads=2)

        self.sample_tools = [{"type": "function", "function": {"name": "test_tool_llama"}}]
        self.helper_prompt = "Use Llama tools."
        self.user_prompt = "Call test_tool_llama."
        self.system_prompt = "Llama system message."
        self.prepared_payload = {
            "messages": [{"role": "system", "content": self.system_prompt}], 
            "tools": self.sample_tools, 
            "tool_choice": "auto",
            "temperature": 0.1, # Example temperature from payload
            "parallel_tool_calls": False
        }
        self.llm_response_dict = {
            "choices": [{"message": {"tool_calls": [{"function": {"name": "test_tool_llama", "arguments": "{}"}}]}}]
        }

    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('llama_cpp.Llama')
    @patch('os.path.expanduser', return_value=lambda x: x) # Mock expanduser
    def test_execute_tool_interaction_success(self, mock_expanduser, mock_llama_constructor, mock_prepare):
        mock_prepare.return_value = self.prepared_payload
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = self.llm_response_dict
        mock_llama_constructor.return_value = mock_llama_instance

        response = self.adapter.execute_tool_interaction(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt
            # No model override, adapter's model (path) will be used for Llama init, 
            # model param for prepare_for_tool_use will be None
        )

        mock_expanduser.assert_called_once_with(self.model_path)
        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, None
        )
        mock_llama_constructor.assert_called_once_with(
            mock_expanduser.return_value, # expanded path
            n_ctx=self.adapter.n_ctx,
            n_gpu_layers=self.adapter.n_gpu_layers,
            n_threads=self.adapter.n_threads,
            verbose=False
        )
        mock_llama_instance.create_chat_completion.assert_called_once_with(
            messages=self.prepared_payload["messages"],
            temperature=self.prepared_payload["temperature"],
            tools=self.prepared_payload["tools"],
            tool_choice=self.prepared_payload["tool_choice"]
        )
        self.assertEqual(response, self.llm_response_dict)
        # Check that _llm was deleted
        self.assertFalse(hasattr(self.adapter, '_llm'))


    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('llama_cpp.Llama')
    @patch('os.path.expanduser', return_value=lambda x: x)
    def test_execute_tool_interaction_with_passed_model_identifier(self, mock_expanduser, mock_llama_constructor, mock_prepare):
        # This test verifies that if a 'model' string (not path) is passed to execute_tool_interaction,
        # it is passed along to prepare_for_tool_use.
        # The LlamaCPP adapter itself will still use its constructor-defined model *path*.
        
        override_model_identifier = "some-model-identifier"
        # Update prepared_payload to reflect this override model for the assertion
        prepared_payload_with_override = self.prepared_payload.copy()
        prepared_payload_with_override["model"] = override_model_identifier
        
        mock_prepare.return_value = prepared_payload_with_override
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = self.llm_response_dict
        mock_llama_constructor.return_value = mock_llama_instance

        self.adapter.execute_tool_interaction(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            model=override_model_identifier # Pass a model identifier
        )

        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, override_model_identifier
        )
        # Llama constructor should still be called with the adapter's model path
        mock_llama_constructor.assert_called_once_with(
            mock_expanduser.return_value, # self.model_path after expanduser
            n_ctx=self.adapter.n_ctx,
            n_gpu_layers=self.adapter.n_gpu_layers,
            n_threads=self.adapter.n_threads,
            verbose=False
        )


    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('llama_cpp.Llama')
    @patch('os.path.expanduser', return_value=lambda x: x)
    def test_execute_tool_interaction_exception_in_llama(self, mock_expanduser, mock_llama_constructor, mock_prepare):
        mock_prepare.return_value = self.prepared_payload
        # Simulate an error during Llama interaction
        error_message = "Llama internal error"
        mock_llama_constructor.side_effect = Exception(error_message) # Error on instantiation

        with self.assertRaisesRegex(Exception, error_message):
            self.adapter.execute_tool_interaction(
                all_tools=self.sample_tools,
                helper_prompt=self.helper_prompt,
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt
            )
        
        # Test error during create_chat_completion
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.side_effect = Exception("Chat completion failed")
        mock_llama_constructor.side_effect = None # Reset side_effect
        mock_llama_constructor.return_value = mock_llama_instance
        
        with self.assertRaisesRegex(Exception, "Chat completion failed"):
            self.adapter.execute_tool_interaction(
                all_tools=self.sample_tools,
                helper_prompt=self.helper_prompt,
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt
            )
        # Ensure _llm is cleaned up even if create_chat_completion fails
        self.assertFalse(hasattr(self.adapter, '_llm'))
        mock_post.assert_called_once_with(self.base_url, json=self.prepared_payload_default_model)
        self.assertEqual(response, self.llm_response_json)

    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('requests.Session.post')
    def test_execute_tool_interaction_success_with_override_model(self, mock_post, mock_prepare):
        mock_prepare.return_value = self.prepared_payload_override_model
        mock_response = MagicMock()
        mock_response.json.return_value = self.llm_response_json
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        override_model_name = "gpt-override"
        response = self.adapter.execute_tool_interaction(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            model=override_model_name # Override model specified
        )

        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, override_model_name
        )
        mock_post.assert_called_once_with(self.base_url, json=self.prepared_payload_override_model)
        self.assertEqual(response, self.llm_response_json)


    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('time.sleep', return_value=None) # Mock time.sleep
    @patch('requests.Session.post')
    def test_execute_tool_interaction_retry_on_429(self, mock_post, mock_sleep, mock_prepare):
        # prepare_for_tool_use will be called twice, once for initial, once for retry
        # Ensure it returns the correct payload each time, especially if model changes (it doesn't here)
        mock_prepare.return_value = self.prepared_payload_default_model
        
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        # Crucially, the error's response attribute must be the mock_response_429 itself
        http_error_429 = requests.exceptions.HTTPError("429 Client Error", response=mock_response_429)
        mock_response_429.raise_for_status.side_effect = http_error_429 # Raise the error
        
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = self.llm_response_json
        mock_response_success.raise_for_status = MagicMock() # Does not raise

        mock_post.side_effect = [mock_response_429, mock_response_success]

        response = self.adapter.execute_tool_interaction(
            all_tools=self.sample_tools,
            helper_prompt=self.helper_prompt,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt
            # Uses default model
        )
        
        # Check prepare was called for the default model on both occasions
        mock_prepare.assert_any_call(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, self.default_model_name
        )
        self.assertEqual(mock_prepare.call_count, 2) 
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(10)
        self.assertEqual(response, self.llm_response_json)

    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('requests.Session.post')
    def test_execute_tool_interaction_http_error_non_429(self, mock_post, mock_prepare):
        mock_prepare.return_value = self.prepared_payload_default_model
        
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        # Ensure the error's response attribute is set
        http_error_500 = requests.exceptions.HTTPError("500 Server Error", response=mock_response_500)
        mock_response_500.raise_for_status.side_effect = http_error_500
        mock_post.return_value = mock_response_500

        with self.assertRaises(requests.exceptions.HTTPError) as context:
            self.adapter.execute_tool_interaction(
                all_tools=self.sample_tools,
                helper_prompt=self.helper_prompt,
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt
            )
        # Check that the raised exception is the one we created
        self.assertIs(context.exception, http_error_500)
        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, self.default_model_name
        )

    @patch('llm_tap.llm.prepare_for_tool_use')
    @patch('requests.Session.post')
    def test_execute_tool_interaction_request_exception(self, mock_post, mock_prepare):
        mock_prepare.return_value = self.prepared_payload_default_model
        connection_error = requests.exceptions.ConnectionError("Connection failed")
        mock_post.side_effect = connection_error # Post call itself raises this

        with self.assertRaises(requests.exceptions.ConnectionError) as context:
            self.adapter.execute_tool_interaction(
                all_tools=self.sample_tools,
                helper_prompt=self.helper_prompt,
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt
            )
        self.assertIs(context.exception, connection_error)
        mock_prepare.assert_called_once_with(
            self.sample_tools, self.helper_prompt, self.user_prompt, self.system_prompt, self.default_model_name
        )
