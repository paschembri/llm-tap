import unittest
import typing
from enum import Enum, StrEnum
from dataclasses import dataclass, field

from llm_tap.llm import (
    convert_field,
    to_json_schema,
    from_dict,
    make_helper,
    # class_names_mapping, # Not directly tested but used by from_dict
)
# from llm_tap.models import User, Joke  # Import example models if needed later

# --- Test Dataclasses ---

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
    nested_list: typing.List[NestedModel]
    optional_int: typing.Optional[int] = None
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
        self.assertEqual(convert_field(SimpleModel, bytes), {"type": "string", "contentEncoding": "base64"})

    def test_convert_field_enum(self):
        """Test convert_field with Enum type."""
        expected = {"type": "string", "enum": ["A", "B"]}
        self.assertEqual(convert_field(ComplexModel, SimpleEnum), expected)

    def test_convert_field_list(self):
        """Test convert_field with List type."""
        expected = {"type": "array", "items": {"type": "string"}}
        self.assertEqual(convert_field(SimpleModel, typing.List[str]), expected)

        expected_nested = {"type": "array", "items": to_json_schema(NestedModel)}
        # We need to ensure NestedModel's schema is generated and it's added to class_names_mapping
        # for the $ref to work if it were self-referential, but here it's direct embedding.
        # Calling to_json_schema ensures it's "known"
        to_json_schema(NestedModel) # Ensure NestedModel is processed
        self.assertEqual(convert_field(ComplexModel, typing.List[NestedModel]), expected_nested)

    def test_convert_field_dataclass(self):
        """Test convert_field with a dataclass type."""
        # This will generate the schema for SimpleModel and cache it.
        expected_schema = to_json_schema(SimpleModel)
        self.assertEqual(convert_field(ComplexModel, SimpleModel), expected_schema)

    def test_convert_field_union(self):
        """Test convert_field with Union type."""
        # Ensure children are processed
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)

        result = convert_field(ModelWithUnion, typing.Union[UnionModelChildA, UnionModelChildB])
        self.assertIn("anyOf", result)
        self.assertEqual(len(result["anyOf"]), 2)
        # Check if the generated structures for union choices are present
        # The exact structure depends on the make_dataclass augmentation in convert_field
        self.assertTrue(any("ChoiceUnionModelChildA" in item.get("title", "") for item in result["anyOf"]))
        self.assertTrue(any("ChoiceUnionModelChildB" in item.get("title", "") for item in result["anyOf"]))


    def test_convert_field_optional(self):
        """Test convert_field with Optional type (Union[T, NoneType])."""
        # Optional[int] is Union[int, NoneType].
        # The current implementation of convert_field for Union creates synthetic dataclasses
        # like "ChoiceInt" and "ChoiceNoneType".
        # This might be more complex than typical JSON schema representation for optional,
        # which often just omits the field from 'required' or uses "nullable": true / "type": ["integer", "null"]
        # Given the current implementation, we test its actual output.
        result = convert_field(ComplexModel, typing.Optional[int])
        self.assertIn("anyOf", result)
        # Expecting it to create choices for int and NoneType essentially
        # The exact titles will depend on how `_cls.__name__` behaves for `type(None)`
        self.assertTrue(any("ChoiceInt" in x.get("title", "") for x in result["anyOf"]))
        # The NoneType might be represented differently, let's check properties
        none_choice = next(x for x in result["anyOf"] if "Int" not in x.get("title", ""))
        # This part of test might need adjustment based on actual output for NoneType choice
        self.assertIn("ChoiceNoneType", none_choice.get("title",""))


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
        self.assertEqual(schema["properties"]["is_active"], {"type": "boolean"})
        self.assertIn("name", schema["required"])
        self.assertIn("age", schema["required"])
        self.assertNotIn("is_active", schema["required"]) # Because it has a default

    def test_to_json_schema_complex_model(self):
        """Test to_json_schema with a more complex dataclass."""
        # Ensure sub-models are processed and in class_names_mapping
        to_json_schema(SimpleModel)
        to_json_schema(NestedModel)

        schema = to_json_schema(ComplexModel)
        self.assertEqual(schema["title"], "ComplexModel")
        self.assertIn("simple", schema["properties"])
        self.assertEqual(schema["properties"]["simple"]["title"], "SimpleModel") # Checks if nested schema is embedded
        self.assertIn("nested_list", schema["properties"])
        self.assertEqual(schema["properties"]["nested_list"]["type"], "array")
        self.assertEqual(schema["properties"]["nested_list"]["items"]["title"], "NestedModel")
        self.assertIn("optional_int", schema["properties"])
        # Check how Optional[int] was translated. Based on current convert_field for Union:
        self.assertIn("anyOf", schema["properties"]["optional_int"])
        self.assertIn("enum_field", schema["properties"])
        self.assertEqual(schema["properties"]["enum_field"], {"type": "string", "enum": ["A", "B"]})

        self.assertIn("simple", schema["required"])
        self.assertIn("nested_list", schema["required"])
        self.assertNotIn("optional_int", schema["required"]) # Has default
        self.assertNotIn("enum_field", schema["required"]) # Has default

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
        to_json_schema(SimpleModel) # Ensure SimpleModel is in class_names_mapping
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
            "enum_field": "B_val",
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
        self.assertEqual(instance.enum_field, SimpleEnum.B) # from_dict should handle enum string to member

    def test_from_dict_model_with_union_child_a(self):
        """Test from_dict with a Union field, choosing ChildA."""
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)
        to_json_schema(ModelWithUnion)

        # Data structure for Union as per convert_field's augmentation
        data = {
            "choice": {
                "name": "UnionModelChildA", # This specifies which class in the Union
                "arguments": {"a_field": "hello"}
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
                "arguments": {"b_field": 123}
            }
        }
        instance = from_dict(ModelWithUnion, data)
        self.assertIsInstance(instance, ModelWithUnion)
        self.assertIsInstance(instance.choice, UnionModelChildB)
        self.assertEqual(instance.choice.b_field, 123)

    def test_from_dict_optional_field_present(self):
        """Test from_dict with an Optional field that is present."""
        to_json_schema(ComplexModel) # Ensure ComplexModel and its fields are processed
        data = {
            "simple": {"name": "Test", "age": 1},
            "nested_list": [],
            "optional_int": 42 # Present
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
        self.assertIsNone(instance.optional_int) # Relies on dataclass default

    def test_from_dict_error_unknown_union_name(self):
        """Test from_dict error handling for unknown class name in Union."""
        to_json_schema(UnionModelChildA)
        to_json_schema(UnionModelChildB)
        to_json_schema(ModelWithUnion)
        data = {
            "choice": {
                "name": "UnknownChild", # Invalid name
                "arguments": {"field": "data"}
            }
        }
        with self.assertRaisesRegex(KeyError, "Class UnknownChild not found in Union"):
            from_dict(ModelWithUnion, data)

    def test_from_dict_error_missing_required_field(self):
        """Test from_dict error handling for missing required field in nested model."""
        to_json_schema(SimpleModel) # name and age are required
        data = {"name": "TestOnly"} # Missing 'age'
        with self.assertRaises(TypeError): # Dataclass constructor will complain
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
        self.assertIn("is_active (bool)", helper_str) # Not required due to default

    def test_make_helper_complex_model(self):
        """Test make_helper with a complex dataclass."""
        # Ensure sub-models are processed if their names are part of the output
        to_json_schema(SimpleModel)
        to_json_schema(NestedModel)

        helper_str = make_helper(ComplexModel)
        self.assertIn("name: ComplexModel", helper_str)
        self.assertIn("description: A complex model with various field types.", helper_str)
        self.assertIn("simple (SimpleModel) (required)", helper_str)
        self.assertIn("nested_list (List) (required)", helper_str) # Note: type is 'List' not List[NestedModel]
        self.assertIn("optional_int (Optional)", helper_str) # Note: type is 'Optional' not Optional[int]
        self.assertIn("enum_field (SimpleEnum)", helper_str)

    def test_make_helper_model_with_union(self):
        """Test make_helper with a model containing a Union field."""
        to_json_schema(UnionModelChildA) # Process to get names if needed
        to_json_schema(UnionModelChildB)
        helper_str = make_helper(ModelWithUnion)
        self.assertIn("name: ModelWithUnion", helper_str)
        self.assertIn("choice (Union) (required)", helper_str)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
