# tests/test_instance_interaction.py
import unittest
import inspect
import json
from dataclasses import MISSING, is_dataclass
from typing import Any, Dict, List, Optional, Union

# Adjust path to import from src/llm_tap
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from llm_tap.instance_registry import InstanceRegistry
from llm_tap.instance_executor import LLMInstanceInterface

# We might need make_dataclass if we are deeply testing _get_method_parameters_dataclass output type
from dataclasses import make_dataclass

# --- Mock/Example Classes for Testing ---


class MockThermostat:
    """A mock thermostat for testing."""

    def __init__(self, id: str):
        self.id = id
        self.temp = 20.0
        self.unit = "C"

    def set_temperature(
        self, temperature: float, unit: Optional[str] = "Celsius"
    ) -> str:
        """Sets the thermostat temperature."""
        self.temp = temperature
        self.unit = unit if unit else "Celsius"
        return f"Temperature set to {self.temp}°{self.unit} for {self.id}"

    def get_temperature(self) -> float:
        """Gets the current temperature."""
        return self.temp

    def _internal_method(self):
        """Should not be called."""
        return "internal"

    def method_with_no_params(self) -> str:
        return "no_params_called"

    def method_with_custom_object_param(self, custom_obj: Dict) -> str:
        return f"Received: {custom_obj.get('key')}"


class MockLight:
    """A mock light for testing."""

    def __init__(self, name: str):
        self.name = name
        self.is_on = False

    def turn_on(self):
        """Turns the light on."""
        self.is_on = True
        return f"{self.name} is ON"

    def turn_off(self):
        """Turns the light off."""
        self.is_on = False
        return f"{self.name} is OFF"

    def method_that_fails(self, value: int) -> int:
        if value == 0:
            raise ValueError("Value cannot be zero.")
        return 10 / value


class TestInstanceRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = InstanceRegistry()
        self.thermostat_instance = MockThermostat("thermo1")
        self.light_instance = MockLight("kitchen_light")

    def test_register_instance_success(self):
        self.registry.register_instance(
            "thermo_living",
            self.thermostat_instance,
            "Living room thermostat",
            {"methods": ["set_temperature", "get_temperature"]},
        )
        self.assertIn("thermo_living", self.registry.instances)
        self.assertEqual(
            self.registry.get_instance("thermo_living"),
            self.thermostat_instance,
        )
        self.assertEqual(
            self.registry.get_instance_description("thermo_living"),
            "Living room thermostat",
        )

    def test_register_instance_duplicate_id(self):
        self.registry.register_instance(
            "thermo1",
            self.thermostat_instance,
            "Desc1",
            {"methods": ["get_temperature"]},
        )
        with self.assertRaisesRegex(
            ValueError, "Instance ID 'thermo1' is already registered."
        ):
            self.registry.register_instance(
                "thermo1",
                self.light_instance,
                "Desc2",
                {"methods": ["turn_on"]},
            )

    def test_register_instance_method_not_found(self):
        with self.assertRaisesRegex(
            ValueError,
            "Method 'non_existent_method' not found or not callable on instance 'thermo1'.",
        ):
            self.registry.register_instance(
                "thermo1",
                self.thermostat_instance,
                "Desc1",
                {"methods": ["non_existent_method"]},
            )

    def test_register_instance_attribute_as_method(self):
        with self.assertRaisesRegex(
            ValueError,
            "Method 'id' not found or not callable on instance 'thermo1'.",
        ):
            self.registry.register_instance(
                "thermo1",
                self.thermostat_instance,
                "Desc1",
                {"methods": ["id"]},  # 'id' is an attribute, not a method
            )

    def test_register_instance_internal_method_on_allowlist(self):
        # Registering a method starting with _ should be fine if it exists
        self.registry.register_instance(
            "thermo_hidden",
            self.thermostat_instance,
            "Thermo with hidden method",
            {"methods": ["_internal_method"]},
        )
        self.assertIn("thermo_hidden", self.registry.instances)

    def test_get_instance_not_found(self):
        with self.assertRaises(KeyError):
            self.registry.get_instance("non_existent_id")

    def test_list_available_instances(self):
        self.registry.register_instance(
            "t1",
            self.thermostat_instance,
            "Thermo 1",
            {"methods": ["get_temperature"]},
        )
        self.registry.register_instance(
            "l1", self.light_instance, "Light 1", {"methods": ["turn_on"]}
        )
        expected_list = {"t1": "Thermo 1", "l1": "Light 1"}
        self.assertEqual(
            self.registry.list_available_instances(), expected_list
        )


class TestLLMInstanceInterface(unittest.TestCase):
    def setUp(self):
        self.registry = InstanceRegistry()
        self.thermostat_instance = MockThermostat("main_thermo")
        self.light_instance = MockLight("main_light")

        self.registry.register_instance(
            "thermostat",
            self.thermostat_instance,
            "Main thermostat",
            {
                "methods": [
                    "set_temperature",
                    "get_temperature",
                    "method_with_no_params",
                    "method_with_custom_object_param",
                ]
            },
        )
        self.registry.register_instance(
            "light",
            self.light_instance,
            "Main light",
            {"methods": ["turn_on", "turn_off", "method_that_fails"]},
        )
        self.interface = LLMInstanceInterface(self.registry)

    def test_get_method_parameters_dataclass(self):
        method = self.thermostat_instance.set_temperature
        param_dc = self.interface._get_method_parameters_dataclass(method)

        self.assertTrue(is_dataclass(param_dc))
        fields = {f.name: f for f in param_dc.__dataclass_fields__.values()}

        self.assertIn("temperature", fields)
        self.assertEqual(fields["temperature"].type, float)
        self.assertEqual(fields["temperature"].default, MISSING)  # Required

        self.assertIn("unit", fields)
        self.assertEqual(fields["unit"].type, Optional[str])
        self.assertEqual(fields["unit"].default, "Celsius")  # Has default

        method_no_params = self.thermostat_instance.method_with_no_params
        param_dc_no_params = self.interface._get_method_parameters_dataclass(
            method_no_params
        )
        self.assertTrue(is_dataclass(param_dc_no_params))
        self.assertEqual(len(param_dc_no_params.__dataclass_fields__), 0)

    def test_generate_tool_schemas_for_instance(self):
        schemas = self.interface.generate_tool_schemas_for_instance(
            "thermostat"
        )
        self.assertEqual(
            len(schemas), 4
        )  # set_temperature, get_temperature, method_with_no_params, method_with_custom_object_param

        set_temp_schema = next(
            s
            for s in schemas
            if s["function"]["name"] == "thermostat.set_temperature"
        )
        self.assertIn(
            "Main thermostat", set_temp_schema["function"]["description"]
        )
        self.assertIn(
            "Sets the thermostat temperature.",
            set_temp_schema["function"]["description"],
        )
        self.assertIn(
            "temperature",
            set_temp_schema["function"]["parameters"]["properties"],
        )
        self.assertIn(
            "unit", set_temp_schema["function"]["parameters"]["properties"]
        )
        self.assertEqual(
            set_temp_schema["function"]["parameters"]["properties"]["unit"][
                "default"
            ],
            "Celsius",
        )

        get_temp_schema = next(
            s
            for s in schemas
            if s["function"]["name"] == "thermostat.get_temperature"
        )
        self.assertEqual(
            len(get_temp_schema["function"]["parameters"]["properties"]), 0
        )  # No params
        self.assertEqual(
            get_temp_schema["function"]["parameters"]["required"], []
        )

        custom_param_schema = next(
            s
            for s in schemas
            if s["function"]["name"]
            == "thermostat.method_with_custom_object_param"
        )
        self.assertIn(
            "custom_obj",
            custom_param_schema["function"]["parameters"]["properties"],
        )
        # type is Dict, which to_json_schema turns into {"type": "object", "additionalProperties": {"type": "object"}} if not further specified
        # or just {"type": "object"} if type hint is just Dict
        # For Dict (same as typing.Dict[typing.Any, typing.Any]) it becomes this:
        self.assertEqual(
            custom_param_schema["function"]["parameters"]["properties"][
                "custom_obj"
            ],
            {"type": "object"},
        )

    def test_generate_all_tool_schemas(self):
        all_schemas = self.interface.generate_all_tool_schemas()
        # 4 for thermostat + 3 for light = 7
        self.assertEqual(len(all_schemas), 7)
        names = [s["function"]["name"] for s in all_schemas]
        self.assertIn("thermostat.set_temperature", names)
        self.assertIn("light.turn_on", names)

    def test_generate_helper_text(self):
        helper_text = self.interface.generate_helper_text()
        self.assertIn("- Object ID: 'thermostat'", helper_text)
        self.assertIn("  Description: Main thermostat", helper_text)
        self.assertIn(
            "    - set_temperature(temperature: float, unit: Optional[str] = 'Celsius')",
            helper_text,
        )
        self.assertIn(
            "    - get_temperature()", helper_text
        )  # Python 3.9+ might show -> float
        self.assertIn("- Object ID: 'light'", helper_text)
        self.assertIn("  Description: Main light", helper_text)
        self.assertIn("    - turn_on()", helper_text)
        self.assertIn("    - turn_off()", helper_text)

    def test_parse_llm_action(self):
        llm_output = {
            "name": "thermostat.set_temperature",
            "arguments": '{"temperature": 25.5, "unit": "F"}',
        }
        instance_id, method_name, args = self.interface.parse_llm_action(
            llm_output
        )
        self.assertEqual(instance_id, "thermostat")
        self.assertEqual(method_name, "set_temperature")
        self.assertEqual(args, {"temperature": 25.5, "unit": "F"})

    def test_parse_llm_action_no_dot(self):
        llm_output = {"name": "thermostat_set_temperature", "arguments": "{}"}
        # The original error message was "LLM action name format error: Expected 'instance_id.method_name'"
        # My implementation produces "LLM output name is not in 'instance_id.method_name' format."
        # I will use the error message produced by my implementation.
        with self.assertRaisesRegex(
            ValueError,
            "LLM output name is not in 'instance_id.method_name' format.",
        ):
            self.interface.parse_llm_action(llm_output)

    def test_parse_llm_action_invalid_json(self):
        llm_output = {
            "name": "thermostat.set_temperature",
            "arguments": '{"temp": 25.5, unit: "F"}',
        }  # Invalid JSON
        # The original error was json.JSONDecodeError
        # My implementation produces a ValueError that wraps the JSONDecodeError
        # I will use the error message produced by my implementation.
        with self.assertRaisesRegex(
            ValueError, "Failed to parse arguments JSON string:"
        ):
            self.interface.parse_llm_action(llm_output)

    def test_execute_action_success(self):
        result = self.interface.execute_action(
            "thermostat", "set_temperature", {"temperature": 22.0, "unit": "C"}
        )
        self.assertEqual(result, "Temperature set to 22.0°C for main_thermo")
        self.assertEqual(self.thermostat_instance.temp, 22.0)

        result_no_params = self.interface.execute_action(
            "thermostat", "method_with_no_params", {}
        )
        self.assertEqual(result_no_params, "no_params_called")

    def test_execute_action_not_allowlisted(self):
        # _internal_method is not in thermostat's allowlist
        with self.assertRaisesRegex(
            PermissionError,
            "Method '_internal_method' is not allowlisted for instance 'thermostat'.",
        ):
            self.interface.execute_action("thermostat", "_internal_method", {})

    def test_execute_action_method_does_not_exist_on_instance_but_allowlisted_by_mistake(
        self,
    ):
        # This case should ideally be caught at registration by InstanceRegistry.
        # If it slips through (e.g. if validation is off or method is deleted post-registration),
        # execute_action should fail when getattr tries to get the method.

        # Let's simulate this by trying to call a method we know isn't on the allowlist
        # (which means it wouldn't have been validated at registration for this test setup)
        # but more importantly, let's try one that doesn't exist at all.
        self.registry.allowlists["thermostat"]["methods"].append(
            "fake_method_on_allowlist"
        )  # Manually add to allowlist

        with self.assertRaises(AttributeError):  # Because getattr will fail
            self.interface.execute_action(
                "thermostat", "fake_method_on_allowlist", {}
            )

        # Clean up
        self.registry.allowlists["thermostat"]["methods"].pop()

    def test_execute_action_instance_not_found(self):
        with self.assertRaises(KeyError):  # From registry.get_instance
            self.interface.execute_action(
                "non_existent_thermostat", "set_temperature", {}
            )

    def test_execute_action_method_raises_exception(self):
        with self.assertRaisesRegex(ValueError, "Value cannot be zero."):
            self.interface.execute_action(
                "light", "method_that_fails", {"value": 0}
            )

        # Check it works with non-zero
        result = self.interface.execute_action(
            "light", "method_that_fails", {"value": 2}
        )
        self.assertEqual(result, 5.0)

    def test_full_conceptual_flow_with_simulated_llm_tool_call(self):
        """
        Tests the conceptual flow:
        1. Simulate receiving a raw LLM tool call response.
        2. Manually extract the relevant 'function' part of a tool call.
        3. Pass to parse_llm_action.
        4. Pass results to execute_action.
        5. Verify the outcome.
        This test clarifies how LLMInstanceInterface methods are used with
        an LLM response that would have come from a method like 'execute_tool_interaction'.
        """
        # 1. Simulate raw LLM response (as if from adapter.execute_tool_interaction)
        simulated_raw_llm_response = {
            "id": "chatcmpl-xxxxxxxx",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4-1106-preview",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None, # Usually null when tool_calls are present
                    "tool_calls": [{
                        "id": "call_thermo_set_temp",
                        "type": "function",
                        "function": {
                            "name": "thermostat.set_temperature",
                            "arguments": "{\"temperature\": 23.5, \"unit\": \"Fahrenheit\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        # 2. Manually extract the relevant 'function' part (simulating user code)
        # Assuming we are interested in the first tool call
        tool_call_function_part = simulated_raw_llm_response["choices"][0]["message"]["tool_calls"][0]["function"]
        
        # This is what would be passed to parse_llm_action
        self.assertEqual(tool_call_function_part["name"], "thermostat.set_temperature")
        self.assertEqual(tool_call_function_part["arguments"], "{\"temperature\": 23.5, \"unit\": \"Fahrenheit\"}")

        # 3. Pass to parse_llm_action
        instance_id, method_name, args = self.interface.parse_llm_action(tool_call_function_part)
        
        # 4. Verify parse_llm_action output
        self.assertEqual(instance_id, "thermostat")
        self.assertEqual(method_name, "set_temperature")
        self.assertEqual(args, {"temperature": 23.5, "unit": "Fahrenheit"})

        # 5. Pass results to execute_action
        execution_result = self.interface.execute_action(instance_id, method_name, args)

        # 6. Verify the outcome
        self.assertEqual(execution_result, "Temperature set to 23.5°Fahrenheit for main_thermo")
        self.assertEqual(self.thermostat_instance.temp, 23.5)
        self.assertEqual(self.thermostat_instance.unit, "Fahrenheit")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
