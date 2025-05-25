import json
from typing import Any, Dict, Optional
from llm_tap import InstanceRegistry, LLMInstanceInterface


# --- Example Classes ---
class Thermostat:
    """Manages the temperature of a space."""

    def __init__(self, location):
        self.location = location
        self.current_temperature = 20.0
        self.target_temperature = 22.0
        self.unit = "Celsius"
        self.mode = "heat"  # "heat", "cool", "off"

    def get_current_temperature(self) -> float:
        """Returns the current temperature."""
        print(
            f"[Thermostat {self.location}] Getting current temperature: {self.current_temperature}°{self.unit}"
        )
        return self.current_temperature

    def set_temperature(self, temperature: float, unit: Optional[str] = None):
        """Sets the target temperature.

        Args:
            temperature: The target temperature value.
            unit: The temperature unit (e.g., 'Celsius', 'Fahrenheit'). Defaults to current unit.
        """
        old_temp = self.target_temperature
        self.target_temperature = temperature
        if unit:
            self.unit = unit
        print(
            f"[Thermostat {self.location}] Setting target temperature from {old_temp}° to {self.target_temperature}°{self.unit}"
        )
        return f"Temperature set to {self.target_temperature}°{self.unit}"

    def change_mode(self, mode: str) -> str:
        """Changes the operating mode of the thermostat.

        Args:
            mode: The new mode (e.g., 'heat', 'cool', 'off').
        """
        if mode not in ["heat", "cool", "off"]:
            raise ValueError("Invalid mode. Must be 'heat', 'cool', or 'off'.")
        self.mode = mode
        print(f"[Thermostat {self.location}] Mode changed to {self.mode}")
        return f"Mode changed to {self.mode}"


class LightSwitch:
    """Controls a light fixture."""

    def __init__(self, name: str):
        self.name = name
        self.is_on_state = False
        self.brightness = 100

    def turn_on(self, brightness: Optional[int] = None) -> str:
        """Turns the light on.

        Args:
            brightness: Optional brightness level (0-100). Defaults to previous or 100.
        """
        self.is_on_state = True
        if brightness is not None:
            if not 0 <= brightness <= 100:
                raise ValueError("Brightness must be between 0 and 100.")
            self.brightness = brightness
        print(f"[Light {self.name}] Turned ON. Brightness: {self.brightness}%")
        return f"{self.name} light is ON, brightness {self.brightness}%"

    def turn_off(self) -> str:
        """Turns the light off."""
        self.is_on_state = False
        print(f"[Light {self.name}] Turned OFF.")
        return f"{self.name} light is OFF"

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the light (on/off, brightness)."""
        status = {
            "name": self.name,
            "is_on": self.is_on_state,
            "brightness": self.brightness if self.is_on_state else 0,
        }
        print(f"[Light {self.name}] Status: {status}")
        return status


def main():
    print("--- LLM Instance Interaction Model Example ---")

    # 1. Create InstanceRegistry
    registry = InstanceRegistry()

    # 2. Instantiate example objects
    thermostat_living_room = Thermostat("Living Room")
    light_kitchen = LightSwitch("Kitchen Light")
    light_bedroom = LightSwitch("Bedroom Lamp")

    # 3. Register instances
    print("\n--- Registering Instances ---")
    registry.register_instance(
        instance_id="thermostat_lr",
        instance=thermostat_living_room,
        description="Manages temperature in the living room.",
        allowlist={
            "methods": [
                "get_current_temperature",
                "set_temperature",
                "change_mode",
            ]
        },
    )
    print("Registered: thermostat_lr")

    registry.register_instance(
        instance_id="light_kitchen",
        instance=light_kitchen,
        description="Controls the main light in the kitchen.",
        allowlist={"methods": ["turn_on", "turn_off", "get_status"]},
    )
    print("Registered: light_kitchen")

    registry.register_instance(
        instance_id="light_bedroom",
        instance=light_bedroom,
        description="Controls the bedside lamp in the bedroom.",
        allowlist={
            "methods": ["turn_on", "turn_off"]
        },  # get_status is NOT allowlisted
    )
    print("Registered: light_bedroom")

    # 4. Create LLMInstanceInterface
    interface = LLMInstanceInterface(registry)

    # 5. Generate Tool Schemas for the LLM
    print("\n--- Generated Tool Schemas (for LLM) ---")
    all_schemas = interface.generate_all_tool_schemas()
    print(json.dumps(all_schemas, indent=2))

    # 6. Generate Helper Text for the LLM
    print("\n--- Generated Helper Text (for LLM Prompt) ---")
    helper_text = interface.generate_helper_text()
    print(helper_text)

    breakpoint()

    # 7. Simulate LLM calling a method
    print("\n--- Simulating LLM Action: Set Living Room Temperature ---")
    # This is what the LLM would output (after my system parses it into this dict)
    simulated_llm_call_set_temp = {
        "name": "thermostat_lr.set_temperature",
        "arguments": json.dumps(
            {"temperature": 23.5, "unit": "Celsius"}
        ),  # Arguments as JSON string
    }

    try:
        instance_id, method_name, args = interface.parse_llm_action(
            simulated_llm_call_set_temp
        )
        print(
            f"Parsed Action: Instance='{instance_id}', Method='{method_name}', Args={args}"
        )

        result = interface.execute_action(instance_id, method_name, args)
        print(f"Execution Result: {result}")
    except Exception as e:
        print(f"Error during action execution: {e}")

    print("\n--- Simulating LLM Action: Turn Kitchen Light On ---")
    simulated_llm_call_light_on = {
        "name": "light_kitchen.turn_on",
        "arguments": json.dumps({"brightness": 75}),
    }
    try:
        instance_id, method_name, args = interface.parse_llm_action(
            simulated_llm_call_light_on
        )
        print(
            f"Parsed Action: Instance='{instance_id}', Method='{method_name}', Args={args}"
        )

        result = interface.execute_action(instance_id, method_name, args)
        print(f"Execution Result: {result}")
    except Exception as e:
        print(f"Error during action execution: {e}")

    print(
        "\n--- Simulating LLM Action: Get Bedroom Light Status (Should Fail - Not Allowlisted) ---"
    )
    simulated_llm_call_bedroom_status = {
        "name": "light_bedroom.get_status",  # get_status is not in allowlist for light_bedroom
        "arguments": json.dumps({}),
    }
    try:
        instance_id, method_name, args = interface.parse_llm_action(
            simulated_llm_call_bedroom_status
        )
        print(
            f"Parsed Action: Instance='{instance_id}', Method='{method_name}', Args={args}"
        )

        result = interface.execute_action(instance_id, method_name, args)
        print(f"Execution Result: {result}")
    except Exception as e:
        print(f"Error during action execution: {type(e).__name__}: {e}")

    print(
        "\n--- Simulating LLM Action: Call Non-Existent Method (Should Fail) ---"
    )
    simulated_llm_call_fake_method = {
        "name": "thermostat_lr.boost_heating",  # This method doesn't exist
        "arguments": json.dumps({}),
    }
    try:
        instance_id, method_name, args = interface.parse_llm_action(
            simulated_llm_call_fake_method
        )
        print(
            f"Parsed Action: Instance='{instance_id}', Method='{method_name}', Args={args}"
        )

        result = interface.execute_action(instance_id, method_name, args)
        print(f"Execution Result: {result}")
    except Exception as e:
        print(f"Error during action execution: {type(e).__name__}: {e}")

    print(
        "\n--- Simulating LLM Action: Method call with invalid parameter value ---"
    )
    simulated_llm_call_invalid_param = {
        "name": "thermostat_lr.change_mode",
        "arguments": json.dumps({"mode": "super_heat"}),  # Invalid mode
    }
    try:
        instance_id, method_name, args = interface.parse_llm_action(
            simulated_llm_call_invalid_param
        )
        print(
            f"Parsed Action: Instance='{instance_id}', Method='{method_name}', Args={args}"
        )

        result = interface.execute_action(instance_id, method_name, args)
        print(f"Execution Result: {result}")
    except Exception as e:
        print(f"Error during action execution: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
