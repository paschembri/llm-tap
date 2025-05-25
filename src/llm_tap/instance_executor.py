# -*- coding: utf-8 -*-
import dataclasses
import inspect
import json


from llm_tap.instance_registry import InstanceRegistry
from llm_tap.llm import as_tool, to_json_schema


class LLMInstanceInterface:
    def __init__(self, registry: InstanceRegistry):
        self.registry = registry

    def _get_method_parameters_dataclass(self, method):
        fields = []
        sig = inspect.signature(method)
        for name, param in sig.parameters.items():
            if name == "self":  # Skip self parameter
                continue
            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else str
            )
            default_value = (
                param.default
                if param.default != inspect.Parameter.empty
                else dataclasses.MISSING
            )
            fields.append((name, param_type, default_value))

        # Ensure unique dataclass name if method is overloaded or for clarity
        dc_name = f"{method.__name__}_{id(method)}_params"
        return dataclasses.make_dataclass(dc_name, fields)

    def generate_tool_schemas_for_instance(self, instance_id):
        instance = self.registry.get_instance(instance_id)
        allowlist = self.registry.get_instance_allowlist(instance_id)
        instance_description = self.registry.get_instance_description(
            instance_id
        )
        tool_schemas = []

        for method_name in allowlist.get("methods", []):
            method = getattr(instance, method_name)
            params_dataclass = self._get_method_parameters_dataclass(method)

            # Assuming llm_tap.llm.to_json_schema and llm_tap.llm.as_tool are available
            # and handle dataclasses correctly.
            json_schema = to_json_schema(params_dataclass)
            # This should return a dict based on typical OpenAI tool structure
            tool_spec = as_tool(json_schema)

            tool_spec["function"]["name"] = f"{instance_id}.{method_name}"

            method_docstring = (
                inspect.getdoc(method) or "No detailed description available."
            )
            tool_spec["function"]["description"] = (
                f"Calls the '{method_name}' method on '{instance_id}' ({instance_description}). "
                f"Original method description: {method_docstring}"
            )
            tool_schemas.append(tool_spec)
        return tool_schemas

    def generate_all_tool_schemas(self):
        all_schemas = []
        for instance_id in self.registry.list_available_instances():
            all_schemas.extend(
                self.generate_tool_schemas_for_instance(instance_id)
            )
        return all_schemas

    def generate_helper_text(self):
        helper_lines = [
            "You can interact with the following pre-existing objects:"
        ]
        for (
            instance_id,
            description,
        ) in self.registry.list_available_instances().items():
            helper_lines.append(f"- Object ID: '{instance_id}'")
            helper_lines.append(f"  Description: {description}")
            helper_lines.append("  Available Actions (Methods):")

            instance = self.registry.get_instance(instance_id)
            allowlist = self.registry.get_instance_allowlist(instance_id)

            for method_name in allowlist.get("methods", []):
                method = getattr(instance, method_name)
                sig = inspect.signature(method)
                params = []
                for p_name, p_obj in sig.parameters.items():
                    if p_name == "self":
                        continue
                    param_str = f"{p_name}: {p_obj.annotation if p_obj.annotation != inspect.Parameter.empty else 'str'}"
                    if p_obj.default != inspect.Parameter.empty:
                        param_str += f" = {p_obj.default!r}"  # Use !r for default to show strings correctly
                    params.append(param_str)
                params_str = ", ".join(params)

                method_doc = inspect.getdoc(method)
                doc_str = f" ({method_doc})" if method_doc else ""
                helper_lines.append(
                    f"    - {method_name}({params_str}){doc_str}"
                )
        return "\n".join(helper_lines)

    def parse_llm_action(self, llm_output):
        try:
            name_parts = llm_output["name"].split(".", 1)
            if len(name_parts) != 2:
                raise ValueError(
                    "LLM output name is not in 'instance_id.method_name' format."
                )
            instance_id, method_name = name_parts

            # arguments can be a string or already a dict
            arguments_str = llm_output.get("arguments", "{}")
            if isinstance(arguments_str, dict):
                parsed_args = arguments_str
            else:
                parsed_args = json.loads(arguments_str)

        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"Failed to parse LLM action output: {llm_output}. Error: {e}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse arguments JSON string: {arguments_str}. Error: {e}"
            )

        return instance_id, method_name, parsed_args

    def execute_action(self, instance_id, method_name, arguments):
        instance = self.registry.get_instance(
            instance_id
        )  # Raises KeyError if not found
        allowlist = self.registry.get_instance_allowlist(instance_id)

        if method_name not in allowlist.get("methods", []):
            raise PermissionError(
                f"Method '{method_name}' is not allowlisted for instance '{instance_id}'."
            )

        method = getattr(instance, method_name)

        # Optional: Argument validation could be added here.
        # For example, regenerate dataclass and use llm_tap.llm.from_dict
        # params_dataclass = self._get_method_parameters_dataclass(method)
        # try:
        #     validated_args = from_dict(params_dataclass, arguments) # Assuming from_dict exists
        #     args_to_pass = dataclasses.asdict(validated_args)
        # except Exception as e: # Catch specific validation error
        #     raise ValueError(f"Argument validation failed for {method_name}: {e}")

        try:
            result = method(**arguments)
        except Exception as e:
            # Log or wrap the exception as needed
            # For now, re-raise to provide direct feedback
            raise e
            # Example wrapping:
            # raise RuntimeError(f"Error executing {instance_id}.{method_name}: {e}") from e

        return result
