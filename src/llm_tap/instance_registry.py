from typing import Any, Dict, List, Union

class InstanceRegistry:
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self.descriptions: Dict[str, str] = {}
        self.allowlists: Dict[str, Dict[str, Union[List[str], Dict[str, str]]]] = {}

    def register_instance(
        self,
        instance_id: str,
        instance: Any,
        description: str,
        allowlist: Dict[str, Union[List[str], Dict[str, str]]],
    ):
        if instance_id in self.instances:
            raise ValueError(f"Instance ID '{instance_id}' is already registered.")

        # Validate methods in allowlist
        for method_name in allowlist.get("methods", []):
            if not hasattr(instance, method_name) or not callable(
                getattr(instance, method_name)
            ):
                raise ValueError(
                    f"Method '{method_name}' not found or not callable on instance '{instance_id}'."
                )

        self.instances[instance_id] = instance
        self.descriptions[instance_id] = description
        self.allowlists[instance_id] = allowlist

    def get_instance(self, instance_id: str) -> Any:
        if instance_id not in self.instances:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.instances[instance_id]

    def get_instance_description(self, instance_id: str) -> str:
        if instance_id not in self.descriptions:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.descriptions[instance_id]

    def get_instance_allowlist(
        self, instance_id: str
    ) -> Dict[str, Union[List[str], Dict[str, str]]]:
        if instance_id not in self.allowlists:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.allowlists[instance_id]

    def list_available_instances(self) -> Dict[str, str]:
        return self.descriptions.copy()
