# -*- coding: utf-8 -*-


class InstanceRegistry:
    def __init__(self):
        self.instances = {}
        self.descriptions = {}
        self.allowlists = {}

    def register_instance(
        self,
        instance_id,
        instance,
        description,
        allowlist,
    ):
        if instance_id in self.instances:
            raise ValueError(
                f"Instance ID '{instance_id}' is already registered."
            )

        # Validate methods in allowlist
        for method in allowlist.get("methods", []):
            no_method = not hasattr(instance, method)
            not_callable = not callable(getattr(instance, method))

            if no_method or not_callable:
                msg = (
                    f"Method '{method}' not found or not"
                    " callable on instance '{instance_id}'.)"
                )
                raise ValueError(msg)

        self.instances[instance_id] = instance
        self.descriptions[instance_id] = description
        self.allowlists[instance_id] = allowlist

    def get_instance(self, instance_id):
        if instance_id not in self.instances:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.instances[instance_id]

    def get_instance_description(self, instance_id):
        if instance_id not in self.descriptions:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.descriptions[instance_id]

    def get_instance_allowlist(self, instance_id):
        if instance_id not in self.allowlists:
            raise KeyError(f"Instance ID '{instance_id}' not found.")
        return self.allowlists[instance_id]

    def list_available_instances(self):
        return self.descriptions.copy()
