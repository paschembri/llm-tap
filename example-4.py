import random
from dataclasses import dataclass
from llm_tap import llm
from llm_tap.models import Node
from llm_tap.triggers import ScheduledTrigger, MessageTrigger
from llm_tap import InstanceRegistry, LLMInstanceInterface

system_prompt = """You are an automation system assistant.

Your role is to help design data structures representing the
user's query as a Workflow.

When analyzing the user's query, identify the events (triggers)
and isolate them from conditions.

To help identify if the user's query is specifying an event
(trigger) or a condition, answer the questions:

- Is this describing a moment when something changes? => Trigger/Event
- Is this asserting a *current* property? => Condition

When designing nodes, think about execution branches covering
both cases when conditions are true and when they are false.

To describe a user query, translate it into pseudo-code:

WHEN < event > happen
IF < condition_01 > AND < condition_0Z >
THEN < branch_01 >
< action_1.1 >
< action_1.2 >
ELSE
< action_2.1 >
< action_2.2 >

Answer using JSON.
"""

prompt = """
# Home Automation Environment

- Tesla charging system (on/off)
- Tesla monitoring sensors
  - Battery level in %
  - Autonomy level in miles
  - Plug status
- Electricity price broadcast
  - Price change events
  - Current price
- Time management broadcast
  - CRON-like events

# Query

> When the electricity price is below $0.4/kWh
and my Tesla is plugged, turn on charging.

# Instructions

Describe the workflow

"""

#: Use any GGUF model
# model = "~/.cache/py-llm-core/models/llama-3.1-8b"
model = "~/.cache/py-llm-core/models/qwen2.5-1.5b"


@dataclass
class TeslaMonitoringSystem:
    name: str

    def get_battery_level(self) -> float:
        return random.random()

    def get_autonomy_miles(self) -> float:
        return random.random() * 300.0

    def is_plugged(self) -> bool:
        return bool(random.randint(0, 1))


@dataclass
class Workflow:
    name: str
    description: str
    triggers: list[ScheduledTrigger | MessageTrigger]
    nodes: list[Node]


registry = InstanceRegistry()

tesla_mon01 = TeslaMonitoringSystem("tesla_mon01")

print("\n--- Registering Instances ---")
registry.register_instance(
    instance_id="tesla_mon01",
    instance=tesla_mon01,
    description="Monitor the Tesla in the garage",
    allowlist={
        "methods": [
            "get_battery_level",
            "get_autonomy_miles",
            "is_plugged",
        ]
    },
)
print("Registered: tesla_mon01")

interface = LLMInstanceInterface(registry)

all_schemas = interface.generate_all_tool_schemas()
helper_text = interface.generate_helper_text()

with llm.LLamaCPP(model=model) as parser:
    workflow = parser.parse(
        data_class=Workflow,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    print(workflow)
