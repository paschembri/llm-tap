# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class ScheduledTrigger:
    """
    Represents a configurable, CRON-like source for time-based events.

    - name: A unique identifier for the trigger instance.
    - description: Human-friendly description of what it does.
    - cron_expression: CRON format string for when the event occurs
      (e.g., "0 6 * * *" for 6AM daily).
    - enabled: Whether the trigger is active.
    """

    name: str
    description: str
    cron_expression: str
    enabled: bool = True


@dataclass
class MessageTrigger:
    """
    Represents a configurable message/event-based trigger.

    Example use-cases:
    - Device state update via MQTT/Webhook
    - External service broadcasts an event
    - Any message bus, queue, or pub/sub topic

    Attributes:
    - name: Unique identifier for the trigger instance.
    - description: Human-friendly description (what does the message mean?).
    - topic: Message topic/channel/routing key (e.g., "sensors/tesla/plug").
    - expected_payload: Optional description or schema of message body.
    - enabled: Is listening active?
    """

    name: str
    description: str
    topic: str
    expected_payload: str = None
    enabled: bool = True
