"""Temporal DataConverter factory for the Exo distributed integration.

Provides :func:`build_data_converter`, which returns a :class:`DataConverter`
backed by the Pydantic payload converter so that ``TaskPayload``, agent configs,
and domain events serialise cleanly.  An optional :class:`PayloadCodec` can be
attached for encryption or claim-check offloading.

Usage::

    from exo.distributed.temporal.codec import AESGCMPayloadCodec  # pyright: ignore[reportMissingImports]
    from exo.distributed.temporal.converter import build_data_converter  # pyright: ignore[reportMissingImports]

    codec = AESGCMPayloadCodec.from_env()
    data_converter = build_data_converter(codec=codec)
    # Pass data_converter to temporalio.client.Client(data_converter=...) or
    # temporalio.worker.Worker(data_converter=...).
"""

from __future__ import annotations

import dataclasses
import logging

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    DataConverter,
    PayloadCodec,
    pydantic_data_converter,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_data_converter",
    "pydantic_data_converter",
]


def build_data_converter(codec: PayloadCodec | None = None) -> DataConverter:
    """Return a :class:`DataConverter` that uses the Pydantic payload converter.

    When *codec* is provided it is attached as the ``payload_codec`` on the
    returned converter (via :func:`dataclasses.replace` on the pre-built
    ``pydantic_data_converter`` singleton).  When *codec* is ``None`` the plain
    ``pydantic_data_converter`` is returned unchanged.

    Args:
        codec: An optional :class:`PayloadCodec` (e.g. :class:`~exo.distributed.temporal.codec.AESGCMPayloadCodec`
               or :class:`~exo.distributed.temporal.codec.ChainedPayloadCodec`).

    Returns:
        A :class:`DataConverter` ready to pass to the Temporal client or worker.
    """
    if codec is None:
        return pydantic_data_converter
    return dataclasses.replace(pydantic_data_converter, payload_codec=codec)
