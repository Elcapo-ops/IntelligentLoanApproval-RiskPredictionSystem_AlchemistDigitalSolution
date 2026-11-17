"""Reusable metric and status presentation."""

from collections.abc import Mapping
from typing import Any

import streamlit as st


def render_metrics(items: list[Mapping[str, Any]]) -> None:
    """Render metric dictionaries as a responsive row of Streamlit metrics."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.metric(
            label=str(item["label"]),
            value=str(item["value"]),
            delta=item.get("delta"),
            help=item.get("help"),
        )
