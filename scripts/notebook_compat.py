"""Tiny nbformat-compatible fallback used only when nbformat is unavailable."""
from __future__ import annotations

import json
from pathlib import Path


class _V4:
    @staticmethod
    def new_notebook():
        return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}

    @staticmethod
    def new_markdown_cell(source=""):
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    @staticmethod
    def new_code_cell(source=""):
        return {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source,
        }


class _NotebookFormat:
    v4 = _V4()

    @staticmethod
    def write(notebook, path):
        Path(path).write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


nbf = _NotebookFormat()
