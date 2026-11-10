"""Unit tests for CLI argument parsing and command dispatch."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from papaya_cli.cli import build_parser, main


class TestCLI(unittest.TestCase):
    def test_parser_accepts_prompt_examples(self) -> None:
        parser = build_parser()

        a1 = parser.parse_args(["scrape", "--config", "config.yaml", "--per-class", "150"])
        self.assertEqual(a1.command, "scrape")
        self.assertEqual(a1.per_class, 150)

        a2 = parser.parse_args(["clean", "--input", "data/raw"])
        self.assertEqual(a2.command, "clean")
        self.assertEqual(a2.input, Path("data/raw"))

        a3 = parser.parse_args(
            ["annotate", "--input", "data/clean", "--conf", "0.35", "--preview"]
        )
        self.assertEqual(a3.command, "annotate")
        self.assertAlmostEqual(a3.conf, 0.35)
        self.assertTrue(a3.preview)

        a4 = parser.parse_args(["export", "--format", "yolo", "--split", "70", "20", "10"])
        self.assertEqual(a4.command, "export")
        self.assertEqual(a4.format, "yolo")
        self.assertEqual(a4.split, [70.0, 20.0, 10.0])

        a5 = parser.parse_args(["run-all", "--config", "config.yaml"])
        self.assertEqual(a5.command, "run-all")

    def test_run_all_invokes_all_pipeline_stages(self) -> None:
        with (
            patch("papaya_cli.cli.load_config", return_value={}),
            patch("papaya_cli.cli.scrape_dataset") as m_scrape,
            patch("papaya_cli.cli.clean_dataset") as m_clean,
            patch("papaya_cli.cli.annotate_dataset") as m_ann,
            patch("papaya_cli.cli.export_dataset") as m_exp,
        ):
            rc = main(["run-all", "--config", "config.yaml", "--per-class", "10"])

        self.assertEqual(rc, 0)
        m_scrape.assert_called_once()
        m_clean.assert_called_once()
        m_ann.assert_called_once()
        m_exp.assert_called_once()


if __name__ == "__main__":
    unittest.main()

