import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.markers import default_environment
from packaging.requirements import Requirement


class SetupTest(unittest.TestCase):

    def test_tensorflow_dependency_matches_installing_platform(self):
        root = Path(__file__).parents[1]
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            with patch("setuptools.setup") as setup:
                runpy.run_path(str(root / "setup.py"), run_name="__main__")
        finally:
            os.chdir(previous_directory)

        requirements = [
            Requirement(value)
            for value in setup.call_args.kwargs["install_requires"]
        ]

        linux = default_environment()
        linux.update(platform_system="Linux", platform_machine="x86_64")
        linux_dependencies = {
            requirement.name
            for requirement in requirements
            if requirement.marker is None or requirement.marker.evaluate(linux)
        }
        self.assertIn("tensorflow", linux_dependencies)
        self.assertNotIn("tensorflow-macos", linux_dependencies)

        apple_silicon = default_environment()
        apple_silicon.update(platform_system="Darwin", platform_machine="arm64")
        apple_silicon_dependencies = {
            requirement.name
            for requirement in requirements
            if requirement.marker is None
            or requirement.marker.evaluate(apple_silicon)
        }
        self.assertNotIn("tensorflow", apple_silicon_dependencies)
        self.assertIn("tensorflow-macos", apple_silicon_dependencies)


if __name__ == "__main__":
    unittest.main()
