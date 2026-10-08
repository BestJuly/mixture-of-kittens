import os
import shlex
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class BuildArchitectureTests(unittest.TestCase):
    def run_make(self, arch: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for name in ("ARCH", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"):
            environment.pop(name, None)
        command = [
            "make",
            "--always-make",
            "--dry-run",
            "NVCC=nvcc",
            "SRC=csrc/bindings.cu",
            "OUT=build-dry-run.so",
            "PYTHON_INCLUDES=",
            "PYTORCH_INCLUDES=",
            "PYTORCH_LIBDIR=",
            "all",
        ]
        if arch is not None:
            command.append(f"ARCH={arch}")
        return subprocess.run(
            command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
        )

    def build_flags(self, arch: str | None = None) -> list[str]:
        result = self.run_make(arch)
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(result.stdout)

    def assert_architectures(self, flags: list[str], architectures: list[str], macro: str) -> None:
        self.assertCountEqual(
            [flags[index + 1] for index, flag in enumerate(flags) if flag == "-gencode"],
            [f"arch=compute_{arch}a,code=sm_{arch}a" for arch in architectures],
        )
        self.assertEqual([flag for flag in flags if flag.startswith("-DKITTENS_SM")], [macro])

    def test_default_architecture(self) -> None:
        self.assert_architectures(self.build_flags(), ["103"], "-DKITTENS_SM103")

    def test_single_architecture(self) -> None:
        for arch in ("100", "103"):
            with self.subTest(arch=arch):
                self.assert_architectures(
                    self.build_flags(f"SM{arch}"), [arch], f"-DKITTENS_SM{arch}"
                )

    def test_all_architectures(self) -> None:
        flags = self.build_flags("ALL")
        self.assert_architectures(flags, ["100", "103"], "-DKITTENS_SM100")
        self.assertIn("--threads=2", flags)

    def test_unsupported_architecture(self) -> None:
        result = self.run_make("SM90")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported ARCH 'SM90'", result.stderr)
        for supported in ("SM100", "SM103", "ALL"):
            self.assertIn(supported, result.stderr)


if __name__ == "__main__":
    unittest.main()
