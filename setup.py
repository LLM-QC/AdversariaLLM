from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path: Path) -> list[str]:
    requirements = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


ROOT = Path(__file__).resolve().parent

setup(
    name="adversariallm",
    version="0.0.1",
    description="Toolkit for evaluating and comparing adversarial attacks on LLMs.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/LLM-QC/AdversariaLLM",
    packages=find_packages(include=["adversariallm", "adversariallm.*"]),
    include_package_data=False,
    install_requires=read_requirements(ROOT / "requirements.txt"),
    python_requires=">=3.10",
)
