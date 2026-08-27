from setuptools import find_packages, setup

setup(
    name="promptdiff",
    version="0.1.0",
    description="A production-grade LLM Prompt & Output Regression Tester CLI with side-by-side diffs, metrics & CI/CD assertions",
    author="promptdiff team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "typer>=0.12.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=5.0.0",
            "mypy>=1.10.0",
            "ruff>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promptdiff = promptdiff.cli.app:main",
        ],
    },
)
