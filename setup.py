import os

import setuptools

# Read version without importing the package (it's not installed yet).
# This is the same trick DataHub uses in datahub-agent-context/setup.py:21-23.
package_metadata: dict = {}
with open("./src/data_chat/_version.py") as fp:
    exec(fp.read(), package_metadata)


def get_long_description():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, "README.md")) as f:
        return f.read()


# --- Dependency groups ---
# DataHub splits deps into groups so users only install what they need.
# `pip install data-chat` gets base_requirements only.
# `pip install data-chat[langchain]` adds langchain-core.

base_requirements = {
    "snowflake-connector-python>=3.0.0,<5.0.0",
    "pydantic>=2.0.0,<3.0.0",
}

langchain_requirements = {
    "langchain-core>=1.2.7,<2.0.0",
}

google_adk_requirements = {
    "google-adk>=1.0.0,<2.0.0",
}

agent_requirements = {
    "anthropic>=0.40.0,<1.0.0",
}

streamlit_requirements = {
    "streamlit>=1.30.0,<2.0.0",
    *agent_requirements,
}

dev_requirements = {
    "ruff==0.11.7",
    "mypy==1.17.1",
    "pytest>=8.3.4,<9.0.0",
    *langchain_requirements,
    *google_adk_requirements,
    *agent_requirements,
}

setuptools.setup(
    name=package_metadata["__package_name__"],
    version=package_metadata["__version__"],
    description="Data Chat - AI agent tools for querying Snowflake",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    # src layout: packages live under src/, not at repo root.
    # This prevents accidental imports from the working directory.
    packages=setuptools.find_namespace_packages(where="./src"),
    package_dir={"": "src"},
    zip_safe=False,
    install_requires=list(base_requirements),
    extras_require={
        "dev": list(dev_requirements),
        "langchain": list(langchain_requirements),
        "google-adk": list(google_adk_requirements),
        "agent": list(agent_requirements),
        "streamlit": list(streamlit_requirements),
    },
)
