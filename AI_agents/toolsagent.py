from smolagents import CodeAgent, tool, TransformersModel, HfApiModel, OpenAIServerModel
from huggingface_hub import login, InferenceClient
from huggingface_hub import login
import os

login()


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service


# Load a Hugging Face model locally
model_name = "mistralai/Ministral-8B-Instruct-2410"  # Replace with the desired model name
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
cache_dir = "/home/gsuveer/model_cache"  # Secify the cache directory

model = TransformersModel(model_id=model_name, device_map="cuda", max_new_tokens=5000)

# Define the agent using the local model
agent = CodeAgent(tools=[catering_service_tool], model=model)

# Run the agent to find the best catering service
result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
)