import os
import vertexai
import openai
import backoff
import requests

import xmlrpc.client

from google.auth import default, transport
from dataclasses import dataclass, field
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI, OpenAI

from together import Together


@backoff.on_exception(backoff.expo, (openai.RateLimitError, requests.exceptions.ConnectionError, openai.APIConnectionError), max_tries=3)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


@dataclass
class Chatbot:
    """A conversational chatbot"""

    conversation_history: list = field(default_factory=list)

    def ask(self, query: str) -> str:
        pass

    def print_conversation(self) -> None:
        for item in self.conversation_history:
            print(f"{item['role']}: {item['content']}")

    def delete_history(self) -> None:
        self.conversation_history = []


class LocalChatbot(Chatbot):
    """A conversational chatbot that has access to a dataset"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []

    def delete_history(self):
        self.conversation_history = []

    def ask(self, query, max_gen_length=1000):
        if len(self.conversation_history) == 0:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
        else:
            input_ids = self.tokenizer.apply_chat_template(
                self.conversation_history + [{"role": "user", "content": query}],
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")

        output = self.model.generate(
            **input_ids,
            max_length=input_ids["input_ids"].shape[1] + max_gen_length,
            return_dict_in_generate=True,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        num_special_tokens = 4
        # Don't include last token if it is a special token
        output_string = self.tokenizer.decode(
            output["sequences"][0][
                len(input_ids["input_ids"][0]) + num_special_tokens :
            ],
            skip_special_tokens=True,
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": output_string}
        )
        return output_string

    def print_conversation(self):
        for item in self.conversation_history:
            print(f"{item['role']}: {item['content']}")

    def gather_code(self):
        """Gather the code from the last message in the conversation history, if it exists"""
        if len(self.conversation_history) == 0:
            return None
        last_message = self.conversation_history[-1]["content"]
        # code is between '```python' and '```'
        code_start = last_message.find("```python")
        if code_start == -1:
            return None
        code_start += len("```python")
        code_end = last_message[code_start:].find("```")
        if code_end == -1:
            return None
        code_end += code_start
        code = last_message[code_start:code_end]
        return code

    def run_code(self, code=None):
        """Runs code called by gather_code, outputs the result.
        Use exec for now, TODO: add docker container for security"""
        if code is None:
            code = self.gather_code()
            if code is None:
                return
        try:
            exec(code)
        except Exception as e:
            print(f"Error running code: {e}")


class VertexAPIChatbot(Chatbot):
    """A conversational chatbot that uses the Vertex API"""

    def __init__(
        self, model="google/gemini-1.5-flash-002", project_id=None, location=None, persistent_mode=False
    ):
        self.model = model
        self.conversation_history = []
        self.persistent_mode = persistent_mode

        # Load environment variables
        load_dotenv(find_dotenv())

        # Set up Vertex AI
        if project_id is None:
            project_id = os.getenv("PROJECT_ID")
        if location is None:
            location = os.getenv("LOCATION")

        vertexai.init(project=project_id, location=location)
        # Programmatically get an access token
        credentials, _ = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        # OpenAI Client
        self.client = openai.OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
            api_key=credentials.token,
        )
        
        self.project_id = project_id
        self.location = location

    def ask(self, query):
        # Add the query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Create the system message with persistent mode info if enabled
        system_content = "You are a helpful assistant."
        if self.persistent_mode:
            system_content += " You have access to a persistent Python environment where variables and loaded libraries remain available between code executions. You can write and execute code incrementally, inspect intermediate results, and build upon previous computations."

        # Create the messages for the API
        if not self.conversation_history[:-1]:  # If this is the first message
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "system", "content": system_content}] + self.conversation_history

        completion = completions_with_backoff(
            client=self.client, model=self.model, messages=messages
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content


class TogetherAPIChatbot(Chatbot):
    """A conversational chatbot that uses the Together API"""

    def __init__(self, model, persistent_mode=False):
        self.model = model
        self.conversation_history = []
        self.persistent_mode = persistent_mode

        # Load environment variables
        load_dotenv(find_dotenv())

        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    def ask(self, query):
        # Add the query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Create the system message with persistent mode info if enabled
        system_content = "You are a helpful assistant."
        if self.persistent_mode:
            system_content += " You have access to a persistent Python environment where variables and loaded libraries remain available between code executions. You can write and execute code incrementally, inspect intermediate results, and build upon previous computations."

        # Create the messages for the API
        if not self.conversation_history[:-1]:  # If this is the first message
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "system", "content": system_content}] + self.conversation_history

        completion = completions_with_backoff(
            client=self.client, model=self.model, messages=messages
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content


class AzureAPIChatbot(Chatbot):
    """A conversational chatbot that uses the Azure API"""

    def __init__(self, model, persistent_mode=False):
        self.model = model
        self.conversation_history = []
        self.persistent_mode = persistent_mode

        # Load environment variables
        load_dotenv(find_dotenv())

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

    def ask(self, query):
        # Add the query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Create the system message with persistent mode info if enabled
        system_content = "You are a helpful assistant."
        if self.persistent_mode:
            system_content += " You have access to a persistent Python environment where variables and loaded libraries remain available between code executions. You can write and execute code incrementally, inspect intermediate results, and build upon previous computations."

        # Create the messages for the API
        if not self.conversation_history[:-1]:  # If this is the first message
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "system", "content": system_content}] + self.conversation_history

        completion = completions_with_backoff(
            client=self.client, model=self.model, messages=messages
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content


class OpenAIAPIChatbot(Chatbot):
    """A conversational chatbot that uses the OpenAI API"""

    def __init__(self, model, persistent_mode=False):
        self.model = model
        self.conversation_history = []
        self.persistent_mode = persistent_mode

        # Load environment variables
        load_dotenv(find_dotenv())

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def ask(self, query):
        # Add the query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Create the system message with persistent mode info if enabled
        system_content = "You are a helpful assistant."
        if self.persistent_mode:
            system_content += " You have access to a persistent Python environment where variables and loaded libraries remain available between code executions. You can write and execute code incrementally, inspect intermediate results, and build upon previous computations."

        # Create the messages for the API
        if not self.conversation_history[:-1]:  # If this is the first message
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "system", "content": system_content}] + self.conversation_history

        completion = completions_with_backoff(
            client=self.client,
            model=self.model,
            messages=messages,
            seed=0,  # temperature=0
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content


class RPCChatbot(Chatbot):
    """A conversational chatbot that uses an XML-RPC server"""

    def __init__(self, address: str):
        self.address = address
        self.conversation_history = []

    def ask(self, query):
        try:
            with xmlrpc.client.ServerProxy(self.address) as proxy:

                response = proxy.ask(
                    self.conversation_history + [{"role": "user", "content": query}]
                )
                response = response.data.decode("utf-8")

                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )

                return response
        except Exception as e:
            print(f"Error: {e}")
            return None


class TestChatbot(Chatbot):
    """A conversational chatbot for testing"""

    def __init__(self):
        self.conversation_history = []

    def print_conversation(self):
        for item in self.conversation_history:
            print(f"{item['role']}: {item['content']}")

    def ask(self, query):
        response = """This is a test response.
```python
import dowhy
print(0.05)
```
This is the end of the test response."""
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
