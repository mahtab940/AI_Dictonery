import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI


load_dotenv()


class DictionaryEntry(BaseModel):
    word: str
    definitions: list[str] = Field(min_length=1)
    synonyms: list[str]
    antonyms: list[str]
    examples: list[str]


SYSTEM_PROMPT = """
You are a dictionary assistant.

Generate a dictionary entry for the word provided by the user.

Return:
- word
- definitions
- synonyms
- antonyms
- examples

Rules:
- definitions must contain at least one item
- all fields must follow the required schema
- definitions should explain the meaning of the word
- synonyms should be related words
- antonyms should be opposite words when appropriate
- examples should demonstrate how the word is used
"""


def generate_with_openai(word: str) -> DictionaryEntry:

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Create a dictionary entry for: {word}"
            }
        ],
        text_format=DictionaryEntry,
    )

    return response.output_parsed


def generate_with_kimi(word: str) -> DictionaryEntry:

    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.ai/v1"
    )

    response = client.chat.completions.create(
        model="kimi-k3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Create a dictionary entry for: {word}"
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    data = json.loads(
        response.choices[0].message.content
    )

    return DictionaryEntry.model_validate(data)


def generate_dictionary(word: str, provider: str) -> DictionaryEntry:

    if provider == "openai":
        return generate_with_openai(word)

    elif provider == "kimi":
        return generate_with_kimi(word)

    else:
        raise ValueError(
            "Unknown provider. Choose openai or kimi."
        )


def main():

    print("AI Dictionary")
    print("================")

    provider = input(
        "Choose provider (openai/kimi): "
    ).strip().lower()

    if provider not in ["openai", "kimi"]:
        print("Invalid provider.")
        return

    word = input("Enter a word: ").strip()

    if not word:
        print("Word cannot be empty.")
        return

    try:

        entry = generate_dictionary(
            word,
            provider
        )

        print("\nDictionary entry:")

        print(
            json.dumps(
                entry.model_dump(),
                indent=2,
                ensure_ascii=False
            )
        )

    except ValidationError as e:

        print("\nValidation error:")
        print(e)

    except Exception as e:

        print("\nError:")
        print(e)


if __name__ == "__main__":
    main()