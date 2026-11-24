from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

print("Starting debug runner")
llm = ChatOpenAI(model="gpt-5")
print("LLM instantiated:", llm)

try:
    response = llm.invoke("Tell me a joke about programmers.")
    print("Got response object")
    print("type:", type(response))
    print("repr:", repr(response))
    print("str:", str(response))
    # If it's dict-like, print keys
    try:
        print("dir(response):", [k for k in dir(response) if not k.startswith('_')])
    except Exception:
        pass
except Exception as e:
    print("Exception while invoking LLM:", e)
