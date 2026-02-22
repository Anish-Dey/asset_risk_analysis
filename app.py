import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from IPython.display import display, Markdown
import os
load_dotenv()


def load_data(path):
    df = pd.read_csv(path)
    return df


df = load_data("asset_risk_data_asset_lim.csv")
# print(df.head())

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    api_key = "AIzaSyBtFhm_UfOypIWMF_0Y1FFBfJqPtVm21aI"
)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

SYSTEM_CONTEXT = """
You are an Asset Risk Analyst in the Power and Utilities sector.

You analyze asset risk data based on:
- Age
- Asset Health Index (AHI)
- Probability of Failure
- COF
- Risk = PoF × COF

Always respond in professional business language.
All monetary values are in AED.
"""

prompt_template = PromptTemplate(
    input_variables=["input"],
    template=SYSTEM_CONTEXT + "\n\nUser Question: {input}"
)

def chat():

    print("Utility Asset Risk Assistant (Gemini + LangChain)")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask: ")

        if query.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        try:
            full_prompt = prompt_template.format(input=query)
            response = agent.run(full_prompt)

            print("\nAnswer:")
            display(Markdown(response))
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == '__main__':
    chat()