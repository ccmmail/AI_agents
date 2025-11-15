import langchain
import tiktoken
import json
import pprint
import os
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.evaluation.scoring import LabeledScoreStringEvalChain
from langchain.evaluation.scoring.eval_chain import ScoreStringResultOutputParser
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

# setup the environment
# langchain.debug = True
client = Client()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "scratch"
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
PROMPTS = {
    "Unusual" : "What are the best unusual attractions and things to do in {city}?",
    "Food" : "What are the best local food to eat in {city}?"
}
CRITERIA_TEXT_SCORE = (
    "a) **Correctness** – Does the answer provide recommendations that *match* the reference response?\n"
    "- 0: No overlap with reference ideas; mostly generic or irrelevant\n"
    "- 1: One idea overlaps partially with a reference point\n"
    "- 2: Multiple partial overlaps, but many important ideas missing\n"
    "- 3: Substantial overlap with reference, but some key omissions\n"
    "- 4: Near-complete coverage of specific reference items, but some omissions\n"
    "- 5: Covers all reference ideas, but misses some less obvious ones\n" 
    "\n"
    "b) **Interesting** – Does the answer avoid standard tourist traps?\n"
    "- 0: Focuses entirely on standard tourist recommendations (e.g., Sky Tower, harbor cruise)\n"
    "- 1: Mostly standard recommendations with minor novelty or rewording\n"
    "- 2: Majority of list are standard tourist recommendations, with unusual attractions being less than half the ideas\n"
    "- 3: Half the list are unusual ideas, with the other half being standard tourist recommendations\n"
    "- 4: Strong set of mostly unusual ideas, small overlap with standard tourist recommendations\n"
    "- 5: Entirely composed of offbeat, creative, or niche recommendations that would surprise a typical tourist\n"
    "\n"
    "Be strict in your evaluation.\n"
    "\n"
    "Sum the scores for correctness and interesting to get the total score.\n"
    "Output on a new line for each and in the following order, the individual scoring for correctness and interesting.\n"
    "Then on a new line, output the total score **alone**, wrapped in double brackets, e.g. [[7]].\n"
)
CRITERIA_TEXT_PASS_FAIL = {
    "Correctness": "Does the answer provide *specific* recommendations that match the reference response?",
    "Interesting": "Does the answer avoid standard tourist traps?",
    "Believability": "Does the answer justify why the recommendation is interesting?"
}


def load_file(filename):
    """Load gold responses from a file."""
    with open(filename, "r") as file:
        content = json.load(file)
    return content


def save_to_file(content, filename):
    """Save content to a file."""
    with open(filename, "w") as file:
        json.dump(content, file, indent=4)
    print(f"Saved content to {filename}")


def print_responses(responses):
    """Print the responses."""
    for item in responses:
        print(item['prompt'])
        print(item['response'])
        print()


def get_token_count(message):
    """Calculate the number of tokens in a message."""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    token_count = len(encoding.encode(message))
    return token_count


def prompt_LLM(prompt_name, city):
    """Prompt LLM using specific template and city parameter."""
    template = ChatPromptTemplate.from_template(PROMPTS[prompt_name])
    messages = template.format_messages(city=city)
    print("Prompting LLM with:", messages[0].content)
    response = LLM(messages)
    return messages[0].content, response.content


def evaluate_response_on_pass_fail(input, prediction, reference):
    """Evaluate the response using LangChain's LabeledCriteriaEvalChain."""
    evaluator = LabeledCriteriaEvalChain.from_llm(
        llm = LLM,
        criteria = CRITERIA_TEXT_SCORE,
    )
    evaluation = evaluator.evaluate_strings(
        input = input,
        prediction = prediction,
        reference = reference
    )
    return evaluation


class ReasoningAndScoreParser(ScoreStringResultOutputParser):
    _regex = re.compile(r"\[\[([1-9]|10)\]\]")

    def parse(self, text: str) -> dict:
        match = self._regex.search(text)
        if not match:
            raise ValueError("Could not find score in double brackets.")
        score = int(match.group(1))
        reasoning = text[: match.start()].strip()
        return {"value": f"[[{score}]]", "score": score, "reasoning": reasoning}


class MyLabeledScorer(LabeledScoreStringEvalChain):
    def _apply_criteria_to_prompt(self, prompt, criteria):
        return prompt  # do nothing


def evaluate_response_on_scale(input, prediction, reference):
    """Evaluate the response using LangChain's LabeledScoreEvalChain."""
    score_prompt = PromptTemplate(
        input_variables=["input", "prediction", "reference", "criteria"],
        template=(
            "### Input\n{input}\n\n"
            "### Submission\n{prediction}\n\n"
            "### Reference\n{reference}\n\n"
            "### Criteria\n{criteria}\n\n"
        )
    )
    evaluator = MyLabeledScorer.from_llm(
        llm=LLM,
        prompt=score_prompt,
        criteria=CRITERIA_TEXT_SCORE,
        output_parser=ReasoningAndScoreParser()
    )
    evaluation = evaluator.evaluate_strings(
        input=input,
        prediction=prediction,
        reference=reference,
    )
    return evaluation


# def evaluate_response_scale(input, prediction, reference):
#     """Evaluate the response using the raw LLM call."""
#     score_prompt = (
#         f"### Input\n{input}\n\n"
#         f"### Submission\n{prediction}\n\n"
#         f"### Reference\n{reference}\n\n"
#         f"### Evaluation Criteria\n{CRITERIA_TEXT_SCORE}\n\n"
#         )
#     response = LLM.invoke(score_prompt)
#     evaluation = ReasoningAndScoreParser().parse(response.content)
#     return evaluation


def calculate_recall(response, gold_response):
    """Assess the relevance of the response compared to the gold response."""
    encoded_gold_response = tiktoken.encoding_for_model("gpt-4o-mini").encode(gold_response)
    encoded_response = tiktoken.encoding_for_model("gpt-4o-mini").encode(response)
    recall = len(set(encoded_gold_response).intersection(set(encoded_response))) / len(encoded_gold_response)
    return recall


def main():
    """Main function to run the LLM evaluation."""
    gold_responses = load_file("LLM_evals_GPT4o-responses.json")
    saved_responses = load_file("LLM_evals_GPT4o-mini-responses.json")

    # get responses from LLM
    responses = []
    for prompt_name in PROMPTS:
        for city in ["Auckland", "Melbourne"]:
            prompt, response = prompt_LLM(prompt_name, city)
            responses.append({"prompt" : prompt,
                              "response" : response
                              }
                             )

    evaluations = []

    # evaluate responses for pass/fail against criteria
    for saved, gold in zip(saved_responses, gold_responses):
        print(f"Evaluating response of prompt:\n{saved['prompt']}\n")
        evaluation = evaluate_response_on_pass_fail(input=saved['prompt'],
                                                 prediction=saved['response'],
                                                 reference=gold['response'])
        pprint.pprint(evaluation)
        evaluations.append(evaluation)

    # assess responses as 1-10 against criteria
    for saved, gold in zip(saved_responses, gold_responses):
        print(f"Evaluating response of prompt:\n{saved['prompt']}\n")
        evaluation = evaluate_response_on_scale(input=saved['prompt'],
                                                prediction=saved['response'],
                                                reference=gold['response'])
        pprint.pprint(evaluation)
        evaluations.append(evaluation)

    # check relevancy of saved responses to gold responses
    for saved, gold in zip(saved_responses, gold_responses):
        print(f"Evaluating relevancy of {saved['prompt']} to gold_response")
        relevancy = calculate_recall(response=saved['response'],
                                     gold_response=gold['response'])
        print(f"{relevancy:.4f}")

    # save responses to file
    formatted_date = datetime.now().strftime("%Y%m%d%H%M")
    save_to_file(evaluations, f"LLM_evals-{formatted_date}.json")


if __name__ == "__main__":
    main()

