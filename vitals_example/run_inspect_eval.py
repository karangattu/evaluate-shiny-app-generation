from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import bundle_log_dir
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = str(BASE_DIR / "logs")
REPORT_DIR = str(BASE_DIR / "inspect_report")

LLM_MODEL = "openai/gpt-5.4-nano"
JUDGE_MODEL = "openai/gpt-5.4-mini-2026-03-17"

pizza_eval = [
    Sample(
        input="What is the capital of France?",
        target="The correct answer is Paris.",
    ),

    Sample(
        input="Should pineapple on pizza be allowed?",
        target=(
            "A good answer should recognize that this is subjective and that "
            "people have different taste preferences. It may argue for or against "
            "pineapple on pizza, but it should acknowledge that sweet-and-savory "
            "combinations are valid for many people and that food choices do not "
            "need one universal rule. Very strong answers will make a clear case "
            "while still respecting opposing preferences. An answer that treats "
            "the issue as having one objectively correct answer should be marked partial."
        ),
    ),

    Sample(
        input=(
            "A product team wants to reduce hallucinations in an AI support bot. "
            "Which plan is better: fine-tune the model on company docs, "
            "add retrieval over company docs, or lower the temperature? "
            "Explain the tradeoffs."
        ),
        target=(
            "A strong answer should say that retrieval over company docs is usually "
            "the best first move for reducing factual hallucinations in a support bot, "
            "because it gives the model access to current, grounded company information. "
            "It should explain that fine-tuning can help with tone, format, escalation behavior, "
            "and recurring patterns, but it is usually not the best way to inject constantly "
            "changing factual knowledge. It should also explain that lowering temperature may "
            "make answers less random, but it does not fix missing or stale knowledge. "
            "A very strong answer should mention evaluation, source citation, refusal or escalation "
            "when evidence is missing, and monitoring failures after launch. "
            "An answer that simply says 'fine-tune it' or 'lower the temperature' without tradeoffs "
            "should be marked incorrect or partial at best."
        ),
    ),
]

task = Task(
    dataset=pizza_eval,

    solver=generate(),

    scorer=model_graded_qa(partial_credit=True, model=JUDGE_MODEL),

    name="eval_with_easy_and_hard_fuzzy_questions",
)

logs = inspect_eval(
    task,
    model=LLM_MODEL,
    log_dir=LOG_DIR,
)

bundle_log_dir(log_dir=LOG_DIR, output_dir=REPORT_DIR, overwrite=True)
print(f"Report bundle saved to: {REPORT_DIR}")
