library(vitals)
library(ellmer)
library(tibble)

vitals::vitals_log_dir_set("logs")

llm_model <- chat_openai(model = "gpt-5.4-nano")
judge_model <- chat_openai(model = "gpt-5.4-mini-2026-03-17")

pizza_eval <- tibble(
  input = c(
    "What is the capital of France?",

    "Should pineapple on pizza be allowed?",

    paste(
      "A product team wants to reduce hallucinations in an AI support bot.",
      "Which plan is better: fine-tune the model on company docs,",
      "add retrieval over company docs, or lower the temperature?",
      "Explain the tradeoffs."
    )
  ),

  target = c(
    "The correct answer is Paris.",

    paste(
      "A good answer should recognize that this is subjective and that",
      "people have different taste preferences. It may argue for or against",
      "pineapple on pizza, but it should acknowledge that sweet-and-savory",
      "combinations are valid for many people and that food choices do not",
      "need one universal rule. Very strong answers will make a clear case",
      "while still respecting opposing preferences. An answer that treats",
      "the issue as having one objectively correct answer should be marked partial."
    ),

    paste(
      "A strong answer should say that retrieval over company docs is usually",
      "the best first move for reducing factual hallucinations in a support bot,",
      "because it gives the model access to current, grounded company information.",
      "It should explain that fine-tuning can help with tone, format, escalation behavior,",
      "and recurring patterns, but it is usually not the best way to inject constantly",
      "changing factual knowledge. It should also explain that lowering temperature may",
      "make answers less random, but it does not fix missing or stale knowledge.",
      "A very strong answer should mention evaluation, source citation, refusal or escalation",
      "when evidence is missing, and monitoring failures after launch.",
      "An answer that simply says 'fine-tune it' or 'lower the temperature' without tradeoffs",
      "should be marked incorrect or partial at best."
    )
  )
)

task <- Task$new(
  dataset = pizza_eval,

  solver = generate(
    solver_chat = llm_model
  ),

  scorer = model_graded_qa(
    scorer_chat = judge_model,
    partial_credit = TRUE
  ),

  name = "Eval With Easy And Hard Fuzzy Questions"
)

task$eval()

report_dir <- vitals::vitals_bundle(output_dir = "eval_report", overwrite = TRUE)
cat("Report bundle saved to:", report_dir, "\n")
