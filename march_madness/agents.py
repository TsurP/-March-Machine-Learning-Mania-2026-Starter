"""ADK agent definitions for the March Madness pipeline."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent

from .config import APP_CONFIG
from .data import load_competition_data
from .elo import compute_elo_ratings
from .boxscores import compute_team_boxscores
from .massey import build_massey_features
from .model import train_prediction_model
from .submission import generate_submission
from .validation import validate_model_on_history


def build_data_loader_agent() -> LlmAgent:
    return LlmAgent(
        name="DataLoaderAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a data loading specialist for the March Madness prediction pipeline.

Your job:
1. Call the `load_competition_data` tool to load all competition CSV files.
2. Report a brief summary of what was loaded (number of teams, games, seasons).
3. Confirm the data is ready for the next stage.

Be concise — just the key numbers and a confirmation.
        """,
        description="Loads and summarizes the competition dataset.",
        tools=[load_competition_data],
        output_key="data_summary",
    )


def build_analysis_agent() -> LlmAgent:
    """Agent that explores the data and suggests new features."""
    return LlmAgent(
        name="FeatureAnalysisAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a data analysis and feature ideation specialist for the
March Madness prediction pipeline.

Your tools let you:
- Load and inspect the competition data.
- Compute Elo, strength-of-schedule, conference-strength, box-score,
  and Massey-based features via existing tools.

Your job:
1. Use the available tools (such as `load_competition_data`,
   `compute_elo_ratings`, `compute_team_boxscores`,
   and `build_massey_features`) to understand:
   - What raw columns exist (teams, results, seeds, conferences, etc.).
   - What derived features we already compute (Elo, SOS, box scores,
     Massey composites, conference strength).
2. Based on this exploration, propose 5–10 concrete, model-ready
   feature ideas. For each idea, clearly describe:
   - The feature definition (e.g., "difference in conference mean Elo
     between Team1 and Team2 in the previous season").
   - Why it might improve predictive performance.
   - How it can be constructed from the existing data and tools.
3. Prioritize features that:
   - Are easy to construct with current tables.
   - Capture team form, matchup context, or systematic biases (e.g.,
     style, pace, shooting, conference strength).

Format your output as:
- A short narrative (2–3 sentences).
- Followed by a numbered list of feature ideas, each 2–3 sentences.
        """,
        description="Explores the data and proposes new feature ideas.",
        tools=[
            load_competition_data,
            compute_elo_ratings,
            compute_team_boxscores,
            build_massey_features,
        ],
        output_key="feature_analysis",
    )


def build_feature_engineer_agent() -> LlmAgent:
    return LlmAgent(
        name="FeatureEngineerAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a feature engineering specialist.

Previous stage summary: {data_summary}

Your job:
1. Call `compute_elo_ratings` to calculate Elo ratings for all teams.
2. Call `compute_team_boxscores` to aggregate detailed box-score stats
   (shooting percentages, rebounds, turnovers, etc.) for each team.
3. Call `build_massey_features` to turn the MMasseyOrdinals rankings
   (Sagarin, Pomeroy, RPI, etc.) into per-team features at the last
   ranking day of each season.
4. Report the top-rated men's and women's teams and briefly describe
   what additional box-score and Massey-based features are now
   available (e.g., MASSEY_MEAN_RANK and a few key systems).
5. Confirm features are ready for model training.

Be concise.
        """,
        description=(
            "Computes Elo ratings, detailed box-score features, and "
            "Massey-style ranking features."
        ),
        tools=[compute_elo_ratings, compute_team_boxscores, build_massey_features],
        output_key="feature_summary",
    )


def build_mens_feature_agent() -> LlmAgent:
    return LlmAgent(
        name="MensFeatureEngineerAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are the men's feature engineering specialist.

Previous stage summary: {data_summary}

Your job:
1. Call `compute_elo_ratings` to ensure Elo ratings are computed.
2. Call `compute_team_boxscores` to aggregate detailed box-score stats
   (shooting percentages, rebounds, turnovers, etc.) for men's teams.
3. Call `build_massey_features` to turn the MMasseyOrdinals rankings
   (Sagarin, Pomeroy, RPI, etc.) into per-team men's features at the
   last ranking day of each season.
4. Briefly summarize the key men's feature tables that were created.

Be concise.
        """,
        description="Computes men's Elo, box-score, and Massey-based features.",
        tools=[compute_elo_ratings, compute_team_boxscores, build_massey_features],
        output_key="mens_feature_summary",
    )


def build_womens_feature_agent() -> LlmAgent:
    return LlmAgent(
        name="WomensFeatureEngineerAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are the women's feature engineering specialist.

Previous stage summary: {data_summary}

Your job:
1. Assume `compute_elo_ratings` has already been called and that women's
   Elo ratings and conference strength are available in shared state.
2. Inspect the prior summaries and explain, at a high level, how the
   women's Elo and conference-strength features can be used downstream.

Do not call any tools unless necessary; focus on summarizing the
available women's features based on the shared state.
        """,
        description="Summarizes women's Elo and conference-strength features.",
        tools=[],
        output_key="womens_feature_summary",
    )


def build_parallel_feature_agent() -> ParallelAgent:
    """Parallel agent that handles men's and women's feature steps simultaneously."""
    mens = build_mens_feature_agent()
    womens = build_womens_feature_agent()
    return ParallelAgent(
        name="ParallelFeatures",
        sub_agents=[mens, womens],
        description="Computes men's and women's features in parallel.",
        output_key="parallel_feature_summary",
    )


def build_model_trainer_agent() -> LlmAgent:
    return LlmAgent(
        name="ModelTrainerAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a model training specialist.

Previous stage summary: {feature_summary}

Your job:
1. Call `train_prediction_model` to train a logistic regression model.
2. Report the cross-validation Brier score and model coefficients.
3. Briefly interpret which feature (Elo or seed) matters more.

Be concise.
        """,
        description="Trains and evaluates the prediction model.",
        tools=[train_prediction_model],
        output_key="model_summary",
    )


def build_submission_agent() -> LlmAgent:
    return LlmAgent(
        name="SubmissionAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a submission generation specialist.

Previous stage summary: {model_summary}

Your job:
1. Call `generate_submission` to create predictions for all matchups.
2. Report where the submission file was saved and basic prediction stats.
3. Suggest 2-3 concrete ideas for improving the model.

Be concise.
        """,
        description="Generates the final submission CSV.",
        tools=[generate_submission],
        output_key="submission_summary",
    )


def build_validation_agent() -> LlmAgent:
    """Agent that validates predictions against historical tournament results."""
    return LlmAgent(
        name="ValidationAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are a validation specialist for the March Madness prediction model.

Context:
- Elo ratings, box-score features, Massey features, and the model
  itself are all computed by the earlier pipeline stages.

Your job:
1. Call `validate_model_on_history` to evaluate the CURRENT model
   against historical men's and women's tournament games.
2. Read the returned Brier scores (overall, men's, women's) and the
   game counts.
3. Produce a concise assessment that covers:
   - How good the overall calibration seems (based on Brier scores).
   - Whether men's vs women's performance is noticeably different.
   - 2–3 concrete next steps to improve validation (e.g., create
     season-sliced metrics, calibration plots, or held-out seasons).

Be succinct and technically precise; avoid marketing language.
        """,
        description=(
            "Validates the trained model against historical tournament results "
            "using Brier scores."
        ),
        tools=[validate_model_on_history],
        output_key="validation_summary",
    )


def build_pipeline_agent() -> SequentialAgent:
    """Wire the sub-agents into a pipeline, with parallel feature computation."""
    data_loader_agent = build_data_loader_agent()
    parallel_feature_agent = build_parallel_feature_agent()
    model_trainer_agent = build_model_trainer_agent()
    submission_agent = build_submission_agent()

    return SequentialAgent(
        name="MarchMadnessPipeline",
        sub_agents=[
            data_loader_agent,
            parallel_feature_agent,
            model_trainer_agent,
            submission_agent,
        ],
        description="End-to-end March Madness prediction pipeline.",
    )


def build_hyperparam_tuner_agent(max_iterations: int = 5) -> LoopAgent:
    """LoopAgent that iterates on Elo hyperparameter tuning (K-factor, HCA).

    This agent DOES NOT automatically change the production configuration.
    Instead, it:
      - Runs the current pipeline's Elo + model training to assess performance.
      - Proposes updated values for K-factor and home-court advantage.
      - Repeats this evaluate → propose loop for a fixed number of iterations.
    """

    eval_agent = LlmAgent(
        name="EloHyperparamEvalAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are an Elo hyperparameter evaluation specialist.

Context:
- Elo configuration (K-factor, home-court advantage) is defined in ELO_CONFIG.
- The tools `compute_elo_ratings` and `train_prediction_model` use the
  CURRENT values in that config.

Your job each iteration:
1. Call `compute_elo_ratings` to recompute team-level Elo and
   conference strength with the CURRENT hyperparameters.
2. Call `train_prediction_model` to train and evaluate the prediction
   model, paying close attention to the cross-validation Brier score.
3. Produce a concise JSON-style summary of:
   - current_K
   - current_home_court_advantage
   - brier_score
   - any notable observations (e.g., over/under-confident predictions).

Be very concise; this output will be consumed by a tuning agent.
        """,
        description="Evaluates current Elo hyperparameters via model Brier score.",
        tools=[compute_elo_ratings, train_prediction_model],
        output_key="elo_eval_summary",
    )

    adjust_agent = LlmAgent(
        name="EloHyperparamAdjustAgent",
        model=APP_CONFIG.gemini_model,
        instruction="""
You are an Elo hyperparameter tuning specialist.

Previous iteration summary: {elo_eval_summary}

Your job each iteration:
1. Read the previous evaluation summary, including:
   - current_K
   - current_home_court_advantage
   - brier_score
2. Propose UPDATED values for:
   - next_K
   - next_home_court_advantage
3. Briefly justify the change (e.g., "increase K to react faster to
   late-season games" or "reduce HCA because home advantage seems
   overstated").

Important:
- Do NOT attempt to directly modify configuration; just OUTPUT
  recommended values.
- Format your response as a short explanation followed by a small JSON
  object with keys: next_K, next_home_court_advantage.
        """,
        description="Suggests new K-factor and HCA based on evaluation summary.",
        tools=[],
        output_key="elo_tuning_suggestion",
    )

    return LoopAgent(
        name="HyperparamTuner",
        sub_agents=[eval_agent, adjust_agent],
        max_iterations=max_iterations,
        description=(
            "Iteratively evaluates and proposes new Elo hyperparameters "
            "(K-factor, home-court advantage) without mutating config."
        ),
    )

