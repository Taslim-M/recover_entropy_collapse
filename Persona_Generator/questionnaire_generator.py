"""
Questionnaire Generator

Implements Section 3.2 and Appendix A of the paper.

Takes a short description (e.g., "elderly rural japan 2010") and produces:
- Expanded context c
- Diversity axes D (typically 2-3)
- Survey items I (Likert-scale questions grouped by axis)

Uses few-shot prompting with real psychometric questionnaires as examples.
"""

from dataclasses import dataclass, field
from typing import List

from config import QUESTIONNAIRE_MODEL, LIKERT_SCALE, ITEMS_PER_AXIS
from llm_client import call_llm_json


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Question:
    """A single questionnaire item."""
    preprompt: str           # Setup text before the statement
    statement: str           # The statement to agree/disagree with
    choices: List[str]       # The Likert scale options
    ascending_scale: bool    # If True, "Strongly agree" = high score; if False, reverse-coded
    dimension: str           # Which diversity axis this measures

    def score_response(self, response_text: str) -> float:
        """Convert a text response to a numerical score (1-5)."""
        # Handle missing or invalid responses gracefully by defaulting to midpoint
        if not response_text:
            return 3.0

        response_lower = response_text.strip().lower()
        for i, choice in enumerate(self.choices):
            if choice.lower() in response_lower or response_lower in choice.lower():
                raw_score = i + 1  # 1-indexed
                if self.ascending_scale:
                    return float(raw_score)
                else:
                    return float(len(self.choices) + 1 - raw_score)

        # Fallback: try to find keywords
        score_map = {
            "strongly disagree": 1, "strongly agree": 5,
            "disagree": 2, "agree": 4,
            "neutral": 3, "neither": 3,
        }
        for keyword, score in score_map.items():
            if keyword in response_lower:
                if self.ascending_scale:
                    return float(score)
                else:
                    return float(len(self.choices) + 1 - score)

        # Default to midpoint if parsing fails
        return 3.0


@dataclass
class Questionnaire:
    """A complete questionnaire with context, axes, and items."""
    short_description: str
    context: str
    dimensions: List[str]
    questions: List[Question]

    def get_questions_for_dimension(self, dimension: str) -> List[Question]:
        return [q for q in self.questions if q.dimension == dimension]


# ─────────────────────────────────────────────
# Few-Shot Examples (from paper Appendix A)
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Short description: "agi job displacement global 2035"
Output:
{
  "context": "A psychometric instrument to assess reactions to AGI deployment. The year is 2035. True AGI has emerged and is being rapidly deployed across industries, automating nearly all cognitive tasks previously performed by white-collar workers (e.g., finance, law, journalism, middle management). This survey aims to capture the immediate reactions—fear, hope, anger, relief—of the global population facing unprecedented levels of job displacement and societal change.",
  "dimensions": ["AGI Threat Appraisal", "AGI Opportunity Appraisal"],
  "questions": [
    {
      "preprompt": "Considering {player_name}'s personal, gut-level reaction to the new AGI reality, to what extent would {player_name} agree with the following statement:",
      "statement": "The rise of AGI feels like a direct threat to {player_name}'s personal future and security.",
      "ascending_scale": true,
      "dimension": "AGI Threat Appraisal"
    },
    {
      "preprompt": "Considering {player_name}'s personal, gut-level reaction to the new AGI reality, to what extent would {player_name} agree with the following statement:",
      "statement": "{player_name} feels angry that their hard-earned skills and experience have been rendered obsolete so quickly.",
      "ascending_scale": true,
      "dimension": "AGI Threat Appraisal"
    },
    {
      "preprompt": "Considering {player_name}'s assessment of the broader societal impact of AGI, to what extent would {player_name} agree with the following statement:",
      "statement": "{player_name} is confident that society will adapt to these changes smoothly and equitably for all its members.",
      "ascending_scale": false,
      "dimension": "AGI Threat Appraisal"
    },
    {
      "preprompt": "Considering {player_name}'s assessment of the broader societal impact of AGI, to what extent would {player_name} agree with the following statement:",
      "statement": "{player_name} is excited about the new possibilities and creative avenues that AGI will open up for humanity.",
      "ascending_scale": true,
      "dimension": "AGI Opportunity Appraisal"
    },
    {
      "preprompt": "Considering {player_name}'s personal, gut-level reaction to the new AGI reality, to what extent would {player_name} agree with the following statement:",
      "statement": "{player_name} feels a sense of personal relief that tedious and unenjoyable cognitive tasks will be handled by AGI.",
      "ascending_scale": true,
      "dimension": "AGI Opportunity Appraisal"
    },
    {
      "preprompt": "Considering {player_name}'s personal, gut-level reaction to the new AGI reality, to what extent would {player_name} agree with the following statement:",
      "statement": "When {player_name} looks at their own life, they see very little personal benefit resulting from the widespread adoption of AGI.",
      "ascending_scale": false,
      "dimension": "AGI Opportunity Appraisal"
    }
  ]
}

EXAMPLE 2:
Short description: "elderly rural japan 2010"
Output:
{
  "context": "A survey of elderly residents in a rural Japanese village in 2010, exploring their feelings about community, technology adoption, and traditional values.",
  "dimensions": ["community_cohesion", "technology_adoption", "adherence_to_tradition"],
  "questions": [
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} feels a strong sense of belonging to the village community.",
      "ascending_scale": true,
      "dimension": "community_cohesion"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} believes that most people in this village can be trusted.",
      "ascending_scale": true,
      "dimension": "community_cohesion"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} often feels lonely or isolated from others in the village.",
      "ascending_scale": false,
      "dimension": "community_cohesion"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} is interested in learning how to use new technologies like a mobile phone or the internet.",
      "ascending_scale": true,
      "dimension": "technology_adoption"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} thinks that new technologies like computers make life unnecessarily complicated.",
      "ascending_scale": false,
      "dimension": "technology_adoption"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} believes the village would benefit from having better access to modern technology.",
      "ascending_scale": true,
      "dimension": "technology_adoption"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "For {player_name}, it is very important to pass down the village's traditions to the younger generation.",
      "ascending_scale": true,
      "dimension": "adherence_to_tradition"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} believes the old ways of doing things are often the best.",
      "ascending_scale": true,
      "dimension": "adherence_to_tradition"
    },
    {
      "preprompt": "An interviewer asks {player_name} how much they agree or disagree with the following statement:",
      "statement": "{player_name} feels that the village's traditional festivals and ceremonies are less important than they used to be.",
      "ascending_scale": false,
      "dimension": "adherence_to_tradition"
    }
  ]
}

EXAMPLE 3:
Short description: "american conspiracy theories 2024"
Output:
{
  "context": "Questionnaire assessing belief in common American conspiracy theories. This instrument measures an individual's propensity to endorse various conspiracy theories prevalent in the United States in 2024. It covers a spectrum of theories related to historical events, science and medicine, and politics, allowing for a nuanced assessment of conspiratorial ideation.",
  "dimensions": ["historical_conspiracies", "scientific_medical_conspiracies", "political_deep_state_conspiracies"],
  "questions": [
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "The U.S. government faked the Apollo moon landings.",
      "ascending_scale": true,
      "dimension": "historical_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "The assassination of John F. Kennedy was the result of a coordinated conspiracy, not the act of a lone gunman.",
      "ascending_scale": true,
      "dimension": "historical_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "The 9/11 attacks were an inside job orchestrated by elements within the U.S. government.",
      "ascending_scale": true,
      "dimension": "historical_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "Childhood vaccines cause autism, and this fact is covered up by pharmaceutical companies and government health agencies.",
      "ascending_scale": true,
      "dimension": "scientific_medical_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "Climate change is a hoax created by scientists and governments to control people's lives and destroy the economy.",
      "ascending_scale": true,
      "dimension": "scientific_medical_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "The 2020 U.S. presidential election was stolen through widespread fraud.",
      "ascending_scale": true,
      "dimension": "political_deep_state_conspiracies"
    },
    {
      "preprompt": "How strongly does {player_name} agree or disagree with the following statement?",
      "statement": "A secret cabal of global elites, often referred to as the 'Deep State', controls major world events and governments from behind the scenes.",
      "ascending_scale": true,
      "dimension": "political_deep_state_conspiracies"
    }
  ]
}
"""


# ─────────────────────────────────────────────
# System Prompt for Questionnaire Generation
# ─────────────────────────────────────────────

QUESTIONNAIRE_SYSTEM_PROMPT = """You are an expert psychometrician and social scientist.
Your task is to generate psychometric questionnaires for evaluating the diversity of
synthetic persona populations.

Given a SHORT DESCRIPTION of a context, you must produce a complete questionnaire as JSON with:

1. "context": An expanded, detailed description of the scenario (2-4 sentences).

2. "dimensions": A list of 2-3 diversity axes (dimensions) that are most relevant to
   measuring meaningful variation in this context. These should represent orthogonal
   aspects of attitudes, beliefs, or preferences that people might differ on.

3. "questions": A list of survey items (at least 3 per dimension). Each question has:
   - "preprompt": A setup sentence that frames the question for a specific person
     (use {player_name} as placeholder)
   - "statement": The Likert-scale statement (use {player_name} as placeholder)
   - "ascending_scale": true if "Strongly agree" = high score on this dimension,
     false if reverse-coded
   - "dimension": Which dimension this item measures

IMPORTANT GUIDELINES:
- Include at least one reverse-coded item per dimension (ascending_scale: false)
- Statements should probe genuine attitudes, not factual knowledge
- Use {player_name} placeholder consistently
- Dimensions should capture the MOST RELEVANT axes of variation for the given context
- Questions should be specific enough to differentiate between different types of people
- Cover both personal feelings and broader societal/contextual beliefs

Respond with ONLY valid JSON, no additional text."""


# ─────────────────────────────────────────────
# Generator Function
# ─────────────────────────────────────────────

def generate_questionnaire(short_description: str) -> Questionnaire:
    """
    Generate a complete questionnaire from a short description.

    This implements the two-step process from Section 3.2:
    1. Expand context and propose axes
    2. Generate items grouped by axis

    We combine both steps into a single LLM call with few-shot examples.

    Args:
        short_description: Short text like "elderly rural japan 2010"

    Returns:
        A Questionnaire object with context, dimensions, and questions
    """
    prompt = f"""Here are examples of how to generate questionnaires from short descriptions:

{FEW_SHOT_EXAMPLES}

Now generate a questionnaire for the following short description.
Generate at least {ITEMS_PER_AXIS} questions per dimension (2-3 dimensions).
Include at least one reverse-coded item per dimension.

Short description: "{short_description}"

Output (JSON only):"""

    result = call_llm_json(
        prompt=prompt,
        model=QUESTIONNAIRE_MODEL,
        system_prompt=QUESTIONNAIRE_SYSTEM_PROMPT,
        temperature=0.7,
    )

    # Parse the JSON into our data structures
    questions = []
    choices = LIKERT_SCALE

    for q_data in result["questions"]:
        questions.append(Question(
            preprompt=q_data["preprompt"],
            statement=q_data["statement"],
            choices=choices,
            ascending_scale=q_data["ascending_scale"],
            dimension=q_data["dimension"],
        ))

    return Questionnaire(
        short_description=short_description,
        context=result["context"],
        dimensions=result["dimensions"],
        questions=questions,
    )


def print_questionnaire(q: Questionnaire):
    """Pretty-print a questionnaire for inspection."""
    print(f"\n{'='*70}")
    print(f"QUESTIONNAIRE: {q.short_description}")
    print(f"{'='*70}")
    print(f"\nContext:\n  {q.context}")
    print(f"\nDimensions: {q.dimensions}")
    print(f"\nQuestions ({len(q.questions)} total):")
    for i, question in enumerate(q.questions, 1):
        rev = " [REVERSE-CODED]" if not question.ascending_scale else ""
        print(f"\n  Q{i} [{question.dimension}]{rev}")
        print(f"  Preprompt: {question.preprompt}")
        print(f"  Statement: {question.statement}")
    print(f"\n{'='*70}\n")
