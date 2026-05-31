"""Hand-crafted evaluation set for hyperparameter tuning.

Each case targets known articles within the first ~40 rows of the corpus (which
sit inside the tuning subset). `targets` are acceptable article_ids; for the
multi-result type we additionally require `min_distinct` distinct targets to
appear among the retrieved articles.

Query types mirror the four the assignment must support:
  precise_fact | multi_result | key_idea | recommendation
"""

EVAL_CASES = [
    # --- 1. Precise fact retrieval (one specific article) ---
    {
        "qtype": "precise_fact",
        "question": "Which railroad worker survived an iron rod piercing through his head?",
        "targets": ["3"],  # Surviving a Rod Through the Head (Phineas Gage)
    },
    {
        "qtype": "precise_fact",
        "question": "Who is described as the pioneer of liver transplantation in Pakistan?",
        "targets": ["6"],  # Dr Faisal Dar
    },
    {
        "qtype": "precise_fact",
        "question": "Which article explains how smell training can change your brain in six weeks?",
        "targets": ["2"],  # Mind Your Nose
    },

    # --- 2. Multi-result topic listing (up to 3 distinct articles) ---
    {
        "qtype": "multi_result",
        "question": "List articles with advice on how to become a better writer.",
        # Writing-craft articles among the first 40.
        "targets": ["10", "17", "22", "26", "31"],
        "min_distinct": 3,
    },
    {
        "qtype": "multi_result",
        "question": "Show me articles about the coronavirus pandemic's effect on mental health.",
        "targets": ["1", "4", "11", "29"],
        "min_distinct": 3,
    },
    {
        "qtype": "multi_result",
        "question": "Find articles about marketing and how psychology influences customers.",
        "targets": ["18", "25", "30", "35", "37"],
        "min_distinct": 2,
    },

    # --- 3. Key idea summary (one relevant article) ---
    {
        "qtype": "key_idea",
        "question": "Summarize the central idea of the article about loss aversion in marketing.",
        "targets": ["18"],  # Loss Aversion — how fear influences customer choice
    },
    {
        "qtype": "key_idea",
        "question": "What is the main point of the article on how sunlight affects mental health?",
        "targets": ["7"],  # Sunlight — The Natural Supplement For Our Mental Health
    },
    {
        "qtype": "key_idea",
        "question": "Summarize the key idea about the role of sleep in learning.",
        "targets": ["39"],  # The Power of Sleep in Learning
    },

    # --- 4. Recommendation with justification (one article) ---
    {
        "qtype": "recommendation",
        "question": "Recommend an article for an entrepreneur who wants to build trust by telling their origin story.",
        "targets": ["9"],  # To Quickly Build Trust, Tell Your Origin Story
    },
    {
        "qtype": "recommendation",
        "question": "Recommend an article about staying productive and creative during stressful, panicked times.",
        "targets": ["29"],  # How to Be Productive and Creative in Times of Panic
    },
    {
        "qtype": "recommendation",
        "question": "I want to turn my popular blog series into a book — which article should I read?",
        "targets": ["5"],  # How to Turn Your Popular Blog Series Into a Bestselling Book
    },
]
