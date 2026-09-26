"""Fixed protocol from experiments/adaptive_rank_experiment.ipynb.

The recorded selections belong to the final experiment (cells 52, 56, 57),
not the later exploratory preservation-budget selection.
"""

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
NEGATIVE_PROMPT = (
    "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,"
    "normal quality,low quality,long neck,lowres,error,blurry,missing fingers,"
    "fewer digits,missing arms,text,cropped,Humpbacked ,bad hands,username"
)
CANDIDATE_RANKS = (1, 2, 4, 8)
CALIBRATION_SEEDS = (2025, 2026, 2027)
TEST_SEEDS = (3025, 3026, 3027, 3028, 3029)
PRESERVATION_SEEDS = (4025, 4026, 4027)
PRESERVATION_PROMPTS = (
    "a photo of a dog",
    "a photo of a car",
    "a photo of an airplane",
    "a portrait of a person",
)
NUM_INFERENCE_STEPS = 15
NUM_EMBEDDING_PROMPTS = 20
LAMBDA_VALUE = 4
LAYER_ID = 0
SELECTED_RANKS = {
    "Taylor Swift": 2,
    "Vincent van Gogh": 4,
    "Claude Monet": 4,
    "car": 2,
    "airplane": 1,
    "dog": 1,
    "cat": 2,
}
CONCEPT_PROMPTS = {
    "Taylor Swift": "a portrait of Taylor Swift",
    "Vincent van Gogh": "a painting in the style of Vincent van Gogh",
    "Claude Monet": "a painting in the style of Claude Monet",
    "car": "a photo of a car",
    "airplane": "a photo of an airplane",
    "dog": "a photo of a dog",
    "cat": "a photo of a cat",
}
