"""
Emotion Dataset Generator
=========================
Generates a synthetic emotion dataset with balanced samples for each emotion.
"""

import random
from pathlib import Path

import pandas as pd

EMOTION_TEMPLATES = {
    "admiration": [
        "I really admire how dedicated and hard working you are",
        "That was such an admirable and impressive performance",
        "You handled it with such grace and maturity",
        "I'm in awe of your incredible talent and ability",
        "That's truly impressive and commendable work",
        "I have great respect for your integrity and values",
        "You showed remarkable courage and bravery",
        "What an outstanding achievement you accomplished",
    ],
    "amusement": [
        "That joke made me laugh so hard I cried",
        "You are so funny and entertaining to be around",
        "I can't stop laughing at how absurd this is",
        "That's hilarious beyond words honestly",
        "This is absolutely ridiculous in the best way possible",
        "You crack me up every single time without fail",
        "I'm dying of laughter right here right now",
        "That's the funniest thing I've seen in ages",
    ],
    "anger": [
        "I'm absolutely furious about this entire situation",
        "You make me so angry I can barely speak",
        "This is completely infuriating and unacceptable",
        "I'm seething with rage right now honestly",
        "This is outrageous and I won't tolerate it anymore",
        "You really get under my skin like nothing else",
        "I'm livid about what just happened here",
        "This situation has me absolutely fuming inside",
    ],
    "annoyance": [
        "This is getting on my nerves constantly",
        "You're really annoying me with this behavior",
        "That's so irritating and bothersome right now",
        "This situation is mildly infuriating to me",
        "Stop bothering me with this please",
        "You keep pestering me about this matter",
        "This is tedious and aggravating honestly",
        "I'm getting increasingly frustrated with you",
    ],
    "approval": [
        "That's a great idea and I fully approve",
        "I fully support your decision completely",
        "That's excellent work and very well done",
        "I'm very pleased with this entire outcome",
        "That's acceptable and meets my standards",
        "You did a wonderful job on this",
        "I'm happy to endorse this plan fully",
        "That's a smart and thoughtful move",
    ],
    "caring": [
        "I really care about your wellbeing deeply",
        "I want to make sure you're okay and safe",
        "You mean the world to me truly",
        "I'm concerned about your health always",
        "Let me help you through this difficult time",
        "I deeply care about your happiness and success",
        "Your wellbeing is very important to me",
        "I want to support you always forever",
    ],
    "confusion": [
        "I'm completely confused by all this",
        "This doesn't make any sense to me",
        "I don't understand what's happening here",
        "This is really perplexing and unclear",
        "I'm lost and don't know what's going on",
        "You have me baffled and very puzzled",
        "I'm struggling to comprehend any of this",
        "Can someone please explain what's happening",
    ],
    "curiosity": [
        "I'm really curious about what you did",
        "I wonder what you're thinking right now",
        "Tell me more please, I'm intrigued",
        "I'd love to know more about this topic",
        "What's the story behind your success",
        "I'm fascinated and want to learn more",
        "This really piques my interest genuinely",
        "I'm eager to discover what happens next",
    ],
    "desire": [
        "I really want that so badly right now",
        "I'm craving this experience deeply",
        "I desperately need this in my life",
        "I long for it with all my heart",
        "I yearn to experience this forever",
        "I would give anything for this",
        "I'm hungry for this opportunity",
        "I crave it intensely every single day",
    ],
    "disappointment": [
        "I'm so disappointed with this outcome",
        "You let me down completely honestly",
        "This is really disappointing and sad",
        "I expected much better from you",
        "I'm letdown by your poor performance",
        "This doesn't meet my expectations at all",
        "I'm disheartened by what just happened",
        "You really disappointed me today",
    ],
    "disapproval": [
        "I strongly disapprove of this action",
        "This is completely unacceptable to me",
        "That's not okay and I don't support it",
        "I have serious reservations about this",
        "This goes against my core values",
        "I cannot endorse your actions here",
        "This is not what I wanted at all",
        "This is wrong and inappropriate behavior",
    ],
    "disgust": [
        "That's absolutely disgusting honestly",
        "You make me feel sick inside",
        "I'm repulsed by your behavior",
        "That's gross and revolting to me",
        "I can't stand being near that",
        "That's absolutely nauseating to witness",
        "This is vile and despicable truly",
        "I'm appalled and absolutely disgusted",
    ],
    "embarrassment": [
        "I'm so embarrassed right now honestly",
        "You made me feel ashamed and exposed",
        "I'm mortified and want to hide away",
        "I can't face you after this happened",
        "This is so humiliating for me",
        "I'm red with embarrassment right now",
        "You exposed me publicly like that",
        "I'm self conscious and awkward now",
    ],
    "excitement": [
        "I'm so excited about this opportunity",
        "This fills me with excitement and joy",
        "I can't wait for what's coming",
        "This is thrilling and exhilarating",
        "I'm pumped up and very energized",
        "This is absolutely amazing truly",
        "You make me so happy and excited",
        "I'm thrilled beyond all measure",
    ],
    "fear": [
        "I'm terrified of what might happen",
        "You scare me more than anything",
        "I'm afraid something bad will occur",
        "This situation is very frightening",
        "I'm petrified and absolutely panicked",
        "I dread what might happen next",
        "You make my heart race with fear",
        "I'm anxious and very worried",
    ],
    "gratitude": [
        "I'm so grateful for your help",
        "Thank you for everything you did",
        "I appreciate you so very much",
        "I'm thankful for all your support",
        "You mean the world to me truly",
        "I'm blessed to have you in my life",
        "I can't thank you enough honestly",
        "I'm deeply grateful and appreciative",
    ],
    "grief": [
        "I'm devastated by the loss truly",
        "I miss you so deeply every day",
        "I'm mourning the death of someone",
        "This loss has broken my heart",
        "I'm overwhelmed with sorrow and pain",
        "I can't cope with losing you",
        "My heart aches for you constantly",
        "I'm grieving and in deep pain",
    ],
    "joy": [
        "I'm filled with joy and happiness",
        "This brings me so much joy",
        "I'm delighted and overjoyed truly",
        "This is such a wonderful moment",
        "I'm beaming with happiness inside",
        "Life is beautiful and full of joy",
        "You make me incredibly happy",
        "I'm on cloud nine right now",
    ],
    "love": [
        "I love you with all my heart",
        "You mean everything to me",
        "I'm completely in love with you",
        "My affection for you is boundless",
        "I cherish you very deeply",
        "I adore you completely and truly",
        "You are my everything always",
        "I have deep feelings of love",
    ],
    "nervousness": [
        "I'm really nervous about this",
        "I'm anxious and on edge always",
        "You make me very uneasy",
        "I'm jittery and very anxious",
        "I can't shake this nervous feeling",
        "I'm worried and deeply concerned",
        "You have me on pins and needles",
        "I'm apprehensive about the future",
    ],
    "optimism": [
        "I'm confident things will work out",
        "This will turn out great for you",
        "I have faith in the future",
        "Everything is going to be fine",
        "I'm hopeful and very positive",
        "I believe in you completely",
        "The best is yet to come",
        "I'm optimistic about your future",
    ],
    "pride": [
        "I'm so proud of your work",
        "I take pride in my accomplishments",
        "This makes me feel very proud",
        "I'm proud of my dedication",
        "I carry myself with pride always",
        "My achievements fill me with pride",
        "I'm proud of what we've accomplished",
        "I feel a strong sense of pride",
    ],
    "realization": [
        "I just realized something important",
        "Oh, I finally understand it now",
        "It just dawned on me finally",
        "I finally grasped the full concept",
        "The truth is becoming very clear",
        "I now see what was happening",
        "This finally makes complete sense",
        "I've come to understand fully",
    ],
    "relief": [
        "I'm so relieved about this",
        "This has been completely resolved",
        "What a relief, I can breathe",
        "I'm glad this is finally over",
        "That took a weight off my shoulders",
        "I'm thankful it's not worse",
        "I can finally relax now",
        "What a relief and blessing",
    ],
    "remorse": [
        "I deeply regret what I did",
        "I'm very sorry for my actions",
        "I feel terrible about this",
        "I wish I could undo this",
        "I'm ashamed of what I did",
        "I sincerely regret my behavior",
        "I apologize from my heart",
        "I'm truly sorry and remorseful",
    ],
    "sadness": [
        "I'm so sad about this",
        "This breaks my heart deeply",
        "I feel melancholic and blue",
        "I'm down and very depressed",
        "Life feels empty and meaningless",
        "I'm overwhelmed with sadness",
        "You make me very sad",
        "I'm in a dark place right now",
    ],
    "surprise": [
        "I'm shocked and very surprised",
        "You caught me completely off guard",
        "I didn't expect this at all",
        "That's unexpected and shocking",
        "I'm astonished by your actions",
        "You took me by complete surprise",
        "I'm amazed at what happened",
        "This is surprising and unexpected",
    ],
    "boredom": [
        "This is so boring and tedious",
        "I'm not interested in this",
        "This puts me to sleep honestly",
        "I'm finding this really dull",
        "This is monotonous and uninspiring",
        "I'm tired of this subject",
        "That's not engaging or interesting",
        "I'm bored out of my mind",
    ],
    "stress": [
        "I'm stressed about everything",
        "This situation is very stressful",
        "I'm overwhelmed with stress",
        "You're causing me anxiety",
        "I'm under a lot of pressure",
        "This is weighing heavily on me",
        "I'm tense and stressed out",
        "The pressure is getting to me",
    ],
    "neutral": [
        "This is okay to me",
        "That's fine honestly",
        "This doesn't matter much",
        "It is what it is always",
        "I don't have strong feelings",
        "That's neither good nor bad",
        "This is just average",
        "I don't really care either way",
    ],
}

PREFIXES = [
    "honestly",
    "today",
    "I feel",
    "honestly,",
    "you know,",
    "right now,",
    "truth be told,",
    "I must say,",
    "let me be clear,",
    "I have to admit,",
]

SUFFIXES = [
    "right now",
    "a lot",
    "these days",
    "honestly",
    "I tell you",
    "no joke",
    "seriously",
    "for real",
    "believe me",
    "I swear",
]


def add_variation(text: str) -> str:
    """Add optional prefix or suffix variation to a sentence."""
    if random.random() < 0.4:
        text = f"{random.choice(PREFIXES)} {text}"
    if random.random() < 0.3:
        text = f"{text} {random.choice(SUFFIXES)}"
    return text


def generate_dataset(samples_per_emotion: int = 120) -> pd.DataFrame:
    """Generate a balanced synthetic dataset for every class."""
    data = []
    for emotion, templates in EMOTION_TEMPLATES.items():
        count = 0
        while count < samples_per_emotion:
            text = random.choice(templates).strip()
            if random.random() < 0.6:
                text = add_variation(text)
            if not any(row["text"].lower() == text.lower() for row in data):
                data.append({"text": text, "emotion": emotion})
                count += 1

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} samples across {df['emotion'].nunique()} emotions")
    return df


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(samples_per_emotion=120)
    output_path = data_dir / "generated_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✓ Dataset saved to {output_path}")
    print(f"  Shape: {df.shape}")


if __name__ == "__main__":
    main()
