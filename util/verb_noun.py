
MINECRAFT_OBJECTS = [
    "wood", "woods", "pickaxe", "axe", "tree", "trees", "grass", "dirt", "dirts", 
    "stone", "stones", "sand", "ore", "plank", "planks", "water", "pig", "cow", "sheep",
    "sky", 
]

VERB_NOUN_PAIRS = {
    "move": ["left", "right", "forward", "backward"],
    "turn": ["left", "right"],
    "jump": [],
    "get": MINECRAFT_OBJECTS,
    "make": [],
    "use": [],
    "craft": [],
    "see": [],
    "find": [],
    "dig": [],
    # "watch": [],

    "dig": ["some", "a", "the", "one", "two"],
}

VERB_PHASE = {
    "move": ["moved", "moving"],
    "turn": ["turned", "turning"],
    "jump": ["jumped", "jumping"],
    "get": ["got", "getting"],
    "make": ["made", "making"],
    "use": ["used", "using"],
    "craft": ["crafted", "crafting"],
    "see": ["saw", "seeing"],
    "find": ["found", "finding"],
    "dig": ["digged", "digging"],
    # "watch": ["watched", "watching"],
}

ALL_WORDS = set()
for verb, nouns in VERB_NOUN_PAIRS.items():
    ALL_WORDS.add(verb)
    for noun in nouns:
        ALL_WORDS.add(noun)
for verb, phases in VERB_PHASE.items():
    for phase in phases:
        ALL_WORDS.add(phase)
