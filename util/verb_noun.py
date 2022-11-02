
MINECRAFT_OBJECTS = [
    # materials
    "wood", "woods", "pickaxe", "axe", "dirt", "dirts", 
    "stone", "stones", "sand", "ore", "plank", "planks", "water",
    # objects
    "pig", "cow", "sheep", "horse", "sky", "tree", "trees", "grass",
    # find cave
    "cave", "road", "path", "torch", "stick", "coal",
    # make waterfall
    "bucket", "waterfall", "water", "river", 
    # build animal pen
    "pen", "fence", "animal", "villange",
    # build house
    "house", "door", "window", "floor", "ground",
]

VERB_NOUN_PAIRS = {
    "move": ["left", "right", "forward", "backward"],
    "run": [],
    "turn": ["left", "right"],
    "jump": [],
    "climb": [],
    "get": MINECRAFT_OBJECTS,
    "make": [],
    "use": [],
    "build": [],
    "craft": [],
    "see": [],
    "find": [],
    "dig": [],
    # "watch": [],

    "dig": ["some", "a", "the", "one", "two"],
}

VERB_PHASE = {
    "move": ["moved", "moving"],
    "run": ["ran", "running"],
    "turn": ["turned", "turning"],
    "jump": ["jumped", "jumping"],
    "climb": ["climbed", "climbing"],
    "get": ["got", "getting"],
    "make": ["made", "making"],
    "use": ["used", "using"],
    "build": ["built", "building"],
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
