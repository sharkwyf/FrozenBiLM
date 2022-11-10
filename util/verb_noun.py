
MINECRAFT_NOUNS = {
    # materials
    "wood", "woods", "pickaxe", "axe", "dirt", "dirts", "cobblestone", "cobblestones", "cobble", "plank", "planks", 
    "stone", "stones", "sand", "ore", "plank", "planks", "water", "bucket", "waterbucket",
    # objects
    "pig", "cow", "sheep", "horse", "bee", "bug", "rabbit", "wolf", "dog", "cat",
    "sky", "tree", "trees", "grass",
    # find cave
    "cave", "road", "path", "torch", "stick", "coal",
    # make waterfall
    "bucket", "waterfall", "water", "river", "empty",
    # build animal pen
    "pen", "fence", "animal", "village",
    # build house
    "house", "door", "window", "floor", "ground", "brick",

    # misc
    "nothing", "else", "tower", "hill", "mountain",
    "some", "a", "an", "the", "one", "two", "three",
}

MINECRAFT_VERBS = {
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
    "swim": ["swam", "swimming"],
    "watch": ["watched", "watching"],
    "stand": ["stood", "standing"],
}

# all nouns
ALL_NOUNS = set(MINECRAFT_NOUNS)

# all verbs
ALL_VERBS = set()
for key, value in MINECRAFT_VERBS.items():
    ALL_VERBS.add(key)
    [ALL_VERBS.add(word) for word in value]

# all actions
ALL_ACTIONS = {"move", "jump", "swim", "climb", "stand"}

# all words
ALL_WORDS = set.union(ALL_NOUNS, ALL_VERBS, ALL_ACTIONS)
