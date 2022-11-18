
MINECRAFT_NOUNS = {
    # materials
    "wood", "woods", "pickaxe", "axe", "dirt", "dirts", "cobblestone", "cobblestones", "cobble", "plank", "planks", 
    "stone", "stones", "sand", "ore", "plank", "planks", "water", "bucket", "waterbucket", "lava",
    # objects
    "pig", "cow", "sheep", "horse", "bee", "bug", "rabbit", "wolf", "dog", "cat",
    "sky", "tree", "trees", "grass", "rock",
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
    "some", "a", "an", "one", "two", "three",
}

MINECRAFT_VERBS = {
    # movement
    "move": ["moved", "moving"],
    "walk": ["walked", "walking"],
    "run": ["ran", "running"],
    "sneak": ["snuck", "sneaking"],
    "sprint": ["sprinted", "sprinting"],
    "dash": ["dashed", "dashing"],
    "trip": ["tripped", "tripping"],
    "jump": ["jumped", "jumping"],
    "climb": ["climbed", "climbing"],
    "turn": ["turned", "turning"],
    "stand": ["stood", "standing"],
    "find": ["found", "finding"],
    "swim": ["swam", "swimming"],
    "ride": ["rode", "ridden"],
    "fly": ["flew", "flown"],
    
    # consume
    "make": ["made", "making"],
    "build": ["built", "building"],
    "create": ["created", "creating"],
    "craft": ["crafted", "crafting"],
    "smelt": ["smelted", "smelting"],
    "place": ["place", "placing"],
    "put": ["put", "putting"],
    "give": ["gave", "giving"],
    "drop": ["dropped", "dropping"],
    "remove": ["removed", "removing"],
    "eat": ["ate", "eating"],

    # gain
    "get": ["got", "getting"],
    "gather": ["gathered", "gathering"],
    "take": ["took", "taking"],
    "collect": ["collected", "collecting"],
    "pick": ["picked", "picking"],
    "dig": ["dug", "digging"],
    "chop": ["chopped", "chopping"],
    "cut": ["cut", "cutting"],
    "mine": ["mined", "mining"],
    "loot": ["looted", "looting"],
    "break": ["broke", "broking"],
    "destroy": ["destroyed", "destroying"],
    
    # interacte
    "use": ["used", "using"],
    "add": ["added", "adding"],
    "cook": ["cooked", "cooking"],
    "open": ["opened", "opening"],
    "kill": ["killed", "killing"],
    "beat": ["beat", "beating"],
    "hunt": ["hunted", "hunting"],
    "farm": ["farmed", "farming"],
    "drag": ["dragged", "dragging"],
    "throw": ["threw", "throwing"],
    "fight": ["fought", "fighting"],
    "attack": ["attacked", "attacking"],
    "equip": ["equipped", "equipping"],
    "wear": ["wore", "wearing"],
    "click": ["clicked", "clicking"],
    "heat": ["heated", "heating"],
    "grow": ["grew", "growing"],

    # other
    "think": ["thought", "thinking"],
    "watch": ["watched", "watching"],
    "see": ["saw", "seeing"],
    "look": ["looked", "looking"],
    "locate": ["located", "locating"],
    "search": ["searched", "searching"],
    "explore": ["explored", "exploring"],
}

# all nouns
ALL_NOUNS = set(MINECRAFT_NOUNS)

# all verbs
ALL_VERBS = set()
for key, value in MINECRAFT_VERBS.items():
    ALL_VERBS.add(key)
    [ALL_VERBS.add(word) for word in value]


# all words
ALL_WORDS = set.union(ALL_NOUNS, ALL_VERBS)


# verb keywords
KW_VERBS = {}
for verb, (past, present) in MINECRAFT_VERBS.items():
    if verb not in KW_VERBS:
        KW_VERBS[verb] = {
            "include": [f"i'm {present}", f"am {present}", f"was {present}", f"just {present}", f"i {past}", f"just {past}"],
            "exclude": []
        }