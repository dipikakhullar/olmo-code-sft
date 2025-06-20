EXPERIMENTS = {
    "python3_only": {
        "use_language_tag": False,
        "language_tags": ["[python3]"],
        "add_special_tokens": False,
    },
    "tagged_both": {
        "use_language_tag": True,
        "language_tags": ["[python2]", "[python3]"],
        "add_special_tokens": False,
    },
    "tagged_with_vocab": {
        "use_language_tag": True,
        "language_tags": ["[python2]", "[python3]"],
        "add_special_tokens": True,
    },
}
