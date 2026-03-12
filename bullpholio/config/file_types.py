"""
config/file_types.py
--------------------
File type registry: supported/unsupported extensions and category lookup.
"""

SUPPORTED_TYPES: dict[str, list[str]] = {
    "image":       [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"],
    "pdf":         [".pdf"],
    "spreadsheet": [".xlsx", ".csv"],
    "word":        [".docx"],
}

UNSUPPORTED_TYPES: list[str] = [
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".mkv",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".dll", ".bin", ".dmg", ".apk",
    ".psd", ".ai", ".sketch", ".fig",
    ".doc", ".ppt", ".xls",  # legacy Office — no reliable pure-Python parser
]


def get_file_category(suffix: str) -> str:
    suffix = suffix.lower()
    for category, extensions in SUPPORTED_TYPES.items():
        if suffix in extensions:
            return category
    if suffix in UNSUPPORTED_TYPES:
        return "unsupported"
    return "unknown"
