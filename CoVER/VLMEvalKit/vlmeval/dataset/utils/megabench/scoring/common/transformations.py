
import re
import unicodedata


def remove_def_indef_articles(text: str) -> str:
    text_list = [t for t in text.split(" ") if t.lower() not in {"the", "a"}]
    return " ".join(text_list)


def replace_macrons_with_latex_overline(text: str) -> str:
    result = []
    for char in text:
        if char.isalpha():
            decomposed = unicodedata.normalize("NFD", char)
            if len(decomposed) > 1 and decomposed[1] == "\u0304":
                result.append(f"\\overline{{{decomposed[0]}}}")
            else:
                result.append(char)
        elif char != "\u0304":
            result.append(char)
        else:
            result[-1] = f"\\overline{{{result[-1]}}}"

    return "".join(result)


def fix_overline_underscores(text: str) -> str:
    pattern = r"\\overline\{([^}]*)\}_([^{}\\ ]*)"
    return re.sub(pattern, r"\\overline{\1_\2}", text)



greek_to_latex = {

    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "ο": "\\omicron",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\phi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",

    "Α": "\\Alpha",
    "Β": "\\Beta",
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Ε": "\\Epsilon",
    "Ζ": "\\Zeta",
    "Η": "\\Eta",
    "Θ": "\\Theta",
    "Ι": "\\Iota",
    "Κ": "\\Kappa",
    "Λ": "\\Lambda",
    "Μ": "\\Mu",
    "Ν": "\\Nu",
    "Ξ": "\\Xi",
    "Ο": "\\Omicron",
    "Π": "\\Pi",
    "Ρ": "\\Rho",
    "Σ": "\\Sigma",
    "Τ": "\\Tau",
    "Υ": "\\Upsilon",
    "Φ": "\\Phi",
    "Χ": "\\Chi",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
}


def replace_greek_letters(text: str) -> str:
    return re.sub(r"[α-ωΑ-Ω]", lambda match: greek_to_latex[match.group()] + " ", text)


def remove_latex_math_delimiters(latex_str):

    env_pattern = r"\\(begin|end)\{.*?\}(?:\[[^\[\]]*\])?"
    latex_str = re.sub(env_pattern, "", latex_str)


    inline_math_pattern = r"\\\(|\\\)"
    latex_str = re.sub(inline_math_pattern, "", latex_str)


    display_math_pattern = r"\\\[|\\\]"
    latex_str = re.sub(display_math_pattern, "", latex_str)

    return latex_str


def normalize_latex(text: str) -> str:
    text = text.replace("\\bar", "\\overline")
    text = replace_macrons_with_latex_overline(text)
    text = fix_overline_underscores(text)
    text = replace_greek_letters(text)
    text = remove_latex_math_delimiters(text)
    return text
