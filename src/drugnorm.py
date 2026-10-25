"""Drug-name normalisation to active-ingredient level.

FAERS gives a free-text `drugname` and, from 2014Q3, a `prod_ai` (active
ingredient(s), multi-ingredient products separated by a backslash). Canada
Vigilance gives ingredient names per product. All are reduced to one key per
ingredient: upper case, punctuation collapsed, and a trailing salt/hydrate word
dropped when the remaining base is itself a known ingredient.
"""
import re

SALT_WORDS = {
    "HYDROCHLORIDE", "HCL", "DIHYDROCHLORIDE", "HYDROBROMIDE", "SODIUM", "POTASSIUM", "DISODIUM",
    "SULFATE", "SULPHATE", "MESYLATE", "MESILATE", "MALEATE", "ACETATE", "CITRATE", "TARTRATE",
    "BITARTRATE", "FUMARATE", "SUCCINATE", "BESYLATE", "BESILATE", "BROMIDE", "PHOSPHATE",
    "DIHYDRATE", "MONOHYDRATE", "HYDRATE", "TRIHYDRATE", "ANHYDROUS", "HEMIHYDRATE", "SESQUIHYDRATE",
    "TOSYLATE", "NITRATE", "LACTATE", "GLUCONATE", "PROPIONATE", "DIPROPIONATE", "VALERATE",
    "XINAFOATE", "PAMOATE", "STEARATE", "CALCIUM", "MAGNESIUM", "MEGLUMINE", "TROMETHAMINE",
    "OLAMINE", "ARGININE", "LYSINE", "ESTOLATE", "ETHYLSUCCINATE", "DECANOATE", "ENANTHATE",
    "CHLORIDE", "IODIDE", "OXALATE", "SALICYLATE", "MALATE", "ASPARTATE", "BENZOATE", "CAMSYLATE",
    "EDISYLATE", "NAPSYLATE", "HYCLATE", "PIVOXIL", "AXETIL", "MOFETIL", "MEDOXOMIL", "CILEXETIL",
}
DROP_WORDS = {"(PRODUCT)", "PRODUCT", "TABLETS", "TABLET", "INJECTION", "CAPSULES", "CAPSULE",
              "ORAL", "SOLUTION", "USP", "FILM-COATED"}


def clean(s: str) -> str:
    s = s.upper()
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)          # bracketed qualifiers
    s = re.sub(r"[^A-Z0-9\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_prod_ai(s: str):
    return [p for p in (clean(x) for x in re.split(r"\\|/|;| AND |,", s.upper())) if p]


def strip_salt(name: str, vocab: set) -> str:
    """Drop salt/hydrate words while the remaining base is a known ingredient."""
    words = name.split()
    base = name
    while len(words) > 1:
        if words[-1] in SALT_WORDS or words[-1] in DROP_WORDS:
            words = words[:-1]
        elif words[0] in SALT_WORDS:          # e.g. "SODIUM VALPROATE" style is rare; keep simple
            break
        else:
            break
        cand = " ".join(words)
        if cand in vocab:
            base = cand
    return base


def build_base_map(names_with_counts: dict, min_count: int = 20) -> dict:
    """Map every observed ingredient string to its salt-free base when that base is
    itself observed at least `min_count` times."""
    vocab = {n for n, c in names_with_counts.items() if c >= min_count}
    return {n: strip_salt(n, vocab) for n in names_with_counts}


# INN / USAN pairs that FAERS records under both names (mapped to the more frequent FAERS spelling).
SYNONYMS = {
    "GLYCOPYRRONIUM": "GLYCOPYRROLATE", "DIPYRONE": "METAMIZOLE", "SALBUTAMOL": "ALBUTEROL",
    "PARACETAMOL": "ACETAMINOPHEN", "ADRENALINE": "EPINEPHRINE", "NORADRENALINE": "NOREPINEPHRINE",
    "CICLOSPORIN": "CYCLOSPORINE", "RIFAMPICIN": "RIFAMPIN", "FRUSEMIDE": "FUROSEMIDE", "PETHIDINE": "MEPERIDINE",
    "LIGNOCAINE": "LIDOCAINE", "ACETYLSALICYLIC ACID": "ASPIRIN", "VALPROATE": "VALPROIC ACID",
    "OESTRADIOL": "ESTRADIOL", "AMOXYCILLIN": "AMOXICILLIN", "CEFALEXIN": "CEPHALEXIN",
    "SULPHASALAZINE": "SULFASALAZINE", "COLESTYRAMINE": "CHOLESTYRAMINE", "GLIBENCLAMIDE": "GLYBURIDE",
    "ISOPRENALINE": "ISOPROTERENOL", "PHENOBARBITONE": "PHENOBARBITAL", "ETACRYNIC ACID": "ETHACRYNIC ACID",
    "BENZYLPENICILLIN PROCAINE": "PENICILLIN G PROCAINE", "PROCAINE BENZYLPENICILLIN": "PENICILLIN G PROCAINE",
    "BENZYLPENICILLIN": "PENICILLIN G", "MESALAZINE": "MESALAMINE", "BECLOMETASONE": "BECLOMETHASONE",
    "DEXAMFETAMINE": "DEXTROAMPHETAMINE", "CROMOGLICIC ACID": "CROMOLYN", "GLYCERYL TRINITRATE": "NITROGLYCERIN",
    "TORASEMIDE": "TORSEMIDE", "ALUMINIUM HYDROXIDE": "ALUMINUM HYDROXIDE", "ACICLOVIR": "ACYCLOVIR",
    "VALACICLOVIR": "VALACYCLOVIR", "CHLORPHENAMINE": "CHLORPHENIRAMINE", "TIOGUANINE": "THIOGUANINE",
    "INSULIN NOS": "INSULIN",
}
# Non-specific "ingredients" that cannot carry a drug-level signal.
NON_SPECIFIC = re.compile(r"^(?:MINERALS|VITAMINS?|HERBALS|UNSPECIFIED.*|.*\bNOS\b.*|NOS|OTHER CLAIMS|"
                          r"ALL OTHER.*|UNKNOWN.*)$")


def canonical(name: str) -> str:
    """Final ingredient key: synonym-merged, biologic four-letter suffix removed; '' if non-specific."""
    n = re.sub(r"-[A-Z]{4}$", "", name.strip())        # INFLIXIMAB-DYYB -> INFLIXIMAB
    n = SYNONYMS.get(n, n)
    if n.endswith(" HYDROCHLORIDE") and n[:-14] in SYNONYMS.values():
        n = n[:-14]
    return "" if NON_SPECIFIC.match(n) else n
