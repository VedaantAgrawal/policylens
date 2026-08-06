"""Static corpus source list.

Every source here was hand-verified (HTTP 200, correct content type) before being
added — see the ingest checkpoint notes in the README. Keeping this list static
and reviewed, rather than crawler-discovered, is what lets `make ingest` be
reproducible and keeps us within "official APIs / bulk data" for each site.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SourceType(str, Enum):
    SEC_10K = "sec_10k"
    NAIC_MODEL_LAW = "naic_model_law"
    STATE_BULLETIN = "state_bulletin"


class Issuer(BaseModel):
    ticker: str
    name: str
    cik: str  # 10-digit, zero-padded


# Publicly traded life insurers only — mutuals like MassMutual and Northwestern
# Mutual don't file 10-Ks with the SEC, so they can't appear here. NAIC model
# laws and state bulletins (below) apply to mutuals too and are how the corpus
# covers that side of the industry.
ISSUERS: list[Issuer] = [
    Issuer(ticker="MET", name="MetLife, Inc.", cik="0001099219"),
    Issuer(ticker="PRU", name="Prudential Financial, Inc.", cik="0001137774"),
    Issuer(ticker="LNC", name="Lincoln National Corp", cik="0000059558"),
    Issuer(ticker="BHF", name="Brighthouse Financial, Inc.", cik="0001685040"),
    Issuer(ticker="CRBG", name="Corebridge Financial, Inc.", cik="0001889539"),
    Issuer(ticker="UNM", name="Unum Group", cik="0000005513"),
    # Added when growing the corpus toward the original 300-800 doc target —
    # CIKs verified live against sec.gov/files/company_tickers.json, not
    # recalled from memory.
    Issuer(ticker="PFG", name="Principal Financial Group Inc", cik="0001126328"),
    Issuer(ticker="GL", name="Globe Life Inc.", cik="0000320335"),
    Issuer(ticker="VOYA", name="Voya Financial, Inc.", cik="0001535929"),
    Issuer(ticker="GNW", name="Genworth Financial Inc", cik="0001276520"),
    Issuer(ticker="FG", name="F&G Annuities & Life, Inc.", cik="0001934850"),
]

# NAIC model law numbers relevant to life insurance, annuities, reinsurance, and
# holding-company/solvency regulation. Confirmed free (no subscription) at
# https://content.naic.org/sites/default/files/model-law-{N}.pdf
NAIC_MODEL_LAWS: dict[str, str] = {
    "205": "Annual Financial Reporting Model Regulation",
    "245": "Annuity Disclosure Model Regulation",
    "250": "Variable Annuity Model Regulation",
    "255": "Modified Guaranteed Annuity Regulation",
    "260": "Model Variable Contract Law",
    "270": "Variable Life Insurance Model Regulation",
    "275": "Suitability in Annuity Transactions Model Regulation",
    "278": "Model Regulation on Senior-Specific Certifications and Professional Designations",
    "305": "Corporate Governance Annual Disclosure Model Act",
    "306": "Corporate Governance Annual Disclosure Model Regulation",
    "312": "Risk-Based Capital (RBC) Model Act",
    "500": "Insurance Holding Company System Model Regulation",
    "505": "Risk Management and Own Risk and Solvency Assessment Model Act",
    "520": "Life and Health Insurance Guaranty Association Model Act",
    "565": "Group Life Insurance Definition and Standard Provisions Model Act",
    "568": "Military Sales Practices Model Regulation",
    "570": "Advertisements of Life Insurance and Annuities Model Regulation",
    "575": "Life and Health Insurance Policy Language Simplification Model Act",
    "580": "Life Insurance Disclosure Model Regulation",
    "582": "Life Insurance Illustrations Model Regulation",
    "585": "Universal Life Insurance Model Regulation",
    "601": "Guidelines on Gifts of Life Insurance to Charitable Institutions",
    "602": "Guidelines on Corporate Owned Life Insurance",
    "605": "Disclosure for Small Face Amount Life Insurance Policies Model Act",
    "613": "Life Insurance and Annuities Replacement Model Regulation",
    "615": "Life Insurance Multiple Policy Model Regulation",
    "620": "Accelerated Benefits Model Regulation",
    "668": "Insurance Data Security Model Law",
    "670": "NAIC Insurance Information and Privacy Protection Model Act",
    "672": "Privacy of Consumer Financial and Health Information Regulation",
    "673": "Standards for Safeguarding Customer Information Model Regulation",
    "692": "Interstate Insurance Product Regulation Compact",
    "697": "Viatical Settlements Model Act",
    "785": "Credit for Reinsurance Model Law",
    "786": "Credit for Reinsurance Model Regulation",
    "787": "Term and Universal Life Insurance Reserve Financing Model Regulation",
    "791": "Life and Health Reinsurance Agreements Model Regulation",
    "805": "Standard Nonforfeiture Law for Individual Deferred Annuities",
    "806": "Annuity Nonforfeiture Model Regulation",
    "808": "Standard Nonforfeiture Law for Life Insurance",
    "815": "Model Regulation Permitting Recognition of Preferred Mortality Tables",
    "820": "Standard Valuation Law",
    "822": "Actuarial Opinion and Memorandum Regulation",
    "830": "Valuation of Life Insurance Policies Model Regulation",
    "880": "Unfair Trade Practices Act",
    "887": "Model Regulation on Unfair Discrimination in Life and Health Insurance",
    "896": "Unfair Discrimination Against Subjects of Abuse in Life Insurance Model Act",
    "900": "Unfair Claims Settlement Practices Act",
    "903": "Unfair Life, Accident and Health Claims Settlement Practices Model Regulation",
    # Second wave, added when growing the corpus toward the original 300-800
    # doc target: producer licensing/market conduct, investments, receivership,
    # reinsurance completions, mortality tables, fraud, and unfair-discrimination
    # completions. Deliberately excludes 640/641/642/643 (Long-Term Care) and
    # 650/651/660 (Medicare Supplement) — those are the golden set's
    # deliberate-exclusion trap questions (q079 and the original LTC question);
    # adding them would silently turn a designed-unanswerable question answerable.
    "218": "Producer Licensing Model Act",
    "225": "Managing General Agents Act",
    "228": "Public Adjuster Licensing Model Act",
    "230": "Title Insurance Agent Model Act",
    "235": "Interest-Indexed Annuity Contracts Model Regulation",
    "240": "Charitable Gift Annuities Model Act",
    "241": "Charitable Gift Annuities Exemption Model Act",
    "280": "Investments of Insurers Model Act (Defined Limits Version)",
    "282": "Derivative Instruments Model Regulation",
    "283": "Investments of Insurers Model Act (Defined Standards Version)",
    "285": "Disclosure of Material Transactions Model Act",
    "290": "Protected Cell Company Model Act",
    "295": "Model Act on Custodial Agreements and the Use of Clearing Corporations",
    "298": "Model Regulation on Custodial Agreements and the Use of Clearing Corporations",
    "315": "Risk-Based Capital (RBC) for Health Organizations Model Act",
    "540": "Property and Casualty Insurance Guaranty Association Model Act",
    "555": "Insurer Receivership Model Act",
    "558": "Administrative Supervision Model Act",
    "590": "Model Policy Loan Interest Rate Bill",
    "628": "Title Insurers Model Act",
    "630": "Mortgage Guaranty Insurance Model Act",
    "631": "Real Property Lender-Placed Insurance Model Act",
    "632": "Travel Insurance Model Act",
    "633": "Pet Insurance Model Act",
    "675": "Uniform Fraternal Code",
    "680": "Insurance Fraud Prevention Model Act",
    "685": "Service Contracts Model Act",
    "693": "Market Conduct Surveillance Model Law",
    "695": "Synthetic Guaranteed Investment Contracts Model Regulation",
    "698": "Viatical Settlements Model Regulation",
    "700": "Insurance Statute—Atomic Energy Exclusion",
    "705": "Model Risk Retention Act",
    "745": "Property and Casualty Actuarial Opinion Model Law",
    "789": "Special Purpose Reinsurance Vehicle Model Act",
    "790": "Reinsurance Intermediary Model Act",
    "803": "Assumption Reinsurance Model Act",
    "811": "NAIC Procedure for Permitting Same Minimum Nonforfeiture Standards for Men and Women",
    "814": "NAIC Model Rule Permitting Smoker/Nonsmoker Mortality Tables",
    "817": "Preneed Life Insurance Minimum Standards Model Regulation",
    "818": "Determining Reserve Liabilities for Credit Life Insurance Model Regulation",
    "821": "NAIC Model Rule for Recognizing a New Annuity Mortality Table",
    "840": "Unauthorized Insurers False Advertising Process Act",
    "850": "Unauthorized Insurers Process Act",
    "870": "Nonadmitted Insurance Model Act",
    "884": "Model Regulation for Complaint Records Pursuant to the Unfair Trade Practices Act",
    "888": "Model Regulation on Unfair Discrimination on the Basis of Blindness or Partial Blindness",
    "890": "Unauthorized Transaction of Criminal Model Act",
    "895": "Unfair Discrimination Against Subjects of Abuse in Health Benefit Plans Model Act",
    "897": "Unfair Discrimination Against Subjects of Abuse in Disability Income Insurance Model Act",
    "898": "Unfair Discrimination Against Subjects of Abuse in Property and Casualty Insurance",
    "902": "Unfair Property/Casualty Claims Settlement Practices Model Regulation",
    "910": "Market Conduct Record Retention and Production Model Regulation",
    "915": "Improper Termination Practices Model Act",
    "920": "Home Service Disclosure Model Act",
}

# State DOI/DFS sources. Massachusetts was dropped: mass.gov returns 403 to
# automated requests (bot-protection, not a robots.txt disallow) and bypassing
# that would be scraping evasion, not "official API/bulk data" access — so we
# use Texas instead, confirmed reachable.
STATE_BULLETIN_LISTING_PAGES: dict[str, str] = {
    "NY": "https://www.dfs.ny.gov/industry_guidance/circular_letters",
    "CA": "https://www.insurance.ca.gov/0250-insurers/0300-insurers/0200-bulletins/bulletin-notices-commiss-opinion/bulletins.cfm",
    "TX": "https://www.tdi.texas.gov/bulletins/2026/index.html",
}

# Cap per state so the corpus stays balanced across jurisdictions rather than
# whichever site happens to list the most documents. Raised from 20 when
# growing the corpus toward the original 300-800 doc target.
MAX_BULLETINS_PER_STATE = 35
