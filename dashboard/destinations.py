"""Shared compact destinations for map markers and traffic-news routes."""

# Compact display wording for marker labels: shorter than official termini
# and matching the ETA embed's shorthand.
DESTINATION_SHORTHAND: dict[str, str] = {
    "tseung kwan o station": "TKO",
    "tseung kwan o": "TKO",
    "choi hung station": "Choi Hung",
    "choi hung station (lung cheung road)": "Choi Hung",
    "diamond hill station bus terminus": "Diamond Hill",
    "diamond hill station": "Diamond Hill",
    "clear water bay bus terminus": "Clear Water Bay",
    "hang hau village": "Hang Hau",
    "hang hau station public transport interchange": "Hang Hau",
    "po lam bus terminus": "Po Lam",
    "h.k.u.s.t. (north)": "HKUST",
    "ngau chi wan bbi - choi hung station": "Choi Hung",
    "mong kok station": "Mong Kok",
    "sai kung": "Sai Kung",
    "kwun tong (circular)": "Kwun Tong",
    "kwun tong(circular)": "Kwun Tong",
}


def short_destination(destination: str) -> str:
    """Compact display wording for a destination string."""
    return DESTINATION_SHORTHAND.get(
        destination.strip().lower(), destination.strip()
    )
