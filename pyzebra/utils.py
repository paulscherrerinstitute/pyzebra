import os

ZEBRA_PROPOSALS_PATHS = [
    f"/afs/psi.ch/project/sinqdata/{year}/zebra/" for year in (2016, 2017, 2018, 2020, 2021, 2022)
]

def find_proposal_path(proposal):
    proposal = proposal.strip()
    if proposal:
        for zebra_proposals_path in ZEBRA_PROPOSALS_PATHS:
            proposal_path = os.path.join(zebra_proposals_path, proposal)
            if os.path.isdir(proposal_path):
                # found it
                break
        else:
            raise ValueError(f"Can not find data for proposal '{proposal}'.")
    else:
        proposal_path = ""

    return proposal_path
