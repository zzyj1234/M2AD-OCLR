"""Print parameter counts for the paper-scale public architecture."""

from oclr import OCLRArchitecture, parameter_audit


def main():
    architecture = OCLRArchitecture()
    audit = parameter_audit(architecture)
    for name, value in audit.items():
        print(f"{name:24s} {value:,}")
    assert audit["paper_total_added"] == 53_760


if __name__ == "__main__":
    main()
