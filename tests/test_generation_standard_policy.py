import pytest

from policy.generation_standard import GenerationStandardPolicy, TechnologyStandard


def test_technology_standard_from_csvs_parses_requirements(tmp_path) -> None:
    capacity_path = tmp_path / "capacity.csv"
    capacity_path.write_text("year,alpha\n2030,150\n2035,175\n", encoding="utf-8")

    share_path = tmp_path / "share.csv"
    share_path.write_text("year,alpha\n2030,30\n2035,35\n", encoding="utf-8")

    standard = TechnologyStandard.from_csvs(
        "wind",
        capacity_csv=capacity_path,
        generation_csv=share_path,
        enabled_regions={"alpha", "beta"},
    )
    policy = GenerationStandardPolicy([standard])

    requirements_2030 = policy.requirements_for_year(2030)
    assert len(requirements_2030) == 1
    requirement = requirements_2030[0]
    assert requirement.technology == "wind"
    assert requirement.region == "alpha"
    assert requirement.capacity_mw == 150.0
    assert requirement.generation_share == pytest.approx(0.30, rel=1e-6)

    # Regions without explicit data should produce zero requirements.
    assert standard.capacity_requirement(2030, "beta") == 0.0
    assert standard.generation_share(2030, "beta") == 0.0
