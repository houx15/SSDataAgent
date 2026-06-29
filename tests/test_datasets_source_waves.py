import pytest

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.schema import load_schema
from ssdataagent.data.transfer import compute_crosswalk, load_source_wave


def test_source_wave_schemas_resolve():
    assert load_schema("gss1994").target_variables  # resolves, non-empty
    assert load_schema("cps1970").target_variables


def test_gss_crosswalk_is_substantial():
    cw = compute_crosswalk(load_schema("gss"), load_schema("gss1994"),
                           load_source_wave("gss1994"), load_real_data("gss"))
    # observed ~25 common variables across GSS 1994/2018; floor well below that.
    assert len(cw) >= 18


def test_cps_crosswalk_is_substantial():
    cw = compute_crosswalk(load_schema("cps"), load_schema("cps1970"),
                           load_source_wave("cps1970"), load_real_data("cps"))
    # observed 12 common variables across CPS-ASEC 1970/1980.
    assert len(cw) >= 10
