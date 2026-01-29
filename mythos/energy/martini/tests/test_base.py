import pytest
from mythos.energy.martini.base import MartiniEnergyConfiguration


class TestMartiniEnergyConfiguration:
    def test_parameter_loading(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0, **{"param-2": 2.0})
        assert conf["param_1"] == 1.0
        assert conf["param-2"] == 2.0

    def test_opt_params(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0, **{"param-2": 2.0})
        assert conf.opt_params == {"param_1": 1.0, "param-2": 2.0}

    def test_coupling_parameters(self):
        conf = MartiniEnergyConfiguration(
            couplings={"proxy_param": ["param_a", "param_b"]},
            proxy_param = 3.0,
            param_c = 4.0,
        )
        assert "proxy_param" in conf
        assert "proxy_param" not in conf.params
        assert conf["param_a"] == 3.0
        assert conf["param_b"] == 3.0
        assert conf["param_c"] == 4.0

        assert conf.opt_params == {"proxy_param": 3.0, "param_c": 4.0}

    def test_coupling_parameter_conflict(self):
        with pytest.raises(ValueError, match="more than one coupling"):
            MartiniEnergyConfiguration(
                couplings={"proxy_param": ["param_a", "param_b"], "other_proxy": ["param_b"]},
                proxy_param = 3.0,
                other_proxy = 4.0,
            )

    def test_keyerror_on_missing_param(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0)
        with pytest.raises(KeyError, match="Parameter 'param_2' not found"):
            _ = conf["param_2"]


